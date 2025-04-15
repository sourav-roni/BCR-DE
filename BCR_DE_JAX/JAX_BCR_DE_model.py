from flax import nnx
import jax
import jax.numpy as jnp 
import diffrax

from jaxwt import wavedec, waverec
import pdb
import cr.wavelets as wt
import numpy as np
from functools import partial


class cheapPartiallyUnsharedConv1d(nnx.Module):
	'''
	This variant of Partially Unshared Convolution (PUC) layer is a bit different from PartiallyUnsharedConv1d
	The difference is in how the patches are laid over the sequence
	and in the end point computation. However this is much faster and is the recommended one
	'''
	def __init__(self, in_channels
			  , out_channels
			  , output_size
			  , kernel_size
			  , stride
			  , one_side_pad_length
			  , num_sparse_LC
			  , dim_d
			  , dim_k
			  , conv_bias
			  , level
			  , nk_LC
			  , rngs):
		super(cheapPartiallyUnsharedConv1d, self).__init__()
		self.num_sparse_LC = num_sparse_LC

		weight_shape = (dim_d, dim_k, out_channels, in_channels, num_sparse_LC, 1, kernel_size)

		initializers = nnx.initializers.kaiming_normal()
		self.weight = nnx.Param(initializers(rngs.params(), weight_shape))

		self.conv_bias = conv_bias
		if self.conv_bias:
			conv_bias_shape = (dim_d, dim_k, out_channels, num_sparse_LC, 1)
			self.conv_bias_param = nnx.Param(jax.random.uniform(rngs.params(), conv_bias_shape))
		
		self.kernel_size = kernel_size
		self.stride = stride
		self.one_side_pad_length = one_side_pad_length


	def __call__(self, x):
		n, d, k, t, l = x.shape    # ~ [bs, dim_d, dim_k, 2, len]
	          # ~ [bs, dim_d, dim_k, 2, len']
		x = jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, 0), (int(self.one_side_pad_length), int(self.one_side_pad_length))))

		# Combine batch dimensions and the "channel" dimensions (here: two)
		# so that we can treat the last dimension (length) as the spatial dimension.
		x_reshaped = jnp.reshape(x, (n * d * k * t, x.shape[-1], 1))
		
		# Extract 1D patches. The conv_general_dilated_patches function expects:
		#   - Input in "NWC" format (batch, spatial length, channels)
		#   - filter_shape as a tuple (kernel_size,)
		#   - window_strides as a tuple (stride,)
		#   - dimension_numbers to specify the ordering.
		patches = jax.lax.conv_general_dilated_patches(
			x_reshaped,
			filter_shape=(self.kernel_size,),
			window_strides=(self.stride,),
			padding='VALID',
			dimension_numbers=("NWC", "WIO", "NWC")
		)
		# The shape of patches is: (bs * dim_d * dim_k * two, L_out, 1, kernel_size)
		# Remove the channel dimension (of size 1):
		# patches = jnp.squeeze(patches, axis=2)  # shape: (bs * dim_d * dim_k * two, L_out, kernel_size)
		
		# Compute L_out from the patches shape
		L_out = patches.shape[1]
		
		# Reshape back to the original batch dimensions with an extra axis for the unfolded windows:
		x = jnp.reshape(patches, (n, d, k, t, L_out, self.kernel_size))

		dim_d, dim_k, out_channels, in_channels, num_sparse_LC, _, kernel_size = self.weight.shape       # input and output channels are both 1 for all practical purpose
		weight = jnp.tile(self.weight, (1, 1, 1, 1, 1, int(np.floor(l / num_sparse_LC)), 1))                 # ~ [dim_d, dim_k, out_channels, in_channels, num_sparse_LC, length_of_each_LC, kernel_size]
		weight = jnp.reshape(weight, (dim_d, dim_k, out_channels, in_channels, -1, kernel_size))                # ~ [dim_d, dim_k, out_channels, in_channels, num_sparse_Lc*length_of_each_LC, kernel_size]
		remainder = x.shape[-2] - weight.shape[-2]
		last = self.weight[:, :, :, :, -1, :, :]
		weight = jnp.concatenate([weight] + [last] * remainder, axis = -2)

		if self.conv_bias:
			conv_bias_param = jnp.tile(self.conv_bias_param, (1, 1, 1, 1, int(np.floor(l / num_sparse_LC))))
			conv_bias_param = jnp.reshape(conv_bias_param , (dim_d, dim_k, out_channels, -1))
			last_conv_bias = self.conv_bias_param[:, :, :, -1, :]
			conv_bias_param = jnp.concatenate([conv_bias_param] + [last_conv_bias]*remainder, axis=-1)

		# Sum in in_channel and kernel_size dims
		out = jnp.einsum("dkoilf,bdkilf->bdkol", weight, x)                # ~ [bs, dim_d, dim_k, 2, len]
		if self.conv_bias:
			out = out + conv_bias_param
		return out

class myDense(nnx.Module):
	'''
	Dense layer using einsum, for multiple dimension
	'''
	def __init__(self
			  , dim_d
			  , dim_k
			  , dense_dim
			  , rngs 
			  , bias=False):
		super(myDense, self).__init__()
		dLayer_shape = (dim_d, dim_k, dense_dim, dense_dim)
		initializers = nnx.initializers.kaiming_normal()
		self.dLayer = nnx.Param(initializers(rngs.params(), dLayer_shape))

		self.bias = bias
		if bias:
			self.d_bias = nnx.Param(jax.random.uniform(rngs.params(), (dim_d, dim_k, self.dLayer.shape[-2])))

	def __call__(self, x):
		transform_x = jnp.einsum('dktq,bdkq->bdkt', self.dLayer, x)
		if self.bias:
			transform_x = transform_x + self.d_bias
		return transform_x
	

class CDE_BCR(nnx.Module):
	'''
	Main model for BCR_DE
	'''
	def __init__(self
			  , time_step
			  , wave
			  , D
			  , D_out
			  , d
			  , k
			  , original_length
			  , num_classes
			  , nonlinearity
			  , n_levels
			  , K_dense
			  , K_LC
			  , nb
			  , num_sparse_LC
			  , use_cheap_sparse_LC
			  , interpol
			  , conv_bias
			  , rngs
			  , predict=False
			  , masked_modelling=False):
		super(CDE_BCR, self).__init__()
		print("Efficient model")
		self.wave = wave
		self.dim_D = D
		self.dim_D_out = D_out
		self.dim_d = d
		self.dim_k = k
		self.time_step = time_step
		self.original_length = original_length
		self.n_levels = n_levels
		self.K_dense = K_dense
		self.K_LC = K_LC
		self.nb = nb
		self.num_classes = num_classes
		self.num_sparse_LC = num_sparse_LC
		self.interpol = interpol
		self.conv_bias = conv_bias


		if nonlinearity == 'relu':
			self.nl_act = nnx.relu
		elif nonlinearity == 'tanh':
			self.nl_act = nnx.tanh
		else:
			print("Invalid activation function")
			exit(0)
		
		self.g_layer = nnx.Linear(self.dim_D, self.dim_d, use_bias=False, rngs = rngs)
		self.h_layer = nnx.Linear(self.dim_d, self.dim_D * self.dim_k, use_bias=False, rngs = rngs)
	
  		# Forward pass to get dimension of dense layer
		x = jnp.array(jax.random.uniform(rngs.params(), (4, 1, self.original_length)))
		self.dense_dim, self.output_sizes = self.fake_pass_get_dim(x)
		self.output_sizes.reverse()
		print("Ouput sizes: ", self.output_sizes)
		self.dk_pair_dense_weight = nnx.Sequential(*[
			myDense(self.dim_d, self.dim_k, self.dense_dim, rngs=rngs, bias=False)
			for _ in range(self.K_dense)
		])
		self.dk_pair_LC_einsum = []
		for i in range(self.n_levels):
			level_LCs = []
			for j in range(0, self.K_LC):
				if use_cheap_sparse_LC:
					LC_layer = cheapPartiallyUnsharedConv1d(in_channels=2*1, out_channels=2*1, output_size=self.output_sizes[-i-1], kernel_size=self.nb, stride=1, 
											one_side_pad_length=np.floor(nb/2), num_sparse_LC=self.num_sparse_LC, dim_d = self.dim_d, dim_k = self.dim_k, conv_bias=True, 
											level=i, nk_LC=j, rngs = rngs)

				level_LCs.append(LC_layer)
			self.dk_pair_LC_einsum.append(level_LCs)
		self.reverse_g_layer = nnx.Linear(self.dim_d, self.dim_D_out, use_bias=False, rngs = rngs)
		if predict:
			print("Adding final prediction layer")
			self.predict = True
			self.prediction_layer = nnx.Sequential(
						nnx.Linear(self.dim_D_out, 20, use_bias=True, rngs = rngs),
						nnx.relu,
						nnx.Linear(20, self.num_classes, use_bias=True, rngs = rngs),
						)
		else:
			self.predict = False
		self.masked_modelling = masked_modelling
	@partial(jax.vmap, in_axes=(None, 0, None))
	def deriv_interpolate_linear(self, ys, t):
		return diffrax.LinearInterpolation(self.time_step, ys).derivative(t)
	
	@partial(jax.vmap, in_axes=(None, 0, None))
	def deriv_interpolate_spline(self, ys, t):
		channels = ys.shape[-1] // 4
		a = ys[..., :channels]            # Contains columns 0 to 6  -> shape (17893, 7)
		b = ys[..., channels:2 * channels]  # Contains columns 7 to 13 -> shape (17893, 7)
		c = ys[..., 2 * channels:3 * channels]  # Contains columns 14 to 20 -> shape (17893, 7)
		d = ys[..., 3 * channels:]    
		return diffrax.CubicInterpolation(self.time_step, [a, b, c, d]).derivative(t)
	def __call__(self, seq, coeffs, time):
		batch_size = seq.shape[0]
		sequence_length = seq.shape[1]
		if self.interpol == 'linear':			
			der_X = jnp.expand_dims(jax.vmap(lambda x : self.deriv_interpolate_linear(coeffs, x))(time).swapaxes(0, 1), -1)
		elif self.interpol =='spline':
			der_X = jnp.expand_dims(jax.vmap(lambda x : self.deriv_interpolate_spline(coeffs, x))(time).swapaxes(0, 1), -1)
		else:
			print('Invalid interpolation type')
			exit(0)
		z = self.nl_act(self.g_layer(seq))
		h_of_z = jnp.reshape(self.nl_act(self.h_layer(z)), (batch_size, sequence_length, self.dim_D, self.dim_k))
		transpose_hz = jnp.permute_dims(h_of_z, (0, 1, 3, 2))
		v = jnp.einsum('blkD,blDo->blko', transpose_hz, der_X).squeeze(-1)
		v = v.transpose(0, 2, 1)

		current_approx = v
		all_detail = []
		all_approx = []
		for l in range(self.n_levels):
			current_approx, current_detail = wt.wavedec(current_approx, self.wave, mode = 'periodization', level = 1)  # [bs, chna, len] 
			all_detail.append(current_detail)
			all_approx.append(current_approx)

		last_approx = all_approx[-1]
		dth_corase_approx = []

		current_approx = jnp.tile(last_approx[:, None, :, :], (1, self.dim_d, 1, 1))
		current_approx = self.dk_pair_dense_weight(current_approx)
		dth_corase_approx = current_approx
		dth_current_approx = dth_corase_approx

		if self.masked_modelling:
			masked_modelling_approx = None
			masked_modelling_detail = []
		for l in reversed(range(self.n_levels)):
			prev_detail_l = all_detail[l]
			prev_approx_l = all_approx[l]

			chi_l = jnp.tile(jnp.expand_dims(jnp.stack([prev_detail_l, prev_approx_l], axis = 2), 1), (1, self.dim_d, 1, 1, 1)) # ~ [bs, dim_d, dim_k, 2, len]

			for k in range(self.K_LC):
				chi_l = self.nl_act(self.dk_pair_LC_einsum[l][k](chi_l))
			current_approx = dth_current_approx
			current_approx = self.shape_correction(chi_l, current_approx)
			padded_current_approx = jnp.stack([jnp.zeros_like(current_approx), current_approx], axis = -2)
			X_l = padded_current_approx + chi_l

			bs, dd, kd, _, length = X_l.shape
			X_l_detail = jnp.reshape(X_l[:, :, :, 0, :], (bs * dd * kd, 1, length))
			X_l_approx = jnp.reshape(X_l[:, :, :, 1, :], (bs * dd * kd, 1, length))
			if self.masked_modelling:
				if l==self.n_levels-1:
					masked_modelling_approx = X_l_approx
				masked_modelling_detail.append(X_l_detail)
			current_next_approx = wt.waverec([X_l_approx, X_l_detail], self.wave, mode = 'periodization')
			current_next_approx = jnp.reshape(current_next_approx, (bs, dd, kd, -1))

			dth_current_approx = current_next_approx
		
		dth_current_approx = jnp.sum(dth_current_approx, axis=2)
		current_approx = dth_current_approx  # ~ [bs, dim_D, len]

		U = self.reverse_g_layer(jnp.permute_dims(current_approx, (0, 2, 1)))

		if self.masked_modelling:
			return U, masked_modelling_approx, masked_modelling_detail

		if self.predict:
			last_observable = U[:, -1, :]
			prediction = self.prediction_layer(last_observable)
			return U, prediction
		else:
			return U

	def shape_correction(self, chi_l, current_approx):
		if chi_l.shape[-1] == current_approx.shape[-1]:
			return current_approx
		else:
			# Calculate how many zeros to pad on the left of the last dimension.
			left_diff = chi_l.shape[-1] - current_approx.shape[-1]
			current_approx = current_approx[..., -left_diff:]
			return current_approx

	def fake_pass_get_dim(self, v):
		current_approx = v
		output_sizes = []
		for l in range(self.n_levels):
			result = wt.wavedec(current_approx, self.wave, mode = 'periodization', level = 1)
			current_approx, current_detail = result[0], result[1]
			assert current_approx.shape[2] == current_detail.shape[2]
			output_sizes.append(current_approx.shape[2])
		return current_approx.shape[2], output_sizes