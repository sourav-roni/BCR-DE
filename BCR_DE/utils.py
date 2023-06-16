'''
Thie file is for all sharable utility function
'''
import torch 
import numpy as np 
import os, sys
import pdb
import pickle
import random
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from tse_loader import load_from_tsfile_to_dataframe, process_data
import pandas as pd

def get_coupled_data(nsample, t, seq_len):
	'''
	Simulate toy coupled differential equation 
	'''
	one = []
	two = []
	for i in range(nsample):
		# scale = np.random.uniform(0,0.1)
		scale = 0
		y = 3*np.exp(t) - np.exp(2*t) + scale*np.random.randn(seq_len)
		x = 2*np.exp(t) - np.exp(2*t) + scale*np.random.randn(seq_len)
		one.append(torch.from_numpy(y).unsqueeze(0))
		two.append(torch.from_numpy(x).unsqueeze(0))
	one = torch.cat(one).unsqueeze(1)
	two = torch.cat(two).unsqueeze(1)
	return one, two

def get_VDP_data(nsample, t, noise=False):
	'''
	Simulate Van-der Pol equation 
	'''
	noise_scale = 0.2
	# perform simulation
	hare = []
	lynx = []
	for i in range(nsample):
		hare_ic = np.random.randint(0,6)
		lynx_ic = np.random.randint(0,6)
		IC = [hare_ic, lynx_ic]                    # initial conditions for H and L
		sol = odeint(VDP_deriv,IC,t)        # compute solution
		H,L = sol.transpose()           # unpack solution 
		if noise:
			H = H + noise_scale*np.random.randn(H.shape[0])
			L = L + noise_scale*np.random.randn(L.shape[0])
		hare.append(torch.from_numpy(H).unsqueeze(0))
		lynx.append(torch.from_numpy(L).unsqueeze(0))
	hare = torch.cat(hare).unsqueeze(1)
	lynx = torch.cat(lynx).unsqueeze(1)
	return hare, lynx

# differential equations
def VDP_deriv(X,t):
	'''
	Derivative for Van-der Pol equation
	'''
	# mu = 2.0
	mu = 6.0
	H,L = X
	dH =  mu*(H - (1/3)*H*H*H - L)
	dL = H/mu
	return [dH,dL]

def get_Benzene_data(dataname, seq_length):
	'''
	Get Benzene data from TSER dataset
	'''
	module = 'AirQuality'
	train_file = '../data/TSER/' + dataname + "_TRAIN.ts"
	test_file = '../data/TSER/' + dataname + "_TEST.ts"
	X_train, y_train = load_from_tsfile_to_dataframe(train_file)
	X_test, y_test = load_from_tsfile_to_dataframe(test_file)
	print("[{}] X_train: {}".format(module, X_train.shape))
	print("[{}] X_test: {}".format(module, X_test.shape))

	# in case there are different lengths in the dataset, we need to consider that.
	# assume that all the dimensions are the same length
	print("[{}] Finding minimum length".format(module))
	min_len = np.inf
	for i in range(len(X_train)):
		x = X_train.iloc[i, :]
		all_len = [len(y) for y in x]
		min_len = min(min(all_len), min_len)
	for i in range(len(X_test)):
		x = X_test.iloc[i, :]
		all_len = [len(y) for y in x]
		min_len = min(min(all_len), min_len)
	print("[{}] Minimum length: {}".format(module, min_len))

	# process the data into numpy array with (n_examples, n_timestep, n_dim)
	print("[{}] Reshaping data".format(module))
	x_train = torch.from_numpy(process_data(X_train, min_len=min_len))
	x_test = torch.from_numpy(process_data(X_test, min_len=min_len))
	val_split = int(x_test.shape[0]/2)
	assert seq_length == x_test.shape[1]

	train_x = x_train[:, :, 5].unsqueeze(1)
	train_y = x_train[:, :, 6].unsqueeze(1)
	valid_x = x_test[:val_split, :, 5].unsqueeze(1)
	valid_y = x_test[:val_split, :, 6].unsqueeze(1)
	test_x = x_test[val_split:, :, 5].unsqueeze(1)
	test_y = x_test[val_split:, :, 6].unsqueeze(1)

	t = np.linspace(0, 10, seq_length)
	t = torch.from_numpy(t).double()

	return train_x, train_y, valid_x, valid_y, test_x, test_y, t


def get_HH_data_ps_2_act():
	'''
	Load data simulated from Hodgkin-Huxely model, this returns the current and potential as input
	for the model to predict the different Sodium and Potassium activation
	'''
	train = torch.load('../data/HH/train.pt')
	valid = torch.load('../data/HH/valid.pt')
	test = torch.load('../data/HH/test.pt')
	print("TRain:",train.shape)
	print("Valid:",valid.shape)
	print("Test:",test.shape)

	train_x = train[:, 1:3, :]
	train_y = train[:, 3:6, :]
	valid_x = valid[:, 1:3, :]
	valid_y = valid[:, 3:6, :]
	test_x = test[:, 1:3, :]
	test_y = test[:, 3:6, :]

	time = train[0, 0, :]
	return train_x, train_y, valid_x, valid_y, test_x, test_y, time

def f(state, t):
	'''
	Derivative for chaotic Lorenz equation
	'''
	rho = 28.0
	sigma = 10.0
	beta = 8.0 / 3.0
	x, y, z = state  # Unpack the state vector
	return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives

def get_Lorenz_data(n_sample, t, noise=False):
	'''
	Simulate chaotic Lorenz system
	'''
	noise_scale = 1.0
	simulation = []
	for i in range(n_sample):
		state0 = [np.random.uniform(0,1), np.random.uniform(0,1), np.random.uniform(0,1)]
		states = odeint(f, state0, t)
		if noise:
			# states = states + noise_scale*np.random.randn(states.shape[0], states.shape[1])
			states[:,-1] = states[:,-1] + noise_scale*np.random.randn(states.shape[0])
		simulation.append(states)
	simulation = torch.from_numpy(np.stack(simulation)).transpose(1,2)
	xy = simulation[:, :2, :]
	z = simulation[:, -1, :].unsqueeze(1)
	return xy, z

def get_Hare_Lynx_data(nsample, t, noise=False):
	'''
	Simulated Lotka-Volterra (predator-prey) system
	'''
	noise_scale = 4.0
	# perform simulation
	hare = []
	lynx = []
	for i in range(nsample):
		hare_ic = np.random.randint(1,100)
		lynx_ic = np.random.randint(1,100)
		IC = [hare_ic, lynx_ic]                    # initial conditions for H and L
		sol = odeint(deriv,IC,t)        # compute solution
		H,L = sol.transpose()           # unpack solution 
		if noise:
			H = H + noise_scale*np.random.randn(H.shape[0])
			L = L + noise_scale*np.random.randn(L.shape[0])
		hare.append(torch.from_numpy(H).unsqueeze(0))
		lynx.append(torch.from_numpy(L).unsqueeze(0))
	hare = torch.cat(hare).unsqueeze(1)
	lynx = torch.cat(lynx).unsqueeze(1)
	return hare, lynx

# differential equations
def deriv(X,t):
	'''
	Derivative for Lotka-Volterra system
	'''
	a = 3.2
	b = 0.6
	c = 50
	d = 0.56
	k = 125
	r = 1.6
	H,L = X
	dH =  r*H*(1-H/k) - a*H*L/(c+H)
	dL = b*a*H*L/(c+H) - d*L
	return [dH,dL]

def save_data(true, pred, epoch, data_type, exp_dir):
	dirname = exp_dir+'/model_prediction/'
	if not os.path.exists(dirname):
		os.makedirs(dirname)
	fname = dirname+data_type+'_'+str(epoch)
	torch.save(true, fname+'_true.pt')
	torch.save(pred, fname+'_pred.pt')
	return

def save_data_couple(true, pred, ip_se, epoch, data_type, exp_dir):
	dirname = exp_dir+'/model_prediction/'
	if not os.path.exists(dirname):
		os.makedirs(dirname)
	fname = dirname+data_type+'_'+str(epoch)
	torch.save(true, fname+'_true.pt')
	torch.save(pred, fname+'_pred.pt')
	torch.save(ip_se, fname+'_ip_se.pt')
	return

def save_mask_data(masked, full, pred, mask_indices, epoch, data_type, exp_dir):
	dirname = exp_dir+'/model_prediction/'
	if not os.path.exists(dirname):
		os.makedirs(dirname)
	fname = dirname+data_type+'_'+str(epoch)
	torch.save(masked, fname+'_masked.pt')
	torch.save(full, fname+'_full.pt')
	torch.save(pred, fname+'_pred.pt')
	torch.save(mask_indices, fname+'_mask_indices.pt')
	return

def save_fixed_mask_data(masked, full, pred, epoch, data_type, exp_dir):
	dirname = exp_dir+'/model_prediction/'
	if not os.path.exists(dirname):
		os.makedirs(dirname)
	fname = dirname+data_type+'_'+str(epoch)
	torch.save(masked, fname+'_masked.pt')
	torch.save(full, fname+'_full.pt')
	torch.save(pred, fname+'_pred.pt')
	return

def plot_and_save(exp, all_losses, only_one=False):
	loss_file = exp+'/all_losses.pkl'
	with open(loss_file, 'wb') as f:
		pickle.dump(all_losses, f)

	plt.plot(all_losses['train_total_loss'], label='Training')
	plt.plot(all_losses['valid_total_loss'], label='Validation')
	plt.plot(all_losses['test_total_loss'], label='Test')
	plt.legend()
	plt.xlabel("Epoch")
	plt.ylabel("Total loss")
	plt.savefig(exp+'/TotalLoss.png')
	plt.cla()
	plt.clf()
	plt.close()

	if only_one:
		return

	plt.plot(all_losses['train_recon_loss'], label='Training')
	plt.plot(all_losses['valid_recon_loss'], label='Validation')
	plt.plot(all_losses['test_recon_loss'], label='Test')
	plt.legend()
	plt.xlabel("Epoch")
	plt.ylabel("Reconstruction loss")
	plt.savefig(exp+'/ReconstructionLoss.png')
	plt.cla()
	plt.clf()
	plt.close()

	plt.plot(all_losses['train_pred_loss'], label='Training')
	plt.plot(all_losses['valid_pred_loss'], label='Validation')
	plt.plot(all_losses['test_pred_loss'], label='Test')
	plt.legend()
	plt.xlabel("Epoch")
	plt.ylabel("Prediction loss")
	plt.savefig(exp+'/PredictionLoss.png')
	plt.cla()
	plt.clf()
	plt.close()
	return

def plot_and_save_l1(exp, all_losses, only_one=False):
	loss_file = exp+'/all_losses.pkl'
	with open(loss_file, 'wb') as f:
		pickle.dump(all_losses, f)

	plt.plot(all_losses['train_total_loss'], label='Training')
	plt.plot(all_losses['valid_total_loss'], label='Validation')
	plt.plot(all_losses['test_total_loss'], label='Test')
	plt.legend()
	plt.xlabel("Epoch")
	plt.ylabel("Total loss")
	plt.savefig(exp+'/TotalLoss.png')
	plt.cla()
	plt.clf()
	plt.close()

	if only_one:
		return

	plt.plot(all_losses['train_recon_loss'], label='Training')
	plt.plot(all_losses['valid_recon_loss'], label='Validation')
	plt.plot(all_losses['test_recon_loss'], label='Test')
	plt.legend()
	plt.xlabel("Epoch")
	plt.ylabel("Reconstruction loss")
	plt.savefig(exp+'/ReconstructionLoss.png')
	plt.cla()
	plt.clf()
	plt.close()

	plt.plot(all_losses['train_l1_loss'], label='Training')
	plt.plot(all_losses['valid_l1_loss'], label='Validation')
	plt.plot(all_losses['test_l1_loss'], label='Test')
	plt.legend()
	plt.xlabel("Epoch")
	plt.ylabel("Prediction loss")
	plt.savefig(exp+'/PredictionLoss.png')
	plt.cla()
	plt.clf()
	plt.close()
	return

def plot_and_save_L1_TV(exp, all_losses):
	loss_file = exp+'/all_losses.pkl'
	with open(loss_file, 'wb') as f:
		pickle.dump(all_losses, f)

	plt.plot(all_losses['train_recon_loss'], label='Training')
	plt.plot(all_losses['valid_recon_loss'], label='Validation')
	plt.plot(all_losses['test_recon_loss'], label='Test')
	plt.legend()
	plt.xlabel("Epoch")
	plt.ylabel("Reconstruction loss")
	plt.savefig(exp+'/ReconLoss.png')
	plt.cla()
	plt.clf()
	plt.close()

	plt.plot(all_losses['train_L1_loss'], label='Training')
	plt.plot(all_losses['valid_L1_loss'], label='Validation')
	plt.plot(all_losses['test_L1_loss'], label='Test')
	plt.legend()
	plt.xlabel("Epoch")
	plt.ylabel("L1 loss")
	plt.savefig(exp+'/L1Loss.png')
	plt.cla()
	plt.clf()
	plt.close()

	plt.plot(all_losses['train_TV_loss'], label='Training')
	plt.plot(all_losses['valid_TV_loss'], label='Validation')
	plt.plot(all_losses['test_TV_loss'], label='Test')
	plt.legend()
	plt.xlabel("Epoch")
	plt.ylabel("TV loss")
	plt.savefig(exp+'/TVLoss.png')
	plt.cla()
	plt.clf()
	plt.close()
	return

def manual_set_seed(seed):
	print("Setting all seeds to: ", seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def load_PPG(noise=False, noise_scale=0.01):
	'''
	Load PPG data
	'''
	train_x, val_x, test_x, train_y, val_y, test_y = self_save_file()
	print("Self save file loaded")
	
	time_step = np.arange(0, 1, 1/train_x.shape[1])

	train_x = torch.from_numpy(train_x[:,:,1:2]).transpose(1,2)
	val_x = torch.from_numpy(val_x[:,:,1:2]).transpose(1,2)
	test_x = torch.from_numpy(test_x[:,:,1:2]).transpose(1,2)

	if noise:
		print("Addding standard Gaussian noise for denoising autoencoding")
		train_y = train_x.clone()
		train_x = train_x + noise_scale*torch.randn(train_x.shape)
		val_y = val_x.clone()
		val_x = val_x + noise_scale*torch.randn(val_x.shape)
		test_y = test_x.clone()
		test_x = test_x + noise_scale*torch.randn(test_x.shape)
	else:
		pass

	print("Training size:", train_x.shape, train_y.shape)
	print("Validation size:", val_x.shape, val_y.shape)
	print("Testing size:", test_x.shape, test_y.shape)
	train = {}
	train['x'] = train_x
	train['y'] = train_y
	valid = {}
	valid['x'] = val_x
	valid['y'] = val_y
	test = {}
	test['x'] = test_x
	test['y'] = test_y
	return train, valid, test, torch.from_numpy(time_step)

def load_ECG(noise=False, noise_scale=0.01):
	'''
	Load ECG data
	'''
	train_x, val_x, test_x, train_y, val_y, test_y = self_save_file()
	print("Self save file loaded")

	time_step = np.arange(0, 1, 1/train_x.shape[1])

	train_x = torch.from_numpy(train_x[:,:,2:]).transpose(1,2)
	val_x = torch.from_numpy(val_x[:,:,2:]).transpose(1,2)
	test_x = torch.from_numpy(test_x[:,:,2:]).transpose(1,2)

	if noise:
		print("Addding standard Gaussian noise for denoising autoencoding")
		train_y = train_x.clone()
		train_x = train_x + noise_scale*torch.randn(train_x.shape)
		val_y = val_x.clone()
		val_x = val_x + noise_scale*torch.randn(val_x.shape)
		test_y = test_x.clone()
		test_x = test_x + noise_scale*torch.randn(test_x.shape)
	else:
		pass

	print("Training size:", train_x.shape, train_y.shape)
	print("Validation size:", val_x.shape, val_y.shape)
	print("Testing size:", test_x.shape, test_y.shape)
	train = {}
	train['x'] = train_x
	train['y'] = train_y
	valid = {}
	valid['x'] = val_x
	valid['y'] = val_y
	test = {}
	test['x'] = test_x
	test['y'] = test_y
	return train, valid, test, torch.from_numpy(time_step)

def self_save_file():
	train_x = np.load('../data/se2se/RR/train_seq.npy')
	val_x = np.load('../data/se2se/RR/val_seq.npy')
	test_x = np.load('../data/se2se/RR/test_seq.npy')
	train_y = np.load('../data/se2se/RR/train_pred.npy')
	val_y = np.load('../data/se2se/RR/val_pred.npy')
	test_y = np.load('../data/se2se/RR/test_pred.npy')
	return train_x, val_x, test_x, train_y, val_y, test_y