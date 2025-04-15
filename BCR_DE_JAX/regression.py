'''
Main file for running regression experiment on 
BIDMC32 dataset, using BCR-DE model.
'''

import math
import os
import sys
import argparse
import random 
import numpy as np 
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

import pdb
import json
import csv
import time as sys_time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import plot_and_save, manual_set_seed
from datasets import Fixed_Synthetic_Dataset
from JAX_BCR_DE_model import CDE_BCR
from flax import nnx
import jax.numpy as jnp

import jax
jax.config.update("jax_enable_x64", True)

import optax
import orbax.checkpoint as ocp

def numpy_collate(batch):
	if isinstance(batch[0], jnp.ndarray):
		return jnp.stack(batch, dtype = jnp.float64)
	elif isinstance(batch[0], (tuple,list)):
		transposed = zip(*batch)
		return [numpy_collate(samples) for samples in transposed]
	else:
		return jnp.array(batch, dtype = jnp.float64)
	

def parse_args():
	'''
	Parse input arguments
	'''
	parser = argparse.ArgumentParser(description="Arguments for CDE in BCR")

	# Configuration file path
	parser.add_argument('--config_path', type=str, default='./configs/regression/RR.json', help='Path to data configuration file')

	# arguments for dataset
	parser.add_argument('--seq_length', type=int, default=4000, help='Total seqeunce length in time series')
	parser.add_argument('--data_path', type=str, default='../data/', help='Path to data directory for BIDMC32 and Eigenworm dataset. Check README')
	parser.add_argument('--attribute_type', type=str, default='RR', choices=['RR', 'HR', 'SpO2'], help='Attribute to regress to')

	# Setting experiment arugments
	parser.add_argument("--seed", default=37825926, type=int, help="Setting seed for the entire experiment")                 
	parser.add_argument("--exp", default='ERR_re', help="Adjusted in code: Experiment foler name")

	# Setting model arguments
	parser.add_argument('--wave', default='db2', type=str, help='Type of pywavelet')
	parser.add_argument('--dim_D', default=3, type=int, help="Dimension of observable variable")
	parser.add_argument('--dim_D_out', default=3, type=int, help="Dimension of predicte variable")
	parser.add_argument('--dim_d', default=6, type=int, help="Latent dimension of evolution")
	parser.add_argument('--dim_k', default=6, type=int, help="Dimension of h_theta (first)")
	parser.add_argument('--num_classes', default=1, type=int, help="Output dimensionality of model")
	parser.add_argument('--nonlinearity', default='tanh', type=str, help='Non lienarity to use')
	parser.add_argument('--n_levels', default=8, type=int, help="Number of levels of wavelet decomposition")
	parser.add_argument('--K_dense', default=2, type=int, help="Number of dense layers")
	parser.add_argument('--K_LC', default=4, type=int, help="Number of LC layers per level")
	parser.add_argument('--nb', default=3, type=int, help="Diagonal banded length, dertimine the kernel size")
	parser.add_argument('--num_sparse_LC', default=6, type=int, help="Number of sparse LC unit")
	parser.add_argument('--interpol', default='spline', type=str, help='Interpolation type to use')
	parser.add_argument('--use_cheap_sparse_LC', default=True, action='store_false', help='Whether to use sparse Locally connected layer')

	# training arguments
	parser.add_argument('--train_bs', default=64, type=int, help='Batchsize for train loader')
	parser.add_argument('--valid_bs', default=256, type=int, help='Batchsize for valid loader')
	parser.add_argument('--test_bs', default=256, type=int, help='Batchsize for test loader')
	parser.add_argument('--epoch', default=200, type=int, help='Number of epochs to train')
	parser.add_argument('--lr', default=0.01, type=float, help="Learning rate for the BCR_DE model")
	parser.add_argument('--wd', default=0.0001, type=float, help="Learning rate for the BCR_DE model")
	parser.add_argument('--model_pred_save_freq', default=10, type=int, help='Saving frequency of model prediction')
			
	args = parser.parse_args()
	return args

def get_memory(device, reset=False, in_mb=True):
	""" Gets the GPU usage. """
	if device is None:
		return float('nan')
	if device.type == 'cuda':
		if reset:
			torch.cuda.reset_max_memory_allocated(device)
		bytes = torch.cuda.max_memory_allocated(device)
		if in_mb:
			bytes = bytes / 1024 / 1024
		return bytes
	else:
		return float('nan')

def compute_acc(pred, target):
	probs = torch.softmax(pred, dim=1)
	winners = probs.argmax(dim=1)
	corrects = (winners == target)
	accuracy = corrects.sum().float() / float( target.size(0) )
	return accuracy

def main():
	print("Solving CDE through BCR")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	args = parse_args()

	arg_dict = vars(args)

	provided_dict = json.load(open(args.config_path))
	arg_dict.update(provided_dict)

	exp_name = args.exp
	print(args)
	if os.path.exists('./Result/'+exp_name):
		nth_exp = len(os.listdir('./Result/'+exp_name+'/Results'))+1
	else:
		nth_exp = 0
	args.exp = './Result/'+exp_name+'/Results/'+str(nth_exp)
	arg_dict['Result_location'] = './Results/'+str(nth_exp)

	if not os.path.exists(args.exp):
		print("Creating experiment directory: ", args.exp)
		os.makedirs(args.exp)

	# list of column names 
	field_names = arg_dict.keys()
	argument_storage_file = './Result/'+exp_name+'/experiment.csv'
	if os.path.exists(argument_storage_file):
		with open(argument_storage_file, 'a') as csv_file:
			dict_object = csv.DictWriter(csv_file, fieldnames=field_names) 
			dict_object.writerow(arg_dict)
	else:
		print("First run of experiment, so create new storage for hyperparameter")
		with open(argument_storage_file, 'w') as csv_file:
			writer = csv.writer(csv_file) 
			writer.writerow(arg_dict.keys()) 
			writer.writerow(arg_dict.values())

	# Setting seeds for reproducibility
	manual_set_seed(args.seed)
	print(args)
	
	if args.attribute_type == 'RR':
		train_x = torch.from_numpy(np.load(args.data_path+'BIDMC/RR/RR_train_x.npy')).transpose(1,2)
		train_y = torch.from_numpy(np.load(args.data_path+'BIDMC/RR/RR_train_y.npy'))

		valid_x = torch.from_numpy(np.load(args.data_path+'BIDMC/RR/RR_val_x.npy')).transpose(1,2)
		valid_y = torch.from_numpy(np.load(args.data_path+'BIDMC/RR/RR_val_y.npy'))

		test_x = torch.from_numpy(np.load(args.data_path+'BIDMC/RR/RR_test_x.npy')).transpose(1,2)
		test_y = torch.from_numpy(np.load(args.data_path+'BIDMC/RR/RR_test_y.npy'))
	elif args.attribute_type == 'HR':
		train_x = torch.from_numpy(np.load(args.data_path+'BIDMC/HR/HR_train_x.npy')).transpose(1,2)
		train_y = torch.from_numpy(np.load(args.data_path+'BIDMC/HR/HR_train_y.npy'))

		valid_x = torch.from_numpy(np.load(args.data_path+'BIDMC/HR/HR_val_x.npy')).transpose(1,2)
		valid_y = torch.from_numpy(np.load(args.data_path+'BIDMC/HR/HR_val_y.npy'))

		test_x = torch.from_numpy(np.load(args.data_path+'BIDMC/HR/HR_test_x.npy')).transpose(1,2)
		test_y = torch.from_numpy(np.load(args.data_path+'BIDMC/HR/HR_test_y.npy'))
	elif args.attribute_type == 'SpO2':
		train_x = torch.from_numpy(np.load(args.data_path+'BIDMC/SpO2/SP_train_x.npy')).transpose(1,2)
		train_y = torch.from_numpy(np.load(args.data_path+'BIDMC/SpO2/SP_train_y.npy'))

		valid_x = torch.from_numpy(np.load(args.data_path+'BIDMC/SpO2/SP_val_x.npy')).transpose(1,2)
		valid_y = torch.from_numpy(np.load(args.data_path+'BIDMC/SpO2/SP_val_y.npy'))

		test_x = torch.from_numpy(np.load(args.data_path+'BIDMC/SpO2/SP_test_x.npy')).transpose(1,2)
		test_y = torch.from_numpy(np.load(args.data_path+'BIDMC/SpO2/SP_test_y.npy'))
	else:
		print("Check regression attribute")
		exit(0)

	print(train_x.shape, train_y.shape, valid_x.shape, valid_y.shape, test_x.shape, test_y.shape)
	time_step = train_x[0, 0, :]
	print("Time step:", time_step.shape)

	argument_file = args.exp+'/arguments.pkl'
	with open(argument_file, 'wb') as f:
		pickle.dump(arg_dict, f)

	train_set = Fixed_Synthetic_Dataset(train_x, train_y, time_step, args.interpol)
	valid_set = Fixed_Synthetic_Dataset(valid_x, valid_y, time_step, args.interpol)
	test_set = Fixed_Synthetic_Dataset(test_x, test_y, time_step, args.interpol)

	time_step = jnp.array(time_step, dtype = jnp.float64)

	train_dl = DataLoader(train_set, batch_size=args.train_bs, shuffle=True, collate_fn = numpy_collate)
	valid_dl = DataLoader(valid_set, batch_size=args.valid_bs, shuffle=True, collate_fn = numpy_collate)
	test_dl = DataLoader(test_set, batch_size=args.test_bs, shuffle=False, collate_fn = numpy_collate)

	rngs = nnx.Rngs(args.seed)
	model = CDE_BCR(time_step=time_step, wave=args.wave,
					D=args.dim_D, D_out=args.dim_D_out, d=args.dim_d, k=args.dim_k, original_length=args.seq_length, 
					num_classes=args.num_classes, nonlinearity=args.nonlinearity, n_levels=args.n_levels, 
					K_dense=args.K_dense, K_LC=args.K_LC, nb=args.nb, 
					num_sparse_LC=args.num_sparse_LC, use_cheap_sparse_LC=args.use_cheap_sparse_LC, interpol=args.interpol, conv_bias=True, rngs = rngs,  
					predict=True, masked_modelling=False)
	
	
	opt = optax.adamw(learning_rate = args.lr, weight_decay = args.wd),
	optimizer = nnx.Optimizer(model, opt)
	all_losses = {}
	all_losses['train_total_loss'] = []
	all_losses['valid_total_loss'] = []
	all_losses['test_total_loss'] = []

	result_dict = {}
	result_dict['train_time'] = []
	result_dict['memory'] = []
	train_step_jit = jax.jit(train_step)
	print("Start training")
	for epoch in range(args.epoch):
		epoch_total_loss = 0
		n_batches = 0
		start_time = sys_time.time() 
		start_memory = get_memory(device, reset=True)

		for x, coeffs, y_true, time in tqdm(train_dl, leave=False):
			x, coeffs, y_true = x, coeffs, y_true
			graphdef, state = nnx.split((model, optimizer))
			state, loss = train_step_jit(graphdef, state, x, coeffs, y_true, time_step)

			nnx.update((model, optimizer), state)
			epoch_total_loss += float(loss)
			n_batches += 1

		result_dict['train_time'].append(sys_time.time()-start_time)
		result_dict['memory'].append(get_memory(device)-start_memory)
		print("Epoch: {}; Train: Total Loss:{}".format(epoch, epoch_total_loss/n_batches))
		all_losses['train_total_loss'].append(epoch_total_loss/n_batches)

		epoch_total_loss = 0
		n_batches = 0
		with torch.no_grad():
			for x, coeffs, y_true, time in tqdm(valid_dl, leave=False):
				x_pred, y_pred = model(x, coeffs, time_step)

				loss = jnp.sqrt(optax.losses.squared_error(y_pred, y_true).mean())

				epoch_total_loss += float(loss)
				n_batches += 1

			print("\t Validation: Total Loss:{}".format(epoch_total_loss/n_batches))
		all_losses['valid_total_loss'].append(epoch_total_loss/n_batches)


		epoch_total_loss = 0
		n_batches = 0
		with torch.no_grad():
			for x, coeffs, y_true, time in tqdm(test_dl, leave=False):
				x_pred, y_pred = model(x, coeffs, time_step)

				loss = jnp.sqrt(optax.losses.squared_error(y_pred, y_true).mean())

				epoch_total_loss += float(loss)
				n_batches += 1

			print("\t Test: Total Loss:{}".format(epoch_total_loss/n_batches))
		all_losses['test_total_loss'].append(epoch_total_loss/n_batches)
		
		plot_and_save(args.exp, all_losses, only_one=True)

		result_file = args.exp+'/result.pkl'
		with open(result_file, 'wb') as f:
			pickle.dump(result_dict, f)

		# if epoch%args.model_pred_save_freq==0:
		# 	torch.save(model.state_dict(), args.exp+'/model_'+str(epoch)+'.pt')

	# torch.save(model.state_dict(), args.exp+'/final_model.pt')
	print("#####################################################")

def loss_fn(model, x, coeffs, y_true, time_step):
	x_pred, y_pred = model(x, coeffs, time_step)
	loss = jnp.sqrt(optax.losses.squared_error(y_pred, y_true).mean())
	return loss 

def train_step(graphdef, state, x, coeffs, y_true, time_step):
	model, optimizer = nnx.merge(graphdef, state)
	loss, grads = nnx.value_and_grad(loss_fn)(model, x, coeffs, y_true, time_step)
	optimizer.update(grads, value = loss)
	_, state = nnx.split((model, optimizer))
	return state, loss

if __name__ == '__main__':
	main()