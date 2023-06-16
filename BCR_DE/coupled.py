'''
Final main file, for coupled differential equation
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

from utils import save_data, plot_and_save, get_VDP_data, get_Hare_Lynx_data, get_coupled_data
from utils import manual_set_seed, save_data_couple, get_Benzene_data, get_Lorenz_data, get_HH_data_ps_2_act
from datasets import Synthetic_Dataset, Fixed_Synthetic_Dataset
from BCR_DE_model import CDE_BCR



def parse_args():
	'''
	Parse input arguments
	'''
	parser = argparse.ArgumentParser(description="Arguments for CDE in BCR")

	# Configuration file path
	parser.add_argument('--config_path', type=str, default='./configs/regression/RR.json', help='Path to data configuration file')

	# arguments for dataset
	parser.add_argument('--seq_length', type=int, default=20000, help='Total seqeunce length in time series')
	parser.add_argument('--couple_type', type=str, default='toy', choices=['toy', 'LV', 'VDP', 'cLorenz', 'HH', 'Benzene'])

	# Setting experiment arugments
	parser.add_argument("--seed", type=int, default=0, help="Setting seed for the entire experiment")
	parser.add_argument("--exp", default='Coupled', help="Adjusted in code: Experiment foler name")

	# Setting model arguments
	parser.add_argument('--wave', default='db2', type=str, help='Type of pywavelet')
	parser.add_argument('--dim_D', default=1, type=int, help="Dimension of observable variable")
	parser.add_argument('--dim_D_out', default=1, type=int, help="Dimension of predicte variable")
	parser.add_argument('--dim_d', default=3, type=int, help="Latent dimension of evolution")
	parser.add_argument('--dim_k', default=3, type=int, help="Dimension of h_theta (first)")
	parser.add_argument('--num_classes', default=1, type=int, help="Output dimensionality of model")
	parser.add_argument('--nonlinearity', default='relu', type=str, help='Non lienarity to use')
	parser.add_argument('--n_levels', default=8, type=int, help="Number of levels of wavelet decomposition")
	parser.add_argument('--K_dense', default=4, type=int, help="Number of dense layers")
	parser.add_argument('--K_LC', default=4, type=int, help="Number of LC layers per level")
	parser.add_argument('--nb', default=3, type=int, help="Diagonal banded length, dertimine the kernel size")
	parser.add_argument('--num_sparse_LC', default=6, type=int, help="Number of sparse LC unit")
	parser.add_argument('--interpol', default='linear', type=str, help='Interpolation type to use')
	parser.add_argument('--use_cheap_sparse_LC', default=True, action='store_false', help='Whether to use sparse Locally connected layer')

	# training arguments
	parser.add_argument('--train_bs', default=64, type=int, help='Batchsize for train loader')
	parser.add_argument('--valid_bs', default=256, type=int, help='Batchsize for valid loader')
	parser.add_argument('--test_bs', default=256, type=int, help='Batchsize for test loader')
	parser.add_argument('--epoch', default=100, type=int, help='Number of epochs to train')
	parser.add_argument('--lr', default=0.01, type=float, help="Learning rate for the BCR_DE model")
	parser.add_argument('--wd', default=0.0001, type=float, help="Learning rate for the BCR_DE model")
	parser.add_argument('--model_pred_save_freq', default=1, type=int, help='Saving frequency of model prediction')
			
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

def main():
	print("Solving CDE through BCR")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	args = parse_args()

	arg_dict = vars(args)
	provided_dict = json.load(open(args.config_path))
	arg_dict.update(provided_dict)

	exp_name = args.exp+'_'+args.couple_type
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
	with open('./Result/'+exp_name+'/experiment.csv', 'a') as csv_file:
		dict_object = csv.DictWriter(csv_file, fieldnames=field_names) 
		dict_object.writerow(arg_dict)

	# Setting seeds for reproducibility
	manual_set_seed(args.seed)
	print(args)

	if args.couple_type == 'toy':
		# toy, dim_d = 3 and dim_k = 3 work well enough
		assert args.dim_D == 1
		assert args.dim_D_out == 1
		t = np.linspace(0,1,args.seq_length)
		train_x, train_y = get_coupled_data(2000, t, args.seq_length)
		valid_x, valid_y = get_coupled_data(1000, t, args.seq_length)
		test_x, test_y = get_coupled_data(1000, t, args.seq_length)
		time_step = torch.from_numpy(t)
	elif args.couple_type == 'LV':
		# LV, dim_d = 6 and dim_k = 6 (higher latent dimension give better performance)
		assert args.dim_D == 1
		assert args.dim_D_out == 1
		t = np.linspace(0, 70, args.seq_length)
		train_x, train_y = get_Hare_Lynx_data(2000, t, noise=False)
		valid_x, valid_y = get_Hare_Lynx_data(1000, t, noise=False)
		test_x, test_y = get_Hare_Lynx_data(1000, t, noise=False)
		time_step = torch.from_numpy(t)
	elif args.couple_type ==  'VDP':
		# VDP, dim_d = 3 and dim_k = 3 work well enough
		assert args.dim_D == 1
		assert args.dim_D_out == 1
		t = np.linspace(0,70,args.seq_length)
		train_x, train_y = get_VDP_data(2000, t, noise=False)
		valid_x, valid_y = get_VDP_data(1000, t, noise=False)
		test_x, test_y = get_VDP_data(1000, t, noise=False)
		time_step = torch.from_numpy(t)
	elif args.couple_type == 'cLorenz':
		# Lorenz, dim_d = 8 and dim_k =8 (higher latent dimension give better performance)
		assert args.dim_D == 2
		assert args.dim_D_out == 1
		t = np.arange(0.0, 100.0, 0.01)
		train_x, train_y = get_Lorenz_data(2000, t, noise=False)
		valid_x, valid_y = get_Lorenz_data(1000, t, noise=False)
		test_x, test_y = get_Lorenz_data(1000, t, noise=False)
		time_step = torch.from_numpy(t)
	elif args.couple_type == 'HH':
		# Hodgkin-Huxely: from potential and stimulus to Na, K activation, dim_d = 3 and dim_k = 3 work well enough
		assert args.dim_D == 2
		assert args.dim_D_out == 3
		train_x, train_y, valid_x, valid_y, test_x, test_y, time_step = get_HH_data_ps_2_act()
	elif args.couple_type == 'Benzene':
		# Benzene, dim_d = 3 and dim_k = 3 work well enough
		train_x, train_y, valid_x, valid_y, test_x, test_y, time_step = get_Benzene_data('BenzeneConcentration', 240)
	else:
		print("Coupled diff eq not yet implemented")


	argument_file = args.exp+'/arguments.pkl'
	with open(argument_file, 'wb') as f:
		pickle.dump(arg_dict, f)

	train_set = Fixed_Synthetic_Dataset(train_x, train_y, time_step, args.interpol)
	valid_set = Fixed_Synthetic_Dataset(valid_x, valid_y, time_step, args.interpol)
	test_set = Fixed_Synthetic_Dataset(test_x, test_y, time_step, args.interpol)

	time_step = time_step.to(device)

	train_dl = DataLoader(train_set, batch_size=args.train_bs)
	valid_dl = DataLoader(valid_set, batch_size=args.valid_bs)
	test_dl = DataLoader(test_set, batch_size=args.test_bs)

	model = CDE_BCR(time_step=time_step, wave=args.wave,
					D=args.dim_D, D_out=args.dim_D_out, d=args.dim_d, k=args.dim_k, original_length=args.seq_length, 
					num_classes=args.num_classes, nonlinearity=args.nonlinearity, n_levels=args.n_levels, 
					K_dense=args.K_dense, K_LC=args.K_LC, nb=args.nb, 
					num_sparse_LC=args.num_sparse_LC, use_cheap_sparse_LC=args.use_cheap_sparse_LC, interpol=args.interpol, conv_bias=True, 
					predict=False, masked_modelling=False).to(device)
	
	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print("Total number of trainable parameters: ", pytorch_total_params)

	loss_fn = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

	all_losses = {}
	all_losses['train_total_loss'] = []
	all_losses['valid_total_loss'] = []
	all_losses['test_total_loss'] = []

	result_dict = {}
	result_dict['number_param'] = pytorch_total_params
	result_dict['train_time'] = []
	result_dict['memory'] = []

	print("Start training")
	for epoch in range(args.epoch):
		epoch_total_loss = 0
		n_batches = 0
		start_memory = get_memory(device, reset=True)
		start_time = sys_time.time() 
		for x, coeffs, y_true, time in tqdm(train_dl, leave=False):
			x, coeffs, y_true = x.to(device).float(), coeffs.to(device).float(), y_true.to(device).float()
			x_pred = model(x, coeffs, time_step)
			loss = loss_fn(x_pred, y_true.transpose(1,2))

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			epoch_total_loss += loss.item()
			n_batches += 1

		result_dict['train_time'].append(sys_time.time()-start_time)
		result_dict['memory'].append(get_memory(device)-start_memory)
		print("Epoch: {}; Train: Total Loss:{}".format(epoch, epoch_total_loss/n_batches))

		all_losses['train_total_loss'].append(epoch_total_loss/n_batches)

		epoch_total_loss = 0
		n_batches = 0
		with torch.no_grad():
			for x, coeffs, y_true, time in tqdm(valid_dl, leave=False):
				x, coeffs, y_true = x.to(device).float(), coeffs.to(device).float(), y_true.to(device).float()
				x_pred = model(x, coeffs, time_step)
				loss = loss_fn(x_pred, y_true.transpose(1,2))

				epoch_total_loss += loss.item()
				n_batches += 1

				# Currently saving only the first batch
				if n_batches ==1 and epoch%args.model_pred_save_freq==0:
					print("Saving model prediction...")
					save_data_couple(y_true.transpose(1,2), x_pred, x, epoch, 'valid', args.exp)

			print("\t Validation: Total Loss:{}".format(epoch_total_loss/n_batches))
		all_losses['valid_total_loss'].append(epoch_total_loss/n_batches)

		# Learning rate scheduler
		val_loss = epoch_total_loss/n_batches
		scheduler.step(val_loss)

		epoch_total_loss = 0
		n_batches = 0
		with torch.no_grad():
			for x, coeffs, y_true, time in tqdm(test_dl, leave=False):
				x, coeffs, y_true = x.to(device).float(), coeffs.to(device).float(), y_true.to(device).float()
				x_pred = model(x, coeffs, time_step)
				loss = loss_fn(x_pred, y_true.transpose(1,2))

				epoch_total_loss += loss.item()
				n_batches += 1

				# Currently saving only the first batch
				if n_batches ==1 and epoch%args.model_pred_save_freq==0:
					print("Saving model prediction...")
					save_data_couple(y_true.transpose(1,2), x_pred, x, epoch, 'test', args.exp)

			print("\t Test: Total Loss:{}".format(epoch_total_loss/n_batches))
		all_losses['test_total_loss'].append(epoch_total_loss/n_batches)
		
		plot_and_save(args.exp, all_losses, only_one=True)

		result_file = args.exp+'/result.pkl'
		with open(result_file, 'wb') as f:
			pickle.dump(result_dict, f)

		if epoch%args.model_pred_save_freq==0:
			torch.save(model.state_dict(), args.exp+'/model_'+str(epoch)+'.pt')

	torch.save(model.state_dict(), args.exp+'/final_model.pt')
	print("#####################################################")

if __name__ == '__main__':
	main()