'''
Main file for missing data reconstruction
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
import time as sys_time
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import save_data, plot_and_save_L1_TV, load_PPG, load_ECG
from utils import manual_set_seed
from datasets import Fixed_Masked_Dataset
from BCR_DE_model import CDE_BCR

def parse_args():
	'''
	Parse input arguments
	'''
	parser = argparse.ArgumentParser(description="Arguments for CDE in BCR")

	# arguments for dataset
	parser.add_argument('--seq_length', type=int, default=4000, help='Total seqeunce length in time series')
	parser.add_argument('--dataset_type', type=str, default='ECG', choices=['ECG', 'PPG'])
	parser.add_argument('--patch_length', type=int, default=100, help='Total mask length in time series')
	parser.add_argument('--patch_start_index', type=int, default=2000, help='Mask start index in time series')

	# Setting experiment arugments
	parser.add_argument("--seed", default=0, type=int, help="Setting seed for the entire experiment")
	parser.add_argument("--exp", default='Masked_Autoencode', help="Adjusted in code: Experiment foler name")

	# Setting model arguments
	parser.add_argument('--wave', default='db2', type=str, help='Type of pywavelet')
	parser.add_argument('--dim_D', default=1, type=int, help="Dimension of observable variable")
	parser.add_argument('--dim_d', default=4, type=int, help="Latent dimension of evolution")
	parser.add_argument('--dim_k', default=4, type=int, help="Dimension of h_theta (first)")
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
	parser.add_argument('--wL1', default=0.00001, type=float, help="Weight for L1 loss")
	parser.add_argument('--wTV', default=0.00001, type=float, help="Weight for TV loss")
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
	exp_name = args.exp+'_'+args.dataset_type

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


	if args.dataset_type == 'PPG':
		train_data, valid_data, test_data, time_step = load_PPG()
	elif args.dataset_type == 'ECG':
		train_data, valid_data, test_data, time_step = load_ECG()
	else:
		print("Check dataset type")


	argument_file = args.exp+'/arguments.pkl'
	with open(argument_file, 'wb') as f:
		pickle.dump(arg_dict, f)

	train_set = Fixed_Masked_Dataset(train_data['x'], train_data['y'], time_step, args.interpol, args.patch_length, args.patch_start_index)
	valid_set = Fixed_Masked_Dataset(valid_data['x'], valid_data['y'], time_step, args.interpol, args.patch_length, args.patch_start_index)
	test_set = Fixed_Masked_Dataset(test_data['x'], test_data['y'], time_step, args.interpol, args.patch_length, args.patch_start_index)

	time_step = time_step.to(device)

	train_dl = DataLoader(train_set, batch_size=args.train_bs)
	valid_dl = DataLoader(valid_set, batch_size=args.valid_bs)
	test_dl = DataLoader(test_set, batch_size=args.test_bs)

	model = CDE_BCR(time_step=time_step, wave=args.wave,
					D=args.dim_D, D_out=args.dim_D, d=args.dim_d, k=args.dim_k, original_length=args.seq_length, 
					num_classes=args.num_classes, nonlinearity=args.nonlinearity, n_levels=args.n_levels, 
					K_dense=args.K_dense, K_LC=args.K_LC, nb=args.nb, 
					num_sparse_LC=args.num_sparse_LC, use_cheap_sparse_LC=args.use_cheap_sparse_LC, interpol=args.interpol, conv_bias=True, 
					predict=False, masked_modelling=True).to(device)
	
	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print("Total number of trainable parameters: ", pytorch_total_params)

	loss_fn = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)

	all_losses = {}
	all_losses['train_recon_loss'] = []
	all_losses['valid_recon_loss'] = []
	all_losses['test_recon_loss'] = []
	all_losses['train_L1_loss'] = []
	all_losses['valid_L1_loss'] = []
	all_losses['test_L1_loss'] = []
	all_losses['train_TV_loss'] = []
	all_losses['valid_TV_loss'] = []
	all_losses['test_TV_loss'] = []

	result_dict = {}
	result_dict['number_param'] = pytorch_total_params
	result_dict['train_time'] = []
	result_dict['memory'] = []

	D = torch.diag(torch.ones(3999),1).to(device) + torch.diag(-1*torch.ones(4000)).to(device)

	print("Start training")
	for epoch in range(args.epoch):
		epoch_recon_loss = 0
		epoch_L1_loss = 0
		epoch_TV_loss = 0
		epoch_total_loss = 0
		n_batches = 0
		start_time = sys_time.time() 
		start_memory = get_memory(device, reset=True)
		for x, ki, coeffs, y_true, time in tqdm(train_dl, leave=False):
			x, coeffs, y_true = x.to(device).float(), coeffs.to(device).float(), y_true.to(device).float()
			x_pred, approx, detail = model(x, coeffs, time_step)

			recon_loss = loss_fn(x_pred[ki], x[ki])
			L1_loss = 0
			L1_loss += torch.norm(approx,1)
			for i in range(len(detail)):
				L1_loss += torch.norm(detail[i],1)
			TV = torch.matmul(D, x_pred)
			TV_loss = torch.norm(TV, 1)
			loss = recon_loss + args.wL1*L1_loss + args.wTV*TV_loss

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			epoch_recon_loss += recon_loss.item()
			epoch_L1_loss += L1_loss.item()
			epoch_TV_loss += TV_loss.item()
			epoch_total_loss += loss.item()
			n_batches += 1

			# Currently saving only the first batch
			if n_batches ==1 and epoch%args.model_pred_save_freq==0:
				print("Saving model prediction...")
				save_data(x, x_pred, epoch, 'train', args.exp)

		result_dict['train_time'].append(sys_time.time()-start_time)
		result_dict['memory'].append(get_memory(device)-start_memory)
		print("Epoch: {}; Train: Recon Loss:{}, L1:{}, TV:{}".format(epoch, epoch_recon_loss/n_batches, epoch_L1_loss/n_batches, epoch_TV_loss/n_batches))

		all_losses['train_recon_loss'].append(epoch_recon_loss/n_batches)
		all_losses['train_L1_loss'].append(epoch_L1_loss/n_batches)
		all_losses['train_TV_loss'].append(epoch_TV_loss/n_batches)

		epoch_recon_loss = 0
		epoch_L1_loss = 0
		epoch_TV_loss = 0
		epoch_total_loss = 0
		n_batches = 0
		with torch.no_grad():
			for x, ki, coeffs, y_true, time in tqdm(valid_dl, leave=False):
				x, coeffs, y_true = x.to(device).float(), coeffs.to(device).float(), y_true.to(device).float()
				x_pred, approx, detail = model(x, coeffs, time_step)

				recon_loss = loss_fn(x_pred[ki], x[ki])
				L1_loss = 0
				L1_loss += torch.norm(approx,1)
				for i in range(len(detail)):
					L1_loss += torch.norm(detail[i],1)
				TV = torch.matmul(D, x_pred)
				TV_loss = torch.norm(TV, 1)
				loss = recon_loss + args.wL1*L1_loss + args.wTV*TV_loss

				epoch_recon_loss += recon_loss.item()
				epoch_L1_loss += L1_loss.item()
				epoch_TV_loss += TV_loss.item()
				epoch_total_loss += loss.item()
				n_batches += 1

				# Currently saving only the first batch
				if n_batches ==1 and epoch%args.model_pred_save_freq==0:
					print("Saving model prediction...")
					save_data(x, x_pred, epoch, 'valid', args.exp)

			print("\t Validation: Recon Loss:{}, L1:{}, TV:{}".format(epoch_recon_loss/n_batches, epoch_L1_loss/n_batches, epoch_TV_loss/n_batches))

		all_losses['valid_recon_loss'].append(epoch_recon_loss/n_batches)
		all_losses['valid_L1_loss'].append(epoch_L1_loss/n_batches)
		all_losses['valid_TV_loss'].append(epoch_TV_loss/n_batches)

		# Learning rate scheduler
		val_loss = epoch_total_loss/n_batches
		scheduler.step(val_loss)

		epoch_recon_loss = 0
		epoch_L1_loss = 0
		epoch_TV_loss = 0
		epoch_total_loss = 0
		n_batches = 0
		with torch.no_grad():
			for x, ki, coeffs, y_true, time in tqdm(test_dl, leave=False):
				x, coeffs, y_true = x.to(device).float(), coeffs.to(device).float(), y_true.to(device).float()
				x_pred, approx, detail = model(x, coeffs, time_step)

				recon_loss = loss_fn(x_pred[ki], x[ki])
				L1_loss = 0
				L1_loss += torch.norm(approx,1)
				for i in range(len(detail)):
					L1_loss += torch.norm(detail[i],1)
				TV = torch.matmul(D, x_pred)
				TV_loss = torch.norm(TV, 1)
				loss = recon_loss + args.wL1*L1_loss + args.wTV*TV_loss

				epoch_recon_loss += recon_loss.item()
				epoch_L1_loss += L1_loss.item()
				epoch_TV_loss += TV_loss.item()
				epoch_total_loss += loss.item()
				n_batches += 1

				# Currently saving only the first batch
				if n_batches ==1 and epoch%args.model_pred_save_freq==0:
					print("Saving model prediction...")
					save_data(x, x_pred, epoch, 'test', args.exp)

			print("\t Test: Recon Loss:{}, L1:{}, TV:{}".format(epoch_recon_loss/n_batches, epoch_L1_loss/n_batches, epoch_TV_loss/n_batches))
		
		all_losses['test_recon_loss'].append(epoch_recon_loss/n_batches)
		all_losses['test_L1_loss'].append(epoch_L1_loss/n_batches)
		all_losses['test_TV_loss'].append(epoch_TV_loss/n_batches)

		plot_and_save_L1_TV(args.exp, all_losses)
		result_file = args.exp+'/result.pkl'
		with open(result_file, 'wb') as f:
			pickle.dump(result_dict, f)

		if epoch%args.model_pred_save_freq==0:
			torch.save(model.state_dict(), args.exp+'/model_'+str(epoch)+'.pt')

	torch.save(model.state_dict(), args.exp+'/final_model.pt')
	print("#####################################################")

if __name__ == '__main__':
	main()