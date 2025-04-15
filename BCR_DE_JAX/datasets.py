'''
This file has the custom dataset loading classes defined
'''
import os
import sys
import numpy as np 
import torch 
import random
import pdb
from torch.utils.data import Dataset
import torchcde

class Synthetic_Dataset(Dataset):
	def __init__(self, controls, responses, time_step, interpol):
		self.all_values = controls.transpose(1,2)
		self.responses = responses
		self.time_step = time_step

		self.n_samples = self.all_values.size(0)
		self.seq_length = self.all_values.size(1)
		self.input_dim = self.all_values.size(2)
		self.interpol = interpol

		if self.interpol == 'linear':
			print("Linear interpolation")
			self.coeffs = self.all_values
		elif self.interpol == 'spline':
			print("Spline based interpolation")
			self.coeffs =  torchcde.hermite_cubic_coefficients_with_backward_differences(self.all_values)
		else:
			print('Invalid interpolation type')
			exit(0)

	def __getitem__(self, idx):
		return self.all_values[idx], self.coeffs[idx], self.responses[idx], self.time_step

	def __len__(self):
		return len(self.responses)

class Fixed_Synthetic_Dataset(Dataset):
	def __init__(self, controls, responses, time_step, interpol):
		self.all_values = controls.transpose(1,2)
		self.responses = responses
		self.time_step = time_step

		print("all_values shape",self.all_values.shape)
		print("reponses shape",self.responses.shape)
		print("time_step shape",self.time_step.shape)

		self.n_samples = self.all_values.size(0)
		self.seq_length = self.all_values.size(1)
		self.input_dim = self.all_values.size(2)
		self.interpol = interpol

		if self.interpol == 'linear':
			print("Changed Linear interpolation")
			self.coeffs = torchcde.linear_interpolation_coeffs(self.all_values)

		elif self.interpol == 'spline':
			print("Spline based interpolation")
			self.coeffs =  torchcde.hermite_cubic_coefficients_with_backward_differences(self.all_values)
		else:
			print('Invalid interpolation type')
			exit(0)

	def __getitem__(self, idx):
		return self.all_values[idx], self.coeffs[idx], self.responses[idx], self.time_step

	def __len__(self):
		return len(self.responses)


class Fixed_Masked_Dataset(Dataset):
	def __init__(self, controls, responses, time_step, interpol, patch_length, patch_start_index):
		self.all_values = controls.transpose(1,2)
		self.responses = responses
		self.time_step = time_step
		self.interpol = interpol

		self.patch_length = patch_length
		self.patch_start_index = patch_start_index

		self.n_samples = self.all_values.size(0)
		self.seq_length = self.all_values.size(1)
		self.input_dim = self.all_values.size(2)

		self.masked_values = self.all_values.detach().clone()
		self.ones = torch.ones(self.all_values.shape)
		self.known_index = self.ones > 0 

		if patch_length == 0:
			print("No masking")
		else:
			print("Patch length:", patch_length)
			print("Patch start:", patch_start_index)
			self.masked_values[:, patch_start_index:patch_start_index+patch_length, :] = torch.randn(self.masked_values[:, patch_start_index:patch_start_index+patch_length, :].shape)
			# self.masked_values[:, patch_start_index:patch_start_index+patch_length, :] = float('nan')
			self.known_index[:, patch_start_index:patch_start_index+patch_length, :] = False

		# Create interpolated path on missing data
		if self.interpol == 'linear':
			print("Linear interpolation")
			self.coeffs = torchcde.linear_interpolation_coeffs(self.masked_values)
		elif self.interpol == 'spline':
			print("Spline based interpolation")
			self.coeffs =  torchcde.hermite_cubic_coefficients_with_backward_differences(self.masked_values)
		else:
			print('Invalid interpolation type')
			exit(0)

	def __getitem__(self, idx):
		return self.masked_values[idx], self.known_index[idx], self.coeffs[idx], self.responses[idx], self.time_step

	def __len__(self):
		return len(self.responses)

