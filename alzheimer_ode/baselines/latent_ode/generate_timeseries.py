###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

# Create a synthetic dataset
from __future__ import absolute_import, division
from __future__ import print_function
import os
import matplotlib
if os.path.exists("/Users/yulia"):
	matplotlib.use('TkAgg')
else:
	matplotlib.use('Agg')

import numpy as np
import numpy.random as npr
from scipy.special import expit as sigmoid
import pickle
import matplotlib.pyplot as plt
import matplotlib.image
import torch
import lib.utils as utils

# ======================================================================================

def get_next_val(init, t, tmin, tmax, final = None):
	if final is None:
		return init
	val = init + (final - init) / (tmax - tmin) * t
	return val


def generate_periodic(time_steps, init_freq, init_amplitude, starting_point, 
	final_freq = None, final_amplitude = None, phi_offset = 0.):

	tmin = time_steps.min()
	tmax = time_steps.max()

	data = []
	t_prev = time_steps[0]
	phi = phi_offset
	for t in time_steps:
		dt = t - t_prev
		amp = get_next_val(init_amplitude, t, tmin, tmax, final_amplitude)
		freq = get_next_val(init_freq, t, tmin, tmax, final_freq)
		phi = phi + 2 * np.pi * freq * dt # integrate to get phase

		y = amp * np.sin(phi) + starting_point
		t_prev = t
		data.append([t,y])
	return np.array(data)

def assign_value_or_sample(value, sampling_interval = [0.,1.]):
	if value is None:
		int_length = sampling_interval[1] - sampling_interval[0]
		return np.random.random() * int_length + sampling_interval[0]
	else:
		return value

class TimeSeries:
	def __init__(self, device = torch.device("cpu")):
		self.device = device
		self.z0 = None

	def init_visualization(self):
		self.fig = plt.figure(figsize=(10, 4), facecolor='white')
		self.ax = self.fig.add_subplot(111, frameon=False)
		plt.show(block=False)

	def visualize(self, truth):
		self.ax.plot(truth[:,0], truth[:,1])

	def add_noise(self, traj_list, time_steps, noise_weight):
		n_samples = traj_list.size(0)

		# Add noise to all the points except the first point
		n_tp = len(time_steps) - 1
		noise = np.random.sample((n_samples, n_tp))
		noise = torch.Tensor(noise).to(self.device)

		traj_list_w_noise = traj_list.clone()
		# Dimension [:,:,0] is a time dimension -- do not add noise to that
		traj_list_w_noise[:,1:,0] += noise_weight * noise
		return traj_list_w_noise



class Periodic_1d(TimeSeries):
	def __init__(self, device = torch.device("cpu"), 
		init_freq = 0.3, init_amplitude = 1.,
		final_amplitude = 10., final_freq = 1., 
		z0 = 0.):
		"""
		If some of the parameters (init_freq, init_amplitude, final_amplitude, final_freq) is not provided, it is randomly sampled.
		For now, all the time series share the time points and the starting point.
		"""
		super(Periodic_1d, self).__init__(device)
		
		self.init_freq = init_freq
		self.init_amplitude = init_amplitude
		self.final_amplitude = final_amplitude
		self.final_freq = final_freq
		self.z0 = z0

	def sample_traj(self, time_steps, n_samples = 1, noise_weight = 1.,
		cut_out_section = None):
		"""
		Sample periodic functions. 
		"""
		traj_list = []
		for i in range(n_samples):
			init_freq = assign_value_or_sample(self.init_freq, [0.4,0.8])
			if self.final_freq is None:
				final_freq = init_freq
			else:
				final_freq = assign_value_or_sample(self.final_freq, [0.4,0.8])
			init_amplitude = assign_value_or_sample(self.init_amplitude, [0.,1.])
			final_amplitude = assign_value_or_sample(self.final_amplitude, [0.,1.])

			noisy_z0 = self.z0 + np.random.normal(loc=0., scale=0.1)

			traj = generate_periodic(time_steps, init_freq = init_freq, 
				init_amplitude = init_amplitude, starting_point = noisy_z0, 
				final_amplitude = final_amplitude, final_freq = final_freq)

			# Cut the time dimension
			traj = np.expand_dims(traj[:,1:], 0)
			traj_list.append(traj)

		# shape: [n_samples, n_timesteps, 2]
		# traj_list[:,:,0] -- time stamps
		# traj_list[:,:,1] -- values at the time stamps
		traj_list = np.array(traj_list)
		traj_list = torch.Tensor().new_tensor(traj_list, device = self.device)
		traj_list = traj_list.squeeze(1)

		traj_list = self.add_noise(traj_list, time_steps, noise_weight)
		return traj_list


from torch.utils.data import Dataset
import os
import io
import re
import pandas as pd
import json
from scipy.io import loadmat

class ATNDataset(Dataset):
	def __init__(self, path, json_file, normalization=True, mix_age=True, device='cpu'):
		self.path = path
		self.json_file = json_file
		self.normalization = normalization
		self.mix_age = mix_age
		self.device = device
		self.json_dict = self._load_json(json_file)
		self.data = self._load_data(path)
		self.valid_amyloid = []
		self.valid_tau = []
		self.valid_ctx = []
		self.valid_age = []
		self.max_visits = 4

		for subject in self.data:

			if self.max_visits < len(subject):
				self.max_visits = len(subject)

			amyloid = np.array([item['amyloid'] for item in subject])
			tau = np.array([item['tau'] for item in subject])
			ctx = np.array([item['ctx'] for item in subject])
			age = np.array([item['age'].item() for item in subject])

			if self.mix_age:
				start_age = age[0]
				max_age = age[-1]
				grid = np.arange(start_age, max_age + 1, 0.25)

				new_age = []
				for a in age:
					nearest = min(grid, key=lambda x: abs(x - a))
					new_age.append(nearest)
				age = np.array(new_age)

			self.valid_amyloid.append(amyloid)
			self.valid_tau.append(tau)
			self.valid_ctx.append(ctx)
			self.valid_age.append(age)

		self.valid_amyloid = np.concatenate(self.valid_amyloid, axis=0)
		self.valid_tau = np.concatenate(self.valid_tau, axis=0)
		self.valid_ctx = np.concatenate(self.valid_ctx, axis=0)
		self.max_age = torch.tensor(np.concatenate(self.valid_age)).max()
		self.min_age = torch.tensor(np.concatenate(self.valid_age)).min()

	def _load_json(self, json_file):
		with open(json_file, 'r') as f:
			return json.load(f)

	def _load_data(self, path):
		all_data = []

		for type_folder in os.listdir(path):
			type_path = os.path.join(path, type_folder)
			if not os.path.isdir(type_path):
				continue

			for subject_folder in os.listdir(type_path):
				subject_path = os.path.join(type_path, subject_folder)
				if not os.path.isdir(subject_path):
					continue

				subject_data = self._process_subject(subject_path)
				if subject_data:
					all_data.append(subject_data)

		return all_data

	def _process_subject(self, subject_path):
		subject_data = []

		time_folders = sorted(os.listdir(subject_path))
		for time_folder in time_folders:
			time_path = os.path.join(subject_path, time_folder)
			if not os.path.isdir(time_path):
				continue

			beta_file_path = os.path.join(time_path, 'beta')
			tau_file_path = os.path.join(time_path, 'tau')

			amyloid_data = self._process_amyloid(beta_file_path)
			tau_data = self._process_amyloid(tau_file_path)

			if amyloid_data is None or tau_data is None:
				continue

			ctx_file_path = os.path.join(time_path, 'catROIs_t1.mat')
			ctx_data = self._process_ctx(ctx_file_path)

			if ctx_data is None:
				continue

			label_path = os.path.join(time_path, 'label.pt')
			label = torch.load(label_path, weights_only=True)
			age_path = os.path.join(time_path, 'age.pt')
			age = torch.load(age_path, weights_only=True)

			combined_data = {
				'amyloid': amyloid_data['Mean'].values,
				'tau': tau_data['Mean'].values,
				'ctx': ctx_data['value'].values,
				'age': age,
				'label': label,
				'subject_id': os.path.basename(subject_path)
			}
			subject_data.append(combined_data)

		return subject_data

	def _process_amyloid(self, file_path):
		if not os.path.exists(file_path):
			return None

		with open(file_path, 'r') as file:
			content = file.read()
			table_start = re.search(r'# ColHeaders\s+Index', content).end()
			table_data = content[table_start:].strip()

			col_separator = '\s+'
			column_names = ['Index', 'SegId', 'NVoxels', 'Volume_mm3', 'StructName', 'Mean', 'StdDev', 'Min', 'Max',
							'Range']
			data = pd.read_csv(io.StringIO(table_data), sep=col_separator, names=column_names)

			row_index = data[data['SegId'] == '11101'].index

			if len(row_index) > 0:
				data = data.iloc[row_index[0]:]
				data = data.drop('Index', axis=1)
			data[['SegId', 'NVoxels', 'Volume_mm3', 'Mean', 'StdDev', 'Min', 'Max', 'Range']] = data[[
				'SegId', 'NVoxels', 'Volume_mm3', 'Mean', 'StdDev', 'Min', 'Max', 'Range']].apply(pd.to_numeric,
																								   errors='coerce')

			return data[['SegId', 'Mean']]

	def _process_ctx(self, file_path):
		if not os.path.exists(file_path):
			return None

		data = loadmat(file_path)
		a = data['S']['aparc_a2009s'][0][0][0][0][1][:].flatten()
		b = data['S']['aparc_a2009s'][0][0][0][0][4][0][0][0].flatten()

		data = pd.DataFrame({'SegId': a, 'value': b})
		data = data.iloc[2:]
		data = data.drop(84)
		data = data.drop(85).reset_index(drop=True)

		def map_to_AAL_ID(segid):
			if isinstance(segid, np.ndarray):
				segid = segid[0]
			segid = segid.strip('[]')

			side = 'lh' if segid.startswith('l') else 'rh'
			core_name = segid[1:]

			for entry in self.json_dict:
				name = entry.get('name', '')

				if name.startswith(f'ctx_{side}_') and name[7:] == core_name:
					return entry.get('AAL_ID', None)

			return None

		data['AAL_ID'] = data['SegId'].apply(lambda x: map_to_AAL_ID(x))

		data['AAL_ID'] = pd.to_numeric(data['AAL_ID'], errors='coerce')

		df_sorted = data.sort_values(by='AAL_ID').reset_index(drop=True)

		return df_sorted[['SegId', 'value']]

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		subject_data = self.data[idx]

		num_time_points = len(subject_data)

		padding_length = max(0, self.max_visits - num_time_points)

		valid_amyloid_matrix = np.array([item['amyloid'] for item in subject_data])
		valid_tau_matrix = np.array([item['tau'] for item in subject_data])
		valid_ctx_matrix = np.array([item['ctx'] for item in subject_data])

		if self.normalization:
			valid_amyloid_matrix = (valid_amyloid_matrix - np.min(self.valid_amyloid, axis=0)) / (np.max(self.valid_amyloid, axis=0) - np.min(self.valid_amyloid, axis=0))
			valid_tau_matrix = (valid_tau_matrix - np.min(self.valid_tau, axis=0)) / (np.max(self.valid_tau, axis=0) - np.min(self.valid_tau, axis=0))
			valid_ctx_matrix = 1 - ((valid_ctx_matrix - np.min(self.valid_ctx, axis=0)) / (np.max(self.valid_ctx, axis=0) - np.min(self.valid_ctx, axis=0)))

		data_matrix = np.concatenate([valid_amyloid_matrix, valid_tau_matrix, valid_ctx_matrix], axis=1)

		num_time_points = len(subject_data)
		padding_length = max(0, self.max_visits - num_time_points)

		data_padded = np.pad(data_matrix, ((0, padding_length), (0, 0)), mode='constant', constant_values=0)

		mask = np.ones((num_time_points, data_matrix.shape[1]))
		mask_padded = np.pad(mask, ((0, padding_length), (0, 0)), mode='constant', constant_values=0)

		raw_age = np.array([item['age'].item() for item in subject_data])
		norm_age = (raw_age - self.min_age.item()) / (self.max_age.item() - self.min_age.item() + 1e-5)
		age_padded = np.pad(norm_age, (0, padding_length), mode='constant', constant_values=0)

		label_id = subject_data[0]['label'].item()
		label_onehot = np.zeros(5)
		label_onehot[int(label_id)] = 1.0

		return {
			"observed_data": torch.tensor(data_padded, dtype=torch.float32).to(self.device),
			"observed_mask": torch.tensor(mask_padded, dtype=torch.float32).to(self.device),
			"observed_tp": torch.tensor(age_padded, dtype=torch.float32).to(self.device),
			"label_onehot": torch.tensor(label_onehot, dtype=torch.float32).to(self.device)
		}

