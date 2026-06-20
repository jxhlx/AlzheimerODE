import os
import torch
import numpy as np
import pandas as pd
import json
import re
import io
from scipy.io import loadmat
from torch.utils.data import Dataset


class ATNDataset(Dataset):
    def __init__(self, data_root, json_file='data/TABLE_Destrieux.json', window=4, period='train', save2npy=False, neg_one_to_one=False,
                 output_dir=None):
        self.path = data_root
        self.json_file = json_file
        self.max_visits = window
        self.device = 'cpu'
        self.period = period

        self.json_dict = self._load_json(json_file)

        self.data = self._load_data(self.path)

        self.valid_amyloid = []
        self.valid_tau = []
        self.valid_ctx = []
        self.valid_age = []

        for subject in self.data:
            amyloid = np.array([item['amyloid'] for item in subject])
            tau = np.array([item['tau'] for item in subject])
            ctx = np.array([item['ctx'] for item in subject])
            age = np.array(
                [item['age'].item() if isinstance(item['age'], torch.Tensor) else item['age'] for item in subject])

            mix_age = True
            if mix_age:
                start_age = age[0]
                max_age_subj = age[-1]
                grid = np.arange(start_age, max_age_subj + 1, 0.25)
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

        self.min_a = np.min(self.valid_amyloid, axis=0)
        self.max_a = np.max(self.valid_amyloid, axis=0)
        self.range_a = self.max_a - self.min_a
        self.range_a[self.range_a == 0] = 1.0

        self.min_t = np.min(self.valid_tau, axis=0)
        self.max_t = np.max(self.valid_tau, axis=0)
        self.range_t = self.max_t - self.min_t
        self.range_t[self.range_t == 0] = 1.0

        self.min_c = np.min(self.valid_ctx, axis=0)
        self.max_c = np.max(self.valid_ctx, axis=0)
        self.range_c = self.max_c - self.min_c
        self.range_c[self.range_c == 0] = 1.0

        self.max_age = np.max(np.concatenate(self.valid_age))
        self.min_age = np.min(np.concatenate(self.valid_age))

        self.feature_dim = self.valid_amyloid.shape[1] + self.valid_tau.shape[1] + self.valid_ctx.shape[1]
        self.total_dim = self.feature_dim + 1 + 5

    def _load_json(self, json_file):
        with open(json_file, 'r') as f:
            return json.load(f)

    def _load_data(self, path):
        all_data = []
        if not os.path.exists(path):
            print(f"Path not found: {path}")
            return []

        for type_folder in sorted(os.listdir(path)):
            type_path = os.path.join(path, type_folder)
            if not os.path.isdir(type_path):
                continue

            for subject_folder in sorted(os.listdir(type_path)):
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
            ctx_file_path = os.path.join(time_path, 'catROIs_t1.mat')

            amyloid_df = self._process_amyloid(beta_file_path)
            tau_df = self._process_amyloid(tau_file_path)
            ctx_df = self._process_ctx(ctx_file_path)

            if amyloid_df is None or tau_df is None or ctx_df is None:
                continue

            amy_val = amyloid_df['Mean'].values
            tau_val = tau_df['Mean'].values
            ctx_val = ctx_df['value'].values

            label_path = os.path.join(time_path, 'label.pt')
            age_path = os.path.join(time_path, 'age.pt')

            if not os.path.exists(label_path) or not os.path.exists(age_path):
                continue

            label = torch.load(label_path)
            age = torch.load(age_path)

            combined_data = {
                'amyloid': amy_val,
                'tau': tau_val,
                'ctx': ctx_val,
                'age': age,
                'label': label,
                'subject_id': os.path.basename(subject_path)
            }
            subject_data.append(combined_data)

        return subject_data

    def _process_amyloid(self, file_path):
        if not os.path.exists(file_path): return None
        try:
            with open(file_path, 'r') as file:
                content = file.read()
                match = re.search(r'# ColHeaders\s+Index', content)
                if not match: return None
                table_start = match.end()
                table_data = content[table_start:].strip()

                col_separator = r'\s+'
                column_names = ['Index', 'SegId', 'NVoxels', 'Volume_mm3', 'StructName', 'Mean', 'StdDev', 'Min', 'Max',
                                'Range']
                data = pd.read_csv(io.StringIO(table_data), sep=col_separator, names=column_names, engine='python')

                row_index = data[data['SegId'] == '11101'].index
                if len(row_index) > 0:
                    data = data.iloc[row_index[0]:]
                    data = data.drop('Index', axis=1)

                cols = ['SegId', 'NVoxels', 'Volume_mm3', 'Mean', 'StdDev', 'Min', 'Max', 'Range']
                data[cols] = data[cols].apply(pd.to_numeric, errors='coerce')

                if len(data) > 148:
                    data = data.iloc[-148:]

                return data[['SegId', 'Mean']]
        except:
            return None

    def _process_ctx(self, file_path):
        if not os.path.exists(file_path): return None
        try:
            data = loadmat(file_path)
            a = data['S']['aparc_a2009s'][0][0][0][0][1][:].flatten()
            b = data['S']['aparc_a2009s'][0][0][0][0][4][0][0][0].flatten()

            data = pd.DataFrame({'SegId': a, 'value': b})
            data = data.iloc[2:]
            if 84 in data.index: data = data.drop(84)
            if 85 in data.index: data = data.drop(85).reset_index(drop=True)

            def map_to_AAL_ID(segid):
                if isinstance(segid, np.ndarray): segid = segid[0]
                segid = str(segid).strip("[]'")
                side = 'lh' if segid.startswith('l') else 'rh'
                core_name = segid[1:]
                for entry in self.json_dict:
                    name = entry.get('name', '')
                    if name.startswith(f'ctx_{side}_') and name[7:] == core_name:
                        return entry.get('AAL_ID', None)
                return None

            data['AAL_ID'] = data['SegId'].apply(lambda x: map_to_AAL_ID(x))
            data['AAL_ID'] = pd.to_numeric(data['AAL_ID'], errors='coerce')
            df_sorted = data.sort_values(by='AAL_ID').dropna(subset=['AAL_ID']).reset_index(drop=True)

            return df_sorted[['SegId', 'value']]
        except:
            return None

    def __getitem__(self, idx):
        subject_data = self.data[idx]
        num_time_points = len(subject_data)

        raw_amyloid = np.array([item['amyloid'] for item in subject_data])
        raw_tau = np.array([item['tau'] for item in subject_data])
        raw_ctx = np.array([item['ctx'] for item in subject_data])

        raw_age = np.array(
            [item['age'].item() if isinstance(item['age'], torch.Tensor) else item['age'] for item in subject_data])
        raw_label = subject_data[0]['label'].item() if isinstance(subject_data[0]['label'], torch.Tensor) else \
        subject_data[0]['label']

        norm_amyloid = (raw_amyloid - self.min_a) / self.range_a
        norm_tau = (raw_tau - self.min_t) / self.range_t
        norm_ctx = 1 - ((raw_ctx - self.min_c) / self.range_c)

        norm_age = (raw_age - self.min_age) / (self.max_age - self.min_age + 1e-5)

        label_onehot = np.zeros(5)
        label_onehot[int(raw_label)] = 1.0

        biomarkers = np.concatenate([norm_amyloid, norm_tau, norm_ctx], axis=1)

        age_vec = norm_age[:, None]

        label_vec = np.tile(label_onehot[None, :], (num_time_points, 1))

        combined = np.concatenate([biomarkers, age_vec, label_vec], axis=1)

        max_len = self.max_visits
        out_data = np.zeros((max_len, self.total_dim))
        out_mask = np.zeros((max_len,))

        curr_len = min(num_time_points, max_len)
        out_data[:curr_len, :] = combined[:curr_len, :]
        out_mask[:curr_len] = 1.0

        return torch.from_numpy(out_data).float(), torch.from_numpy(out_mask).float()

    def __len__(self):
        return len(self.data)
