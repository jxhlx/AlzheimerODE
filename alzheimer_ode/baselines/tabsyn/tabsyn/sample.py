import torch

import argparse
import warnings
import time
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from tabsyn.model import MLPDiffusion, Model
from tabsyn.latent_utils import get_input_generate, recover_data, split_num_cat_target
from tabsyn.diffusion_utils import sample

warnings.filterwarnings('ignore')


def main(args):
    dataname = args.dataname
    device = args.device
    steps = args.steps
    save_path = args.save_path

    train_z, _, _, ckpt_path, info, num_inverse, cat_inverse = get_input_generate(args)
    in_dim = train_z.shape[1] 

    mean = train_z.mean(0)

    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    
    model = Model(denoise_fn = denoise_fn, hid_dim = train_z.shape[1]).to(device)

    model.load_state_dict(torch.load(f'{ckpt_path}/model.pt'))

    '''
        Generating samples    
    '''
    start_time = time.time()

    num_samples = train_z.shape[0]
    sample_dim = in_dim

    def xsample(X_test, split='val'):
        sample_device = torch.device(device)
        batch_size = 1000

        if split == 'val':
            y = np.array(X_test['age'])
            X_num_shape = len([c for c in X_test.columns.tolist() if 'Node' in c])
            X_cat = np.stack([np.array(X_test[c]) for c in X_test.columns.tolist() if 'label' in c], 1)

        sample_gen = torch.zeros(y.shape[0], X_num_shape).cpu()
        remain_sample_id = torch.arange(len(sample_gen))
        org_X_cat = torch.from_numpy(X_cat.astype(int)).to(sample_device)
        org_y = torch.from_numpy(y.astype(int)).to(sample_device)
        max_iter = 10
        sample_iter = 0
        while len(remain_sample_id) > 0:
            if sample_iter > max_iter: break
            sample_iter += 1
            print(f'Remain {len(remain_sample_id)} Samples being generating')
            X_cat = org_X_cat[remain_sample_id.to(sample_device)]
            y = org_y[remain_sample_id.to(sample_device)]
            # Create synthetic data
            x_next = sample(model.denoise_fn_D, num_samples, sample_dim)
            x_next = x_next * 2 + mean.to(sample_device)

            syn_data = x_next.float().cpu().numpy()
            syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, cat_inverse, args.device)

            syn_df = recover_data(syn_num, syn_cat, syn_target, info)

            idx_name_mapping = info['idx_name_mapping']
            idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

            syn_df.rename(columns=idx_name_mapping, inplace=True)
            gen_data = syn_df

            gen_y = np.array(gen_data['age'])
            gen_X_cat = np.stack([np.array(gen_data[c]) for c in gen_data.columns.tolist() if 'label' in c], 1)

            X_num = np.stack([np.array(gen_data[c]) for c in gen_data.columns.tolist() if 'Node' in c], 1)

            X_num = torch.from_numpy(X_num.astype(float))
            gen_X_cat = torch.from_numpy(gen_X_cat.astype(int)).to(sample_device)
            # gen_y = torch.from_numpy(np.round(gen_y.astype(float), decimals=1)).to(device)
            gen_y = torch.from_numpy(gen_y.astype(int)).to(sample_device)
            batch_mask_cat_cond = torch.any(X_cat[:, None] != gen_X_cat[None, :], dim=-1).cpu()  # N_y x N_sample
            batch_mask_y_cond = (y[:, None] != gen_y[None, :]).cpu()  # N_y x N_sample
            batch_mask_cond = torch.logical_or(batch_mask_y_cond, batch_mask_cat_cond)
            mask_cond = batch_mask_cond.all(dim=1)  # N_y
            all_yi = torch.where(~mask_cond)[0]
            yi_ls, samplei_ls = torch.where(~batch_mask_cond)
            sample_ind = []
            for yi, samplei in zip(yi_ls, samplei_ls):
                if yi in all_yi:
                    sample_ind.append(samplei)
                    all_yi = all_yi[all_yi != yi]

            sample_ind = torch.LongTensor(sample_ind)
            sample_gen[remain_sample_id[~mask_cond]] = X_num[sample_ind].float()
            remain_sample_id = remain_sample_id[mask_cond]

        os.makedirs(args.matched_output_dir, exist_ok=True)
        np.save(Path(args.matched_output_dir) / f'X_num_{split}', sample_gen.numpy())

    if args.condition_path and Path(args.condition_path).exists():
        xt = pd.read_csv(args.condition_path)
        xsample(xt)

    x_next = sample(model.denoise_fn_D, num_samples, sample_dim)
    x_next = x_next * 2 + mean.to(device)

    syn_data = x_next.float().cpu().numpy()
    syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, cat_inverse, args.device) 

    syn_df = recover_data(syn_num, syn_cat, syn_target, info)

    idx_name_mapping = info['idx_name_mapping']
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

    syn_df.rename(columns = idx_name_mapping, inplace=True)
    syn_df.to_csv(save_path, index = False)
    
    end_time = time.time()
    print('Time:', end_time - start_time)

    print('Saving sampled data to {}'.format(save_path))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generation')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU index.')
    parser.add_argument('--epoch', type=int, default=None, help='Epoch.')
    parser.add_argument('--steps', type=int, default=None, help='Number of function evaluations.')
    parser.add_argument('--condition_path', type=str, default=str(ROOT_DIR / 'data' / 'atn' / 'test.csv'))
    parser.add_argument('--matched_output_dir', type=str, default=str(ROOT_DIR / 'outputs' / 'tabsyn' / 'ATN' / '1'))

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'
