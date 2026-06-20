import time

import torch
import numpy as np
import argparse
import os
import sys
import collections
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.stats import wasserstein_distance_nd
import torch.nn.functional as F

from Utils.io_utils import load_yaml_config, instantiate_from_config
from Utils.Data_utils.atn_dataset import ATNDataset

ROOT_DIR = Path(__file__).resolve().parent


class Evaluator:
    def metrics(self, all_samples, test_x):
        if (all_samples.min() < -1e+10) or (all_samples.max() > 1e+10):
            print('sampled values are exploded !!!')
            return None

        sampled_data_np = all_samples.detach().cpu().numpy()
        real_data_np = test_x.detach().cpu().numpy()

        wd = max(0.0, float(wasserstein_distance_nd(sampled_data_np, real_data_np)))

        sampled_prob = F.softmax(all_samples, dim=1)
        real_prob = F.softmax(test_x, dim=1)

        m = 0.5 * (sampled_prob + real_prob)

        def safe_kld(p, q):
            p = p + 1e-10
            q = q + 1e-10
            return (p * (p.log() - q.log())).sum(dim=1)

        jsd_val = 0.5 * (safe_kld(sampled_prob, m) + safe_kld(real_prob, m))
        jsd_mean = jsd_val.mean().item()

        mse = F.mse_loss(all_samples, test_x)
        rmse = torch.sqrt(mse).item()

        gt_range = test_x.max() - test_x.min()
        if gt_range > 1e-8:
            nrmse = rmse / gt_range.item()
        else:
            nrmse = float("inf")

        return {
            'NRMSE': nrmse,
            'RMSE': rmse,
            'JSD': jsd_mean,
            'WD': wd,
            'CosSim': self.cosine_similarity_metric(all_samples, test_x),
        }

    def trajectory_metrics(self, generated_trajectories, real_trajectories):
        all_ade = []

        if len(generated_trajectories) != len(real_trajectories):
            print("Warning: Trajectory count mismatch!")
            return {'ADE': 0.0}

        for gen_traj, real_traj in zip(generated_trajectories, real_trajectories):
            gen_traj = gen_traj.cpu()
            real_traj = real_traj.cpu()

            if gen_traj.shape != real_traj.shape:
                min_len = min(gen_traj.shape[0], real_traj.shape[0])
                gen_traj = gen_traj[:min_len]
                real_traj = real_traj[:min_len]

            if gen_traj.shape[0] == 0:
                continue

            displacement_errors = torch.norm(gen_traj - real_traj, p=2, dim=1)
            ade = torch.mean(displacement_errors)
            all_ade.append(ade.item())

        mean_ade = np.mean(all_ade) if all_ade else 0.0

        return {'ADE': mean_ade}
    
    def cosine_similarity_metric(self, all_samples, test_x):
        if all_samples.shape != test_x.shape:
            return 0.0

        if all_samples.shape[1] % 3 != 0:
            return 0.0

        samples = all_samples.detach().cpu()
        real = test_x.detach().cpu()
        region_dim = samples.shape[1] // 3

        sample_blocks = [samples[:, :region_dim], samples[:, region_dim:2 * region_dim], samples[:, 2 * region_dim:]]
        real_blocks = [real[:, :region_dim], real[:, region_dim:2 * region_dim], real[:, 2 * region_dim:]]

        cos_values = []
        for idx, (sample_block, real_block) in enumerate(zip(sample_blocks, real_blocks)):
            min_vals = real_block.min(dim=0).values
            max_vals = real_block.max(dim=0).values
            denom = (max_vals - min_vals)
            denom = torch.where(denom > 1e-8, denom, torch.ones_like(denom))

            sample_norm = (sample_block - min_vals) / denom
            real_norm = (real_block - min_vals) / denom

            if idx == 2:
                sample_norm = 1.0 - sample_norm
                real_norm = 1.0 - real_norm

            cos = F.cosine_similarity(sample_norm, real_norm, dim=1)
            if cos.numel() > 0:
                cos_values.append(cos.mean().item())

        if not cos_values:
            return 0.0
        return float(np.mean(cos_values))



def simple_collate_fn(batch):
    data_list = [item[0] for item in batch]
    mask_list = [item[1] for item in batch]

    batch_data = torch.stack(data_list)
    batch_mask = torch.stack(mask_list)

    return batch_data, batch_mask



def evaluate_single_model(ckpt_path, config, test_loader, denorm_func, device, args):
    print(f"\n--- Evaluating Checkpoint: {os.path.basename(ckpt_path)} ---")

    model = instantiate_from_config(config['model']).to(device)

    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        if 'ema' in checkpoint:
            state_dict = checkpoint['ema']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('ema_model.'):
                new_key = k.replace('ema_model.', '')
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict, strict=False)
    except Exception as e:
        print(f"Error loading {ckpt_path}: {e}")
        return None

    model.eval()
    evaluator = Evaluator()

    all_gen_points = []
    all_true_points = []
    all_gen_trajs = []
    all_true_trajs = []

    DATA_DIM = 444

    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (batch_x, batch_mask) in enumerate(tqdm(test_loader, desc="Sampling", leave=False)):
            batch_x = batch_x.to(device)
            batch_mask = batch_mask.to(device)

            infill_mask = torch.zeros_like(batch_x).bool()
            infill_mask[:, :, DATA_DIM:] = True

            target = torch.zeros_like(batch_x)
            target[:, :, DATA_DIM:] = batch_x[:, :, DATA_DIM:]

            model_kwargs = {'coef': 0.1, 'learning_rate': 0.1}

            samples = model.sample_infill(
                shape=batch_x.shape,
                target=target,
                partial_mask=infill_mask,
                clip_denoised=True,
                model_kwargs=model_kwargs
            )

            generated_norm = samples[:, :, :DATA_DIM]
            true_data_norm = batch_x[:, :, :DATA_DIM]

            B = generated_norm.shape[0]
            for i in range(B):
                valid_indices = torch.where(batch_mask[i] > 0)[0]

                if len(valid_indices) == 0: continue

                generated_traj = denorm_func(generated_norm[i][valid_indices])
                t_traj = denorm_func(true_data_norm[i][valid_indices])

                all_gen_trajs.append(generated_traj)
                all_true_trajs.append(t_traj)

                all_gen_points.append(generated_traj)
                all_true_points.append(t_traj)

    if len(all_gen_points) == 0:
        print("Warning: No valid data points found.")
        return None
    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time}")

    all_gen_tensor = torch.cat(all_gen_points, dim=0)
    all_true_tensor = torch.cat(all_true_points, dim=0)

    dist_metrics = evaluator.metrics(all_gen_tensor, all_true_tensor)
    if dist_metrics is None:
        return None
    traj_metrics = evaluator.trajectory_metrics(all_gen_trajs, all_true_trajs)

    final_metrics = {**dist_metrics, **traj_metrics}

    result_str = ", ".join([f"{k}: {v:.4f}" for k, v in final_metrics.items()])
    print(f"Results -> {result_str}")

    return final_metrics



def test_all(args):
    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    config = load_yaml_config(args.config)

    print(f"Loading Test Dataset from {args.test_path}...")

    config['dataloader']['test_dataset']['params']['data_root'] = args.test_path

    test_ds = ATNDataset(
        data_root=args.test_path,
        json_file=args.json_file,
        window=4,
        period='test',
        save2npy=False,
        neg_one_to_one=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=simple_collate_fn
    )

    min_a = torch.tensor(test_ds.min_a, device=device).float()
    range_a = torch.tensor(test_ds.range_a, device=device).float()
    min_t = torch.tensor(test_ds.min_t, device=device).float()
    range_t = torch.tensor(test_ds.range_t, device=device).float()
    min_c = torch.tensor(test_ds.min_c, device=device).float()
    range_c = torch.tensor(test_ds.range_c, device=device).float()

    def denormalize_batch(data_tensor):
        """
        data_tensor: [..., 444] -> [-1, 1]
        Structure: 148 A, 148 T, 148 C
        """
        curr_a = data_tensor[..., :148]
        curr_t = data_tensor[..., 148:296]
        curr_c = data_tensor[..., 296:]

        real_a = curr_a * range_a + min_a
        real_t = curr_t * range_t + min_t

        norm_01_c = (curr_c + 1) / 2
        real_c = (1.0 - norm_01_c) * range_c + min_c

        return torch.cat([real_a, real_t, real_c], dim=-1)

    ckpt_list = args.ckpt
    agg_results = collections.defaultdict(list)

    print(f"Found {len(ckpt_list)} checkpoints to evaluate.")

    for ckpt in ckpt_list:
        if not os.path.exists(ckpt):
            print(f"Skipping non-existent file: {ckpt}")
            continue

        metrics = evaluate_single_model(ckpt, config, test_loader, denormalize_batch, device, args)

        if metrics is not None:
            for k, v in metrics.items():
                agg_results[k].append(v)

    if not agg_results:
        print("No results collected.")
        return

    print("\n" + "=" * 60)
    print(f"FINAL REPORT (Evaluated {len(ckpt_list)} checkpoints)")
    print("=" * 60)
    print(f"{'Metric':<10} | {'Mean':<10} | {'Std':<10}")
    print("-" * 60)

    for metric, values in agg_results.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"{metric:<10} | {mean_val:.4f}     | {std_val:.4f}")

    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=str(ROOT_DIR / 'Config' / 'atn.yaml'))

    parser.add_argument('--ckpt', type=str, nargs='+', required=True, help='Path(s) to checkpoint(s)')

    parser.add_argument('--test_path', type=str, default=str(ROOT_DIR / 'data' / 'ALL' / 'test'), help='Path to test data')
    parser.add_argument('--json_file', type=str, default=str(ROOT_DIR / 'data' / 'TABLE_Destrieux.json'))
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    test_all(args)
