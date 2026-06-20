import time

import torch
import numpy as np
import argparse
import os
import scipy.stats
from pathlib import Path
from torch.utils.data import DataLoader
from scipy.spatial.distance import jensenshannon
from torch.distributions.normal import Normal
from scipy.stats import wasserstein_distance_nd
import torch.nn.functional as F
import collections

import lib.utils as utils
from lib.create_latent_ode_model import create_LatentODE_model
from generate_timeseries import ATNDataset

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
            "NRMSE": nrmse,
            "RMSE": rmse,
            "JSD": jsd_mean,
            "WD": wd,
            "CosSim": self.cosine_similarity_metric(all_samples, test_x),
        }
    
    def cosine_similarity_metric(self, all_samples, test_x):
        if all_samples.shape != test_x.shape:
            return 0.0

        if all_samples.shape[1] % 3 != 0:
            return 0.0

        samples = all_samples.detach().cpu()
        real = test_x.detach().cpu()
        region_dim = samples.shape[1] // 3

        sample_blocks = [
            samples[:, :region_dim],
            samples[:, region_dim:2 * region_dim],
            samples[:, 2 * region_dim:],
        ]
        real_blocks = [
            real[:, :region_dim],
            real[:, region_dim:2 * region_dim],
            real[:, 2 * region_dim:],
        ]

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

    def trajectory_metrics(self, generated_trajectories, real_trajectories):
        all_ade = []

        if len(generated_trajectories) != len(real_trajectories):
            print("Warning: Trajectory count mismatch!")
            return 0.0

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

        return mean_ade


def atn_collate_fn(batch):
    device = batch[0]['observed_data'].device
    D = batch[0]['observed_data'].shape[-1]
    labels = []
    all_tp = []
    for item in batch:
        mask = item['observed_mask']
        tp = item['observed_tp']
        valid_idx = (mask.sum(dim=-1) > 0)
        valid_tp = tp[valid_idx]
        all_tp.append(valid_tp)
        labels.append(item['label_onehot'])

    if len(all_tp) > 0:
        union_tp = torch.cat(all_tp)
        union_tp = torch.unique(union_tp, sorted=True)
    else:
        union_tp = torch.tensor([0.], device=device)

    union_tp = union_tp.to(device)
    B = len(batch)
    T_union = len(union_tp)

    out_data = torch.zeros(B, T_union, D, device=device)
    out_mask = torch.zeros(B, T_union, D, device=device)

    for b, item in enumerate(batch):
        curr_tp = item['observed_tp']
        curr_data = item['observed_data']
        curr_mask = item['observed_mask']
        valid_idx = (curr_mask.sum(dim=-1) > 0)
        curr_valid_tp = curr_tp[valid_idx]
        curr_valid_data = curr_data[valid_idx]

        if len(curr_valid_tp) == 0: continue

        idx_in_union = torch.searchsorted(union_tp, curr_valid_tp)
        out_data[b, idx_in_union] = curr_valid_data
        out_mask[b, idx_in_union] = 1.0

    out_labels = torch.stack(labels)
    return {
        "observed_data": out_data,
        "observed_tp": union_tp,
        "observed_mask": out_mask,
        "data_to_predict": out_data.clone(),
        "tp_to_predict": union_tp.clone(),
        "mask_predicted_data": out_mask.clone(),
        "labels": out_labels,
        "mode": "interp"
    }



def run_single_evaluation(ckpt_path, test_loader, denormalize_batch_func, args, device):
    print(f"\nEvaluating Checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)
    ckpt_args = checkpoint['args']
    state_dict = checkpoint['state_dict']

    ckpt_args.condition_dim = 5
    input_dim = 0
    for key, val in state_dict.items():
        if 'decoder' in key and 'bias' in key:
            input_dim = val.shape[0]
            break
    if input_dim == 0: input_dim = 150

    from torch.distributions.normal import Normal
    z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.0]).to(device))
    obsrv_std = torch.Tensor([0.01]).to(device)

    model = create_LatentODE_model(
        ckpt_args, input_dim, z0_prior=z0_prior, obsrv_std=obsrv_std, device=device,
        classif_per_tp=getattr(ckpt_args, 'classif_per_tp', False),
        n_labels=getattr(ckpt_args, 'n_labels', 1),
        condition_dim=5
    )
    model.load_state_dict(state_dict)
    model.eval()

    evaluator = Evaluator()

    all_gen_points_list = []
    all_true_points_list = []
    all_gen_traj_list = []
    all_true_traj_list = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            labels = batch['labels']
            time_steps = batch['observed_tp']
            true_data = batch['observed_data']
            mask = batch['observed_mask']

            sample_paths = model.sample_traj_from_prior(
                time_steps,
                cond=labels,
                n_traj_samples=args.n_samples
            )

            sample_mean = sample_paths.mean(dim=0)

            batch_size = true_data.size(0)

            for i in range(batch_size):
                patient_time_mask = (mask[i].sum(dim=-1) > 0)
                if patient_time_mask.sum() == 0: continue

                generated_norm = sample_mean[i][patient_time_mask]
                t_traj_norm = true_data[i][patient_time_mask]

                generated_real = denormalize_batch_func(generated_norm)
                t_traj_real = denormalize_batch_func(t_traj_norm)

                all_gen_traj_list.append(generated_real)
                all_true_traj_list.append(t_traj_real)
                all_gen_points_list.append(generated_real)
                all_true_points_list.append(t_traj_real)

    all_gen_tensor = torch.cat(all_gen_points_list, dim=0)
    all_true_tensor = torch.cat(all_true_points_list, dim=0)

    ade = evaluator.trajectory_metrics(all_gen_traj_list, all_true_traj_list)

    metrics = evaluator.metrics(all_gen_tensor, all_true_tensor)
    if metrics is None:
        return None
    metrics["ADE"] = ade

    print(f"  -> Results: RMSE={metrics['RMSE']:.4f}, WD={metrics['WD']:.4f}, ADE={ade:.4f}, CosSim={metrics['CosSim']:.4f}")

    return metrics



def evaluate_all(args):
    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    print("Loading Data (Once)...")
    full_dataset = ATNDataset(
        path=args.data_path,
        json_file=args.json_file,
        normalization=True,
        mix_age=True,
        device=device
    )
    test_loader = DataLoader(
        full_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=atn_collate_fn
    )

    print("Preparing Denormalization Parameters...")
    dim_a = full_dataset.valid_amyloid.shape[1]
    dim_t = full_dataset.valid_tau.shape[1]

    min_a = torch.tensor(np.min(full_dataset.valid_amyloid, axis=0), dtype=torch.float32).to(device)
    max_a = torch.tensor(np.max(full_dataset.valid_amyloid, axis=0), dtype=torch.float32).to(device)
    rng_a = max_a - min_a
    rng_a[rng_a == 0] = 1.0

    min_t = torch.tensor(np.min(full_dataset.valid_tau, axis=0), dtype=torch.float32).to(device)
    max_t = torch.tensor(np.max(full_dataset.valid_tau, axis=0), dtype=torch.float32).to(device)
    rng_t = max_t - min_t
    rng_t[rng_t == 0] = 1.0

    min_c = torch.tensor(np.min(full_dataset.valid_ctx, axis=0), dtype=torch.float32).to(device)
    max_c = torch.tensor(np.max(full_dataset.valid_ctx, axis=0), dtype=torch.float32).to(device)
    rng_c = max_c - min_c
    rng_c[rng_c == 0] = 1.0

    def denormalize_batch(data_tensor):
        curr_a = data_tensor[..., :dim_a]
        curr_t = data_tensor[..., dim_a: dim_a + dim_t]
        curr_c = data_tensor[..., dim_a + dim_t:]

        real_a = curr_a * rng_a + min_a
        real_t = curr_t * rng_t + min_t
        real_c = (1.0 - curr_c) * rng_c + min_c

        return torch.cat([real_a, real_t, real_c], dim=-1)

    metrics_history = collections.defaultdict(list)

    ckpt_list = args.ckpt
    if isinstance(ckpt_list, str):
        ckpt_list = [ckpt_list]

    for ckpt_path in ckpt_list:
        try:
            res = run_single_evaluation(ckpt_path, test_loader, denormalize_batch, args, device)
            if res is None:
                continue
            for k, v in res.items():
                if v is not None:
                    metrics_history[k].append(v)
        except Exception as e:
            print(f"Error evaluating {ckpt_path}: {e}")
            continue

    print("\n" + "=" * 50)
    print(f"FINAL AGGREGATED REPORT (N={len(ckpt_list)} models)")
    print("=" * 50)
    print(f"{'Metric':<10} | {'Mean':<10} | {'Std':<10}")
    print("-" * 50)

    for metric_name, values in metrics_history.items():
        if len(values) > 0:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{metric_name:<10} | {mean_val:.4f}     | {std_val:.4f}")
        else:
            print(f"{metric_name:<10} | N/A        | N/A")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, nargs='+', required=True,
                        help='Path(s) to checkpoint(s). Separate with spaces.')
    parser.add_argument('--data_path', type=str, default=str(ROOT_DIR / 'data' / 'ALL' / 'test'),
                        help='Path to pickle data')
    parser.add_argument('--json_file', type=str, default=str(ROOT_DIR / 'data' / 'TABLE_Destrieux.json'),
                        help='Path to json file')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Number of trajectories to sample per patient for averaging')

    start_time = time.time()
    args = parser.parse_args()
    evaluate_all(args)
    end_time = time.time()
    print(f"\nEvaluation Time: {end_time - start_time}")
