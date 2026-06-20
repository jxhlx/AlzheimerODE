import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wasserstein_distance_nd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import src
from ef_vfm.modules.main_modules import UniModMLP
from ef_vfm.models.flow_model import ExpVFM
from ef_vfm.trainer import split_num_cat_target
from utils_train import EFVFMDataset


def cosine_similarity_metric(all_samples, test_x):
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


def compute_metrics(all_samples: torch.Tensor, test_x: torch.Tensor):
    if (all_samples.min() < -1e+10) or (all_samples.max() > 1e+10):
        print("sampled values are exploded !!!")
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

    displacement_errors = torch.norm(all_samples - test_x, p=2, dim=1)
    ade = torch.mean(displacement_errors).item()
    cos = cosine_similarity_metric(all_samples, test_x)
    return {
        "NRMSE": nrmse,
        "RMSE": rmse,
        "JSD": jsd_mean,
        "WD": wd,
        "ADE": ade,
        "CosSim": cos,
    }


def load_test_data(data_dir):
    x_num_path = os.path.join(data_dir, "X_num_test.npy")
    x_cat_path = os.path.join(data_dir, "X_cat_test.npy")
    y_path = os.path.join(data_dir, "y_test.npy")

    x_num = np.load(x_num_path, allow_pickle=True) if os.path.exists(x_num_path) else None
    x_cat = np.load(x_cat_path, allow_pickle=True) if os.path.exists(x_cat_path) else None
    y = np.load(y_path, allow_pickle=True) if os.path.exists(y_path) else None

    return x_num, x_cat, y


def build_key(cat_row, y_row):
    items = []
    if cat_row is not None:
        if np.ndim(cat_row) == 0:
            items.append(cat_row.item())
        else:
            items.extend(np.atleast_1d(cat_row).tolist())
    if y_row is not None:
        if np.ndim(y_row) == 0:
            items.append(y_row.item())
        else:
            items.extend(np.atleast_1d(y_row).tolist())
    return tuple(map(str, items))


def build_gen_map(syn_num, syn_cat, syn_target):
    gen_map = {}
    for i in range(syn_num.shape[0]):
        key = build_key(
            syn_cat[i] if syn_cat is not None else None,
            syn_target[i] if syn_target is not None else None,
        )
        gen_map.setdefault(key, []).append(syn_num[i])
    return gen_map


def build_cat_map(syn_num, syn_cat):
    cat_map = {}
    if syn_cat is None:
        return cat_map
    for i in range(syn_num.shape[0]):
        key = build_key(syn_cat[i], None)
        cat_map.setdefault(key, []).append(syn_num[i])
    return cat_map


def build_y_map(syn_num, syn_target):
    y_map = {}
    if syn_target is None:
        return y_map
    for i in range(syn_num.shape[0]):
        key = build_key(None, syn_target[i])
        y_map.setdefault(key, []).append(syn_num[i])
    return y_map


def load_raw_config(config_path, ckpt_path=None):
    raw_config = src.load_config(config_path)
    if ckpt_path is not None:
        cached_config_path = os.path.join(os.path.dirname(ckpt_path), "config.pkl")
        if os.path.exists(cached_config_path):
            with open(cached_config_path, "rb") as f:
                raw_config = pickle.load(f)
                print(f"Found cached config at {cached_config_path}")
    return raw_config


def main():
    parser = argparse.ArgumentParser(description="Evaluate EF-VFM with matched cat+y and num-only metrics")
    parser.add_argument("--dataname", type=str, default="atn")
    parser.add_argument("--exp_name", type=str, default=None, help="If set, only evaluate this experiment")
    parser.add_argument("--num_samples", type=int, default=1000, help="Total samples to generate for matching")
    parser.add_argument("--sample_multiplier", type=int, default=10, help="num_samples = multiplier * test_size if num_samples is not set")
    parser.add_argument("--sample_batch_size", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max_rounds", type=int, default=5, help="Max sampling rounds for matching")

    args = parser.parse_args()

    if args.gpu != -1 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    data_dir = os.path.join(ROOT_DIR, "data", args.dataname)
    info_path = os.path.join(data_dir, "info.json")
    with open(info_path, "r") as f:
        info = json.load(f)

    ckpt_root = os.path.join(ROOT_DIR, "ef_vfm", "ckpt", args.dataname)
    if not os.path.exists(ckpt_root):
        raise FileNotFoundError(f"No ckpt root found under {ckpt_root}")
    if args.exp_name is not None:
        exp_names = [args.exp_name]
    else:
        exp_names = [
            name for name in os.listdir(ckpt_root)
            if os.path.isdir(os.path.join(ckpt_root, name))
        ]
        exp_names = sorted(exp_names)

    config_path = os.path.join(ROOT_DIR, "ef_vfm", "configs", "ef_vfm_configs.toml")
    raw_config = load_raw_config(config_path)

    dequant_dist = raw_config["data"]["dequant_dist"]
    int_dequant_factor = raw_config["data"]["int_dequant_factor"]

    dataset = EFVFMDataset(
        args.dataname,
        data_dir,
        info,
        isTrain=True,
        dequant_dist=dequant_dist,
        int_dequant_factor=int_dequant_factor,
    )
    d_numerical = dataset.d_numerical
    categories = dataset.categories

    raw_config["unimodmlp_params"]["d_numerical"] = d_numerical
    raw_config["unimodmlp_params"]["categories"] = categories.tolist()


    x_num_test, x_cat_test, y_test = load_test_data(data_dir)
    if x_num_test is None:
        raise ValueError("X_num_test.npy not found; cannot evaluate num-only metrics.")
    if y_test is None:
        raise ValueError("y_test.npy not found; cannot build cat+y matching keys.")

    test_size = y_test.shape[0]
    num_samples = args.num_samples or (test_size * args.sample_multiplier)
    sample_batch_size = args.sample_batch_size or raw_config["sample"]["batch_size"]

    print(f"Test size: {test_size}")

    all_metrics = []

    for exp_name in exp_names:
        ckpt_parent = os.path.join(ckpt_root, exp_name)
        ckpt_candidates = [
            x for x in os.listdir(ckpt_parent)
            if x.startswith("best_ema_model")
        ] if os.path.exists(ckpt_parent) else []
        if not ckpt_candidates:
            print(f"Skipping {exp_name}: no best_ema_model checkpoint found")
            continue
        ckpt_candidates = sorted(ckpt_candidates)
        ckpt_path = os.path.join(ckpt_parent, ckpt_candidates[0])
        print(ckpt_path)

        raw_config = load_raw_config(config_path, ckpt_path)
        raw_config["unimodmlp_params"]["d_numerical"] = d_numerical
        raw_config["unimodmlp_params"]["categories"] = categories.tolist()

        model = UniModMLP(**raw_config["unimodmlp_params"]).to(device)
        flow_model = ExpVFM(
            num_classes=categories,
            num_numerical_features=d_numerical,
            vf_fn=model,
            device=device,
        ).to(device)

        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and "vf_fn" in state:
            flow_model._vf_fn.load_state_dict(state["vf_fn"], strict=True)
        else:
            flow_model._vf_fn.load_state_dict(state, strict=True)

        start = time.time()

        flow_model.eval()

        matched_gen = [None] * test_size
        remaining_indices = list(range(test_size))
        last_syn_num = None
        last_syn_cat = None
        last_syn_target = None

        for round_idx in range(args.max_rounds):
            print(f"[{exp_name}] Generating {num_samples} samples for matching (round {round_idx + 1}/{args.max_rounds})")
            syn_data = flow_model.sample_all(num_samples, sample_batch_size, keep_nan_samples=True)

            syn_num, syn_cat, syn_target = split_num_cat_target(
                syn_data,
                info,
                dataset.num_inverse,
                dataset.int_inverse,
                dataset.cat_inverse,
            )

            last_syn_num = syn_num
            last_syn_cat = syn_cat
            last_syn_target = syn_target

            gen_map = build_gen_map(syn_num, syn_cat, syn_target)

            next_remaining = []
            matched_this_round = 0
            for i in remaining_indices:
                key = build_key(
                    x_cat_test[i] if x_cat_test is not None else None,
                    y_test[i] if y_test is not None else None,
                )
                bucket = gen_map.get(key)
                if bucket:
                    matched_gen[i] = bucket.pop()
                    matched_this_round += 1
                else:
                    next_remaining.append(i)

            remaining_indices = next_remaining
            matched_total = test_size - len(remaining_indices)
            print(f"[{exp_name}] Exact matched this round: {matched_this_round}, total: {matched_total}/{test_size}")
            if not remaining_indices:
                break

        if last_syn_num is None:
            print(f"Skipping {exp_name}: no samples generated")
            continue

        if remaining_indices:
            print(f"[{exp_name}] Unmatched after max rounds: {len(remaining_indices)}. Applying relaxed matching on last round samples.")
            cat_map = build_cat_map(last_syn_num, last_syn_cat)
            y_map = build_y_map(last_syn_num, last_syn_target)

            still_remaining = []
            for i in remaining_indices:
                y_key = build_key(None, y_test[i] if y_test is not None else None)
                bucket = y_map.get(y_key)
                if bucket:
                    matched_gen[i] = bucket.pop()
                else:
                    still_remaining.append(i)
            remaining_indices = still_remaining

            still_remaining = []
            for i in remaining_indices:
                cat_key = build_key(x_cat_test[i] if x_cat_test is not None else None, None)
                bucket = cat_map.get(cat_key)
                if bucket:
                    matched_gen[i] = bucket.pop()
                else:
                    still_remaining.append(i)
            remaining_indices = still_remaining

            if remaining_indices:
                print(f"[{exp_name}] Fallback still unmatched: {len(remaining_indices)}. Filling with random samples from last round.")
                for i in remaining_indices:
                    matched_gen[i] = last_syn_num[np.random.randint(0, last_syn_num.shape[0])]

        matched_count = sum(x is not None for x in matched_gen)
        print(f"[{exp_name}] Final matched {matched_count}/{test_size} samples")
        end = time.time()
        print(f"[{exp_name}] Elapsed time: {end - start} seconds")

        matched_gen = np.stack(matched_gen)
        save_path = os.path.join(ckpt_parent, "X_num_test.npy")
        np.save(save_path, matched_gen)
        print(f"[{exp_name}] Saved X_num_test.npy to {save_path}")

        gen_tensor = torch.tensor(matched_gen, dtype=torch.float32)
        test_tensor = torch.tensor(x_num_test, dtype=torch.float32)

        metrics = compute_metrics(gen_tensor, test_tensor)
        if metrics is None:
            continue

        all_metrics.append(metrics)
        result_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"[{exp_name}] Results -> {result_str}")

    if not all_metrics:
        print("No valid metrics collected.")
        return

    metric_keys = all_metrics[0].keys()
    mean_metrics = {}
    std_metrics = {}
    for k in metric_keys:
        values = np.array([m[k] for m in all_metrics], dtype=np.float64)
        mean_metrics[k] = np.mean(values)
        std_metrics[k] = np.std(values)

    summary_str = ", ".join([f"{k}: {mean_metrics[k]:.3f}±{std_metrics[k]:.3f}" for k in metric_keys])
    print(f"Summary (mean±std over {len(all_metrics)} exps) -> {summary_str}")


if __name__ == "__main__":
    main()
