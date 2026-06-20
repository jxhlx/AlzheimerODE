import argparse
import os
import re
from collections import defaultdict
import time
from pathlib import Path
import numpy as np
import torch

from atn_dataset import ATNTrainDataset, build_atn_samples
from train_atn_flow import build_model, evaluate_model, filter_train_samples

ROOT_DIR = Path(__file__).resolve().parents[1]


def find_last_checkpoint(ckpt_dir: str):
    if not os.path.isdir(ckpt_dir):
        return None
    candidates = []
    for fname in os.listdir(ckpt_dir):
        if not fname.endswith(".pt"):
            continue
        match = re.search(r"model_epoch_(\d+)\.pt", fname)
        if match:
            candidates.append((int(match.group(1)), os.path.join(ckpt_dir, fname)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    print(f"Found checkpoint: {candidates[-1][1]}")
    return candidates[-1][1]


def collect_experiment_dirs(ckpt_root: str):
    if not os.path.isdir(ckpt_root):
        return []
    subdirs = [os.path.join(ckpt_root, d) for d in sorted(os.listdir(ckpt_root))]
    subdirs = [d for d in subdirs if os.path.isdir(d)]
    if subdirs:
        return subdirs
    return [ckpt_root]


def summarize_metrics(metric_list):
    if not metric_list:
        return {}
    keys = metric_list[0].keys()
    summary = {}
    for key in keys:
        values = [m[key] for m in metric_list if m is not None and key in m]
        if not values:
            continue
        summary[key] = {
            "mean": float(np.mean(values).round(3)),
            "std": float(np.std(values).round(3)),
        }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-root", type=str, default=str(ROOT_DIR / "checkpoints"))
    parser.add_argument("--train-path", type=str, default=str(ROOT_DIR / "data" / "ALL" / "train"))
    parser.add_argument("--test-path", type=str, default=str(ROOT_DIR / "data" / "ALL" / "test"))
    parser.add_argument("--json-file", type=str, default=str(ROOT_DIR / "data" / "TABLE_Destrieux.json"))
    parser.add_argument("--prediction-length", type=int, default=6)
    parser.add_argument("--ndim-z", type=int, default=4)
    parser.add_argument("--ndim-tot", type=int, default=0)
    parser.add_argument("--label-onehot", action="store_true", default=True)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_samples, test_samples = build_atn_samples(
        train_path=args.train_path,
        test_path=args.test_path,
        json_file=args.json_file,
        prediction_length=args.prediction_length,
    )

    train_samples = filter_train_samples(train_samples, min_timepoints=2)

    if len(test_samples) == 0:
        raise RuntimeError("No test samples found.")

    ndim_x = test_samples[0].target.shape[0]
    if args.label_onehot:
        ndim_y = 1 + args.num_classes
    else:
        ndim_y = 2
    ndim_z = args.ndim_z
    ndim_tot = args.ndim_tot if args.ndim_tot > 0 else (ndim_x + ndim_y + ndim_z)

    batch_size = args.batch_size if args.batch_size > 0 else len(test_samples)
    test_loader = torch.utils.data.DataLoader(
        ATNTrainDataset(test_samples),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    metric_list = []
    traj_metric_list = []

    exp_dirs = collect_experiment_dirs(args.ckpt_root)
    start = time.time()
    for exp_dir in exp_dirs:
        
        ckpt_path = find_last_checkpoint(exp_dir)
        if ckpt_path is None:
            continue

        model = build_model(ndim_tot, ndim_z).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)

        eval_results = evaluate_model(
            model,
            test_loader,
            device,
            ndim_x,
            ndim_y,
            ndim_z,
            ndim_tot,
            args.num_classes,
            args.label_onehot,
        )

        if eval_results is None:
            continue

        metric_list.append(eval_results["metrics"])
        traj_metric_list.append(eval_results["trajectory_metrics"])
        
    end = time.time()
    print(f"Total evaluation time: {end - start:.2f} seconds")

    metrics_summary = summarize_metrics(metric_list)
    traj_summary = summarize_metrics(traj_metric_list)

    print("Metrics mean/std:", metrics_summary)
    print("Trajectory metrics mean/std:", traj_summary)


if __name__ == "__main__":
    main()
