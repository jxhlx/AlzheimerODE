import os
import yaml
import numpy as np
from collections import defaultdict


def collect_metrics(root_dir):
    """
    Traverse subfolders under root_dir and collect metrics from results.yaml
    """
    metrics_dict = defaultdict(list)

    for subdir in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        yaml_path = os.path.join(subdir_path, "results.yaml")
        if not os.path.isfile(yaml_path):
            continue

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        test_metrics = data.get("metrics", {}).get("test", {})
        for k, v in test_metrics.items():
            if k == "TIME_SEC":
                continue
            metrics_dict[k].append(float(v))

    return metrics_dict


def compute_mean_std(metrics_dict):
    """
    Compute mean and std (3 decimal places) for each metric
    """
    stats = {}
    for k, values in metrics_dict.items():
        values = np.array(values)
        stats[k] = {
            "mean": round(np.mean(values), 3),
            "std": round(np.std(values), 3)
        }
    return stats


def main(root_dir):
    metrics_dict = collect_metrics(root_dir)
    stats = compute_mean_std(metrics_dict)

    print("=== Metrics (mean ± std) ===")
    for k in sorted(stats.keys()):
        m = stats[k]["mean"]
        s = stats[k]["std"]
        print(f"{k:10s}: {m:.3f} ± {s:.3f}")


if __name__ == "__main__":
    root_dir = "logs"
    main(root_dir)
