import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wasserstein_distance_nd


class ATNEvaluator:
    def metrics(self, all_samples, test_x):
        if (all_samples.min() < -1e10) or (all_samples.max() > 1e10):
            return None

        sample_array = all_samples.detach().cpu().numpy()
        real_array = test_x.detach().cpu().numpy()
        wd = max(0.0, float(wasserstein_distance_nd(sample_array, real_array)))

        sample_prob = F.softmax(all_samples, dim=1)
        real_prob = F.softmax(test_x, dim=1)
        mixture = 0.5 * (sample_prob + real_prob)

        jsd = 0.5 * (self.kld(sample_prob, mixture) + self.kld(real_prob, mixture))
        rmse = torch.sqrt(F.mse_loss(all_samples, test_x)).item()

        value_range = test_x.max() - test_x.min()
        nrmse = rmse / value_range.item() if value_range > 1e-8 else float("inf")

        return {
            "NRMSE": nrmse,
            "RMSE": rmse,
            "JSD": jsd.mean().item(),
            "WD": wd,
            "CosSim": self.cosine_similarity_metric(all_samples, test_x),
        }

    def kld(self, p, q):
        p = p + 1e-10
        q = q + 1e-10
        return (p * (p.log() - q.log())).sum(dim=1)

    def trajectory_metrics(self, generated_trajectories, real_trajectories):
        if len(generated_trajectories) != len(real_trajectories):
            return {"ADE": 0.0}

        ade_values = []
        for generated, real in zip(generated_trajectories, real_trajectories):
            generated = generated.cpu()
            real = real.cpu()

            if generated.shape != real.shape:
                n_steps = min(generated.shape[0], real.shape[0])
                generated = generated[:n_steps]
                real = real[:n_steps]

            if generated.shape[0] == 0:
                continue

            displacement = torch.norm(generated - real, p=2, dim=1)
            ade_values.append(displacement.mean().item())

        return {"ADE": float(np.mean(ade_values)) if ade_values else 0.0}

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
            denom = real_block.max(dim=0).values - min_vals
            denom = torch.where(denom > 1e-8, denom, torch.ones_like(denom))

            sample_norm = (sample_block - min_vals) / denom
            real_norm = (real_block - min_vals) / denom

            if idx == 2:
                sample_norm = 1.0 - sample_norm
                real_norm = 1.0 - real_norm

            cos_values.append(F.cosine_similarity(sample_norm, real_norm, dim=1).mean().item())

        return float(np.mean(cos_values)) if cos_values else 0.0
