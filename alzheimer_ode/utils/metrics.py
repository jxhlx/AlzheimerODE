from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from scipy.stats import wasserstein_distance_nd


EPS = 1e-8
PROB_EPS = 1e-10
EXPLOSION_THRESHOLD = 1e10


def _cpu(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().cpu()


def _check_pair(samples: torch.Tensor, real: torch.Tensor) -> None:
    if samples.shape != real.shape:
        raise ValueError(f"samples and real must have the same shape, got {samples.shape} and {real.shape}")
    if samples.ndim != 2:
        raise ValueError(f"metrics expect 2D tensors, got {samples.ndim}D")
    if torch.any(torch.abs(samples) > EXPLOSION_THRESHOLD).item():
        raise ValueError("sample values are exploded")


def _kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = p.clamp_min(PROB_EPS)
    q = q.clamp_min(PROB_EPS)
    return (p * (p.log() - q.log())).sum(dim=1)


def jsd(samples: torch.Tensor, real: torch.Tensor) -> float:
    sample_prob = F.softmax(samples, dim=1)
    real_prob = F.softmax(real, dim=1)
    mix = 0.5 * (sample_prob + real_prob)
    return float((0.5 * (_kl(sample_prob, mix) + _kl(real_prob, mix))).mean())


def rmse(samples: torch.Tensor, real: torch.Tensor) -> float:
    return float(torch.sqrt(F.mse_loss(samples, real)))


def root_mean_square_error(samples: torch.Tensor, real: torch.Tensor) -> float:
    return rmse(_cpu(samples), _cpu(real))


def nrmse(samples: torch.Tensor, real: torch.Tensor) -> float:
    value_range = float(real.max() - real.min())
    if value_range <= EPS:
        return float("inf")
    return rmse(samples, real) / value_range


def wd(samples: torch.Tensor, real: torch.Tensor) -> float:
    return max(0.0, float(wasserstein_distance_nd(samples.numpy(), real.numpy())))


def ade(sample_trajs: Sequence[torch.Tensor], real_trajs: Sequence[torch.Tensor]) -> float:
    if len(sample_trajs) != len(real_trajs):
        raise ValueError("sample and real trajectory counts must match")
    if not sample_trajs:
        return 0.0

    errors = []
    for samples, real in zip(sample_trajs, real_trajs):
        if samples.shape != real.shape:
            n = min(samples.shape[0], real.shape[0])
            samples = samples[:n]
            real = real[:n]
        if samples.numel() == 0:
            continue
        errors.append(float(torch.linalg.vector_norm(samples - real, ord=2, dim=1).mean()))

    if not errors:
        return 0.0
    return float(torch.tensor(errors, dtype=torch.float32).mean())


def _atn_blocks(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.shape[1] % 3 != 0:
        raise ValueError(f"ATN metrics expect feature dimension divisible by 3, got {x.shape[1]}")
    region_dim = x.shape[1] // 3
    return (
        x[:, :region_dim],
        x[:, region_dim:2 * region_dim],
        x[:, 2 * region_dim:],
    )


def _scale_like_real(samples: torch.Tensor, real: torch.Tensor, invert: bool) -> tuple[torch.Tensor, torch.Tensor]:
    lo = real.min(dim=0).values
    span = real.max(dim=0).values - lo
    span = torch.where(span > EPS, span, torch.ones_like(span))
    samples = (samples - lo) / span
    real = (real - lo) / span
    if invert:
        samples = 1.0 - samples
        real = 1.0 - real
    return samples, real


def cos_sim(samples: torch.Tensor, real: torch.Tensor) -> float:
    scores = []
    for idx, (sample_block, real_block) in enumerate(zip(_atn_blocks(samples), _atn_blocks(real))):
        sample_block, real_block = _scale_like_real(sample_block, real_block, invert=idx == 2)
        scores.append(float(F.cosine_similarity(sample_block, real_block, dim=1).mean()))
    return float(torch.tensor(scores, dtype=torch.float32).mean())


def distribution_metrics(
    samples: torch.Tensor,
    real: torch.Tensor,
    *,
    sample_trajs: Sequence[torch.Tensor] | None = None,
    real_trajs: Sequence[torch.Tensor] | None = None,
) -> dict[str, float]:
    _check_pair(samples, real)
    samples = _cpu(samples)
    real = _cpu(real)

    if sample_trajs is None or real_trajs is None:
        sample_trajs = (samples,)
        real_trajs = (real,)
    else:
        sample_trajs = tuple(_cpu(item) for item in sample_trajs)
        real_trajs = tuple(_cpu(item) for item in real_trajs)

    return {
        "NRMSE": nrmse(samples, real),
        "RMSE": rmse(samples, real),
        "JSD": jsd(samples, real),
        "WD": wd(samples, real),
        "ADE": ade(sample_trajs, real_trajs),
        "CosSim": cos_sim(samples, real),
    }
