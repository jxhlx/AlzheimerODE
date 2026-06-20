from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

import torch
from torch.utils.data import Dataset


REGION_DIM = 148
ATN_DIM = 3 * REGION_DIM


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


@dataclass
class ATNStats:
    amyloid_min: torch.Tensor
    amyloid_max: torch.Tensor
    tau_min: torch.Tensor
    tau_max: torch.Tensor
    ctx_min: torch.Tensor
    ctx_max: torch.Tensor

    def to(self, device: torch.device | str) -> "ATNStats":
        return ATNStats(
            amyloid_min=self.amyloid_min.to(device),
            amyloid_max=self.amyloid_max.to(device),
            tau_min=self.tau_min.to(device),
            tau_max=self.tau_max.to(device),
            ctx_min=self.ctx_min.to(device),
            ctx_max=self.ctx_max.to(device),
        )

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        amyloid, tau, ctx = split_atn(data)
        amyloid = (amyloid - self.amyloid_min) / (self.amyloid_max - self.amyloid_min + 1e-8)
        tau = (tau - self.tau_min) / (self.tau_max - self.tau_min + 1e-8)
        ctx = 1.0 - (ctx - self.ctx_min) / (self.ctx_max - self.ctx_min + 1e-8)
        return torch.cat([amyloid, tau, ctx], dim=-1)

    def unnormalize(self, data: torch.Tensor) -> torch.Tensor:
        amyloid, tau, ctx = split_atn(data)
        amyloid = amyloid * (self.amyloid_max - self.amyloid_min) + self.amyloid_min
        tau = tau * (self.tau_max - self.tau_min) + self.tau_min
        ctx = (1.0 - ctx) * (self.ctx_max - self.ctx_min) + self.ctx_min
        return torch.cat([amyloid, tau, ctx], dim=-1)


def split_atn(data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return data[..., :REGION_DIM], data[..., REGION_DIM:2 * REGION_DIM], data[..., 2 * REGION_DIM:]


class ATNDataset(Dataset):
    """读取 ATN 数据"""

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        device: str | torch.device = "cpu",
        normalize: bool = True,
        min_visits: int = 4,
        stats: ATNStats | None = None,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        self.data_dir = Path(data_dir)
        self.split = split
        self.device = torch.device(device)
        self.normalize = normalize
        self.min_visits = min_visits

        split_data_dir = self.data_dir / f"{split}_data"
        self.subject_ids = [name[:-3] for name in os.listdir(split_data_dir) if name.endswith(".pt")]
        if not self.subject_ids:
            raise FileNotFoundError(f"No .pt files found under {split_data_dir}")

        self.raw_data: list[torch.Tensor] = []
        self.raw_age: list[torch.Tensor] = []
        self.raw_label: list[torch.Tensor] = []
        for subject_id in self.subject_ids:
            self.raw_data.append(_load_tensor(self.data_dir / f"{split}_data" / f"{subject_id}.pt").float())
            self.raw_age.append(_load_tensor(self.data_dir / f"{split}_age" / f"{subject_id}.pt").float())
            self.raw_label.append(_load_tensor(self.data_dir / f"{split}_label" / f"{subject_id}.pt").long())

        self.max_visits = max(self.min_visits, max(item.shape[0] for item in self.raw_data))
        self.stats = (stats if stats is not None else self._compute_stats()).to(self.device)

    def _compute_stats(self) -> ATNStats:
        stacked = torch.cat(self.raw_data, dim=0)
        amyloid, tau, ctx = split_atn(stacked)
        return ATNStats(
            amyloid_min=amyloid.min(dim=0).values,
            amyloid_max=amyloid.max(dim=0).values,
            tau_min=tau.min(dim=0).values,
            tau_max=tau.max(dim=0).values,
            ctx_min=ctx.min(dim=0).values,
            ctx_max=ctx.max(dim=0).values,
        )

    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, index: int):
        data = self.raw_data[index]
        age = self.raw_age[index]
        label = self.raw_label[index]
        subject_id = self.subject_ids[index]

        data = data.to(self.device)
        age = age.to(self.device)
        label = label.to(self.device)
        if self.normalize:
            data = self.stats.normalize(data)

        padding = self.max_visits - data.shape[0]
        if padding > 0:
            data = torch.cat([data, torch.zeros(padding, ATN_DIM, device=self.device)], dim=0)
            age = torch.cat([age, torch.zeros(padding, device=self.device)], dim=0)
            label = torch.cat([label, label[-1].repeat(padding)], dim=0)

        amyloid, tau, ctx = split_atn(data)
        return amyloid, tau, ctx, age, label, subject_id

    def unnormalize_atn(self, data: torch.Tensor) -> torch.Tensor:
        return self.stats.unnormalize(data)


def build_subject_groups(dataset: ATNDataset):
    unlabeled = []
    labeled = []
    for item in dataset:
        age = item[3]
        if age.shape[0] < 2 or age[1].item() == 0.0:
            unlabeled.append(item)
        else:
            labeled.append(item)
    return unlabeled, labeled


def extract_survival_training_arrays(dataset: ATNDataset):
    features, times, events, labels = [], [], [], []
    for amyloid, tau, ctx, age, label, _ in dataset:
        valid = torch.where(age > 0)[0]
        for idx in valid:
            label_value = int(label[idx].item())
            if label_value in (0, 1):
                event = [0, 0]
            elif label_value in (2, 3):
                event = [1, 0]
            else:
                event = [1, 1]
            features.append(torch.cat([amyloid[idx], tau[idx], ctx[idx]], dim=-1).detach().cpu())
            times.append(float(age[idx].item()))
            events.append(event)
            labels.append(label_value)

    return (
        torch.stack(features).numpy(),
        torch.tensor(times, dtype=torch.float32).numpy(),
        torch.tensor(events, dtype=torch.float32).numpy(),
        torch.tensor(labels, dtype=torch.long).numpy(),
    )
