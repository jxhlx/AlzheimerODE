from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ATNSystemParameters:
    k_pa: float = 0.0086
    k_da: float = 0.0247
    k_na: float = 6.0647
    k_mn: float = 5.6317
    k_pt: float = 0.0020
    k_dt: float = 0.0108
    k_at: float = 0.0119
    k_ma: float = 0.5068
    k_r: float = 0.0092
    k_tn: float = 0.0668
    k_mt: float = 9.0314
    alpha: float = 4.4260
    beta: float = 9.0782
    gamma: float = 0.4271


class ATNSystem:
    def __init__(self, device: torch.device | str, parameters: ATNSystemParameters | None = None) -> None:
        self.device = torch.device(device)
        self.params = parameters or ATNSystemParameters()
        for name, value in self.params.__dict__.items():
            setattr(self, name, torch.tensor(value, dtype=torch.float32, device=self.device))

    def hill(self, channel: int, value: torch.Tensor) -> torch.Tensor:
        value = torch.clamp(value, min=1e-6)
        if channel == 0:
            return self.k_na * value.pow(self.alpha) / (self.k_mn.pow(self.alpha) + value.pow(self.alpha) + 1e-6)
        if channel == 1:
            return self.k_at * value.pow(self.beta) / (self.k_ma.pow(self.beta) + value.pow(self.beta) + 1e-6)
        if channel == 2:
            return self.k_tn * value.pow(self.gamma) / (self.k_mt.pow(self.gamma) + value.pow(self.gamma) + 1e-6)
        raise ValueError(f"unknown ATN channel {channel}")

    def parameter_calculate(self, amyloid: torch.Tensor, tau: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        d_amyloid = self.k_pa - self.k_da * amyloid + self.hill(0, ctx)
        d_tau = self.k_pt - self.k_dt * tau + self.hill(1, amyloid)
        d_ctx = -(self.k_r * ctx) + self.hill(2, tau)
        return torch.cat([d_amyloid, d_tau, d_ctx], dim=-1)
