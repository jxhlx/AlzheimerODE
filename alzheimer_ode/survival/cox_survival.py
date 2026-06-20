from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import UnivariateSpline
from sklearn.utils import check_random_state
from sksurv.linear_model.coxph import BreslowEstimator
from tqdm import tqdm


def _to_numpy(array):
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _load_checkpoint(path: str | Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_representation_network(input_dim: int, layers: list[int] | None, activation: str = "ReLU6", bias: bool = False) -> nn.Sequential:
    layers = layers or []
    if activation == "ReLU6":
        act = nn.ReLU6()
    elif activation == "ReLU":
        act = nn.ReLU()
    elif activation == "SeLU":
        act = nn.SELU()
    elif activation == "Tanh":
        act = nn.Tanh()
    else:
        raise ValueError(f"unsupported activation {activation!r}")

    modules: list[nn.Module] = []
    previous = input_dim
    for hidden in layers:
        modules.append(nn.Linear(previous, hidden, bias=bias))
        modules.append(act)
        previous = hidden
    return nn.Sequential(*modules)


class CoxSurvivalNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        survival_components: int,
        gamma: float = 1.0,
        use_activation: bool = False,
        layers: list[int] | None = None,
        device: str = "cpu",
        region_specific: bool = False,
        region: int = 148,
    ) -> None:
        super().__init__()
        self.survival_components = int(survival_components)
        self.gamma = gamma
        self.use_activation = use_activation
        self.region_specific = region_specific
        self.region = region
        self.device = torch.device(device)

        last_dim = input_dim if not layers else layers[-1]
        self.gate = nn.Linear(last_dim, self.survival_components, bias=False)
        self.expert = nn.Linear(
            last_dim,
            self.survival_components * region if region_specific else self.survival_components,
            bias=False,
        )
        self.embedding = build_representation_network(input_dim, layers, "ReLU6")
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.float()
        model_device = next(self.parameters()).device
        self.device = model_device
        if x.device != model_device:
            x = x.to(model_device)
        hidden = self.embedding(x)
        gate_logits = torch.log_softmax(self.gate(hidden), dim=1)
        if self.use_activation:
            expert_logits = self.gamma * torch.tanh(self.expert(hidden))
        else:
            expert_logits = torch.clamp(self.expert(hidden), min=-self.gamma, max=self.gamma)
        if self.region_specific:
            expert_logits = expert_logits.view(hidden.shape[0], self.survival_components, self.region)
        return gate_logits, expert_logits


class TorchBaselineSpline(nn.Module):
    def __init__(self, x, y, ext: int = 3, device: str = "cpu") -> None:
        super().__init__()
        self.device = torch.device(device)
        self.register_buffer("x", torch.as_tensor(x, dtype=torch.float32, device=self.device).flatten())
        self.register_buffer("y", torch.as_tensor(y, dtype=torch.float32, device=self.device).flatten())
        order = torch.argsort(self.x)
        self.x = self.x[order]
        self.y = self.y[order]
        self.ext = ext

    def forward(self, query: torch.Tensor | float) -> torch.Tensor:
        query = torch.as_tensor(query, dtype=torch.float32, device=self.device)
        is_scalar = query.ndim == 0
        if is_scalar:
            query = query.unsqueeze(0)
        flat = query.reshape(-1)

        if self.x.numel() == 1:
            values = self.y.expand_as(flat)
        else:
            idx = torch.searchsorted(self.x, flat, right=True) - 1
            idx = idx.clamp(0, self.x.numel() - 2)
            x0 = self.x[idx]
            x1 = self.x[idx + 1]
            y0 = self.y[idx]
            y1 = self.y[idx + 1]
            slope = (y1 - y0) / (x1 - x0 + 1e-8)
            values = y0 + slope * (flat - x0)

            if self.ext == 1:
                outside = (flat < self.x[0]) | (flat > self.x[-1])
                values = torch.where(outside, torch.zeros_like(values), values)
            elif self.ext == 3:
                values = torch.where(flat < self.x[0], self.y[0], values)
                values = torch.where(flat > self.x[-1], self.y[-1], values)

        values = values.reshape(query.shape)
        return values.squeeze(0) if is_scalar else values

    def derivative(self, query: torch.Tensor | float) -> torch.Tensor:
        query = torch.as_tensor(query, dtype=torch.float32, device=self.device)
        is_scalar = query.ndim == 0
        if is_scalar:
            query = query.unsqueeze(0)
        flat = query.reshape(-1)

        if self.x.numel() == 1:
            slopes = torch.zeros_like(flat)
        else:
            idx = torch.searchsorted(self.x, flat, right=True) - 1
            idx = idx.clamp(0, self.x.numel() - 2)
            slopes = (self.y[idx + 1] - self.y[idx]) / (self.x[idx + 1] - self.x[idx] + 1e-8)
            if self.ext in {1, 3}:
                slopes = torch.where((flat < self.x[0]) | (flat > self.x[-1]), torch.zeros_like(slopes), slopes)

        slopes = slopes.reshape(query.shape)
        return slopes.squeeze(0) if is_scalar else slopes

    def state_dict_payload(self) -> dict[str, torch.Tensor | int]:
        return {"x": self.x.detach().cpu(), "y": self.y.detach().cpu(), "ext": self.ext}

    @classmethod
    def from_state_dict_payload(cls, payload: dict, device: str = "cpu") -> "TorchBaselineSpline":
        return cls(payload["x"], payload["y"], ext=int(payload.get("ext", 3)), device=device)


def sample_hard_z(log_posteriors: torch.Tensor) -> torch.Tensor:
    return torch.multinomial(log_posteriors.exp(), num_samples=1)[:, 0]


def hard_z(log_posteriors: torch.Tensor) -> torch.Tensor:
    return torch.argmax(log_posteriors, dim=1)


def _partial_log_likelihood(log_risks: torch.Tensor, times: torch.Tensor, events: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    noisy_times = times.detach().cpu().numpy() + eps * np.random.random(len(times))
    order = torch.as_tensor(np.argsort(-noisy_times), device=times.device, dtype=torch.long)
    log_risks = log_risks[order]
    events = events[order]
    log_denom = torch.logcumsumexp(log_risks, dim=0)
    pll = log_risks - log_denom
    return -pll[events == 1].sum()


def _fit_spline_from_breslow(breslow: BreslowEstimator, smoothing_factor: float, device: str) -> TorchBaselineSpline:
    survival = breslow.baseline_survival_
    spline = UnivariateSpline(survival.x, survival.y, s=smoothing_factor, ext=3, k=1)
    knots = spline.get_knots()
    values = spline(knots)
    return TorchBaselineSpline(knots, values, ext=3, device=device)


def _fit_breslow_splines(
    model: CoxSurvivalNetwork,
    x: torch.Tensor,
    t: torch.Tensor,
    e: torch.Tensor,
    posteriors: torch.Tensor | None = None,
    smoothing_factor: float = 1e-4,
    device: str = "cpu",
) -> dict[int, TorchBaselineSpline]:
    gates, risks = model(x)
    x_np = _to_numpy(x)
    t_np = _to_numpy(t)
    e_np = _to_numpy(e)
    if posteriors is None:
        assignments = sample_hard_z(gates)
    else:
        assignments = sample_hard_z(posteriors)
    if risks.ndim == 3:
        risks_np = _to_numpy(risks)
    else:
        risks_np = _to_numpy(risks)

    splines: dict[int, TorchBaselineSpline] = {}
    for cluster in range(model.survival_components):
        indices = (assignments == cluster).nonzero(as_tuple=False).flatten()
        if indices.numel() == 0:
            splines[cluster] = TorchBaselineSpline([0.0, 1.0], [1.0, 1.0], device=device)
            continue

        if risks_np.ndim == 3:
            cluster_risk = risks_np[indices.cpu().numpy(), cluster, :].mean(axis=1)
        else:
            cluster_risk = risks_np[indices.cpu().numpy(), cluster]

        cluster_events = e_np[indices.cpu().numpy()]
        if cluster_events.ndim == 2:
            cluster_events = cluster_events[:, min(cluster, cluster_events.shape[1] - 1)]
        cluster_times = t_np[indices.cpu().numpy()]

        try:
            breslow = BreslowEstimator().fit(cluster_risk, cluster_events.astype(bool), cluster_times)
            splines[cluster] = _fit_spline_from_breslow(breslow, smoothing_factor=smoothing_factor, device=device)
        except Exception:
            if cluster_times.size == 0:
                splines[cluster] = TorchBaselineSpline([0.0, 1.0], [1.0, 1.0], device=device)
            else:
                sorted_times = np.unique(np.sort(cluster_times))
                if sorted_times.size == 1:
                    sorted_times = np.array([0.0, float(sorted_times[0]) + 1e-3], dtype=np.float32)
                    values = np.array([1.0, 1.0], dtype=np.float32)
                else:
                    values = np.linspace(1.0, 0.2, num=sorted_times.size, dtype=np.float32)
                splines[cluster] = TorchBaselineSpline(sorted_times, values, device=device)
    return splines


def _survival_probability(log_risks: torch.Tensor, spline: TorchBaselineSpline, time_value, region_specific: bool) -> torch.Tensor:
    if region_specific:
        base = spline(time_value)
        if not isinstance(base, torch.Tensor):
            base = torch.as_tensor(base, dtype=torch.float32, device=log_risks.device)
        if base.ndim == 0:
            base = base.view(1)
        if base.ndim == 1:
            base = base.unsqueeze(-1)
        return base.pow(torch.exp(log_risks))
    base = spline(time_value)
    if not isinstance(base, torch.Tensor):
        base = torch.as_tensor(base, dtype=torch.float32, device=log_risks.device)
    return base.pow(torch.exp(log_risks))


def _survival_density(log_risks: torch.Tensor, spline: TorchBaselineSpline, time_value, region_specific: bool) -> torch.Tensor:
    if region_specific:
        base = spline(time_value)
        derivative = spline.derivative(time_value)
        if base.ndim == 0:
            base = base.view(1)
            derivative = derivative.view(1)
        if base.ndim == 1:
            base = base.unsqueeze(-1)
            derivative = derivative.unsqueeze(-1)
        risk = torch.exp(log_risks)
        density = -risk * base.pow(risk - 1) * derivative
        return density.clamp_min(1e-12)
    base = spline(time_value)
    derivative = spline.derivative(time_value)
    risk = torch.exp(log_risks)
    return (-risk * base.pow(risk - 1) * derivative).clamp_min(1e-12)


def _event_likelihood(model: CoxSurvivalNetwork, splines: dict[int, TorchBaselineSpline], x: torch.Tensor, t: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    gates, log_risks = model(x)
    region_specific = log_risks.ndim == 3

    survival_terms = []
    density_terms = []
    for cluster in range(model.survival_components):
        cluster_risks = log_risks[:, cluster, :] if region_specific else log_risks[:, cluster]
        survival = _survival_probability(cluster_risks, splines[cluster], t, region_specific)
        density = _survival_density(cluster_risks, splines[cluster], t, region_specific)
        if region_specific:
            survival = survival.mean(dim=-1)
            density = density.mean(dim=-1)
        survival_terms.append(survival)
        density_terms.append(density)

    survival = torch.stack(survival_terms, dim=1)
    density = torch.stack(density_terms, dim=1)
    event_prob = torch.where(e.bool(), density, survival).clamp_min(1e-12)
    return gates + torch.log(event_prob)


def _q_loss(model: CoxSurvivalNetwork, x: torch.Tensor, t: torch.Tensor, e: torch.Tensor, posteriors: torch.Tensor, typ: str = "soft") -> torch.Tensor:
    if typ == "hard":
        assignments = hard_z(posteriors)
    else:
        assignments = sample_hard_z(posteriors)

    gates, log_risks = model(x)
    loss = torch.zeros((), device=x.device)
    for cluster in range(model.survival_components):
        cluster_mask = assignments == cluster
        if cluster_mask.any():
            cluster_risks = log_risks[cluster_mask, cluster]
            cluster_events = e[cluster_mask][:, min(cluster, e.shape[1] - 1)]
            loss = loss + _partial_log_likelihood(cluster_risks, t[cluster_mask], cluster_events)

    gate_loss = -(posteriors.exp() * gates).sum()
    loss = loss + gate_loss

    labels = []
    logits = []
    for idx in range(e.shape[0]):
        if torch.all(e[idx] == torch.tensor([0, 0], device=e.device)):
            labels.append(0)
            logits.append(gates[idx])
        elif torch.all(e[idx] == torch.tensor([1, 1], device=e.device)):
            labels.append(1)
            logits.append(gates[idx])
    if logits:
        loss = loss + 100.0 * nn.NLLLoss()(torch.stack(logits), torch.tensor(labels, device=x.device))

    for cluster in range(1, model.survival_components):
        loss = loss + torch.relu(torch.mean(torch.exp(log_risks[:, cluster - 1]) - torch.exp(log_risks[:, cluster])))
    return loss


def _e_step(model: CoxSurvivalNetwork, splines: dict[int, TorchBaselineSpline] | None, x: torch.Tensor, t: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    if splines is None:
        return torch.log_softmax(torch.rand((x.shape[0], model.survival_components), device=x.device), dim=1)
    scores = _event_likelihood(model, splines, x, t, e)
    scores = torch.nan_to_num(scores, nan=-10.0, neginf=-10.0, posinf=10.0).clamp_min(-10.0)
    return torch.log_softmax(scores, dim=1)


def _train_epoch(
    model: CoxSurvivalNetwork,
    x: torch.Tensor,
    t: torch.Tensor,
    e: torch.Tensor,
    splines: dict[int, TorchBaselineSpline] | None,
    optimizer: torch.optim.Optimizer,
    batch_size: int,
    seed: int | None,
    typ: str,
    use_posteriors: bool,
    update_splines_after: int,
    smoothing_factor: float,
    device: str,
) -> dict[int, TorchBaselineSpline]:
    x = torch.as_tensor(x, dtype=torch.float32, device=device)
    t = torch.as_tensor(t, dtype=torch.float32, device=device)
    e = torch.as_tensor(e, dtype=torch.float32, device=device)
    if seed is None:
        order_np = np.random.permutation(x.shape[0])
    else:
        order_np = check_random_state(seed).permutation(x.shape[0])
    order = torch.as_tensor(order_np, dtype=torch.long, device=x.device)
    x, t, e = x[order], t[order], e[order]
    for batch_number, batch_index in enumerate(range(0, x.shape[0], batch_size)):
        xb = x[batch_index : batch_index + batch_size]
        tb = t[batch_index : batch_index + batch_size]
        eb = e[batch_index : batch_index + batch_size]
        with torch.no_grad():
            posteriors = _e_step(model, splines, xb, tb, eb)
        optimizer.zero_grad()
        loss = _q_loss(model, xb, tb, eb, posteriors, typ=typ)
        loss.backward()
        optimizer.step()
        if batch_number % max(1, update_splines_after) == 0:
            with torch.no_grad():
                all_posteriors = _e_step(model, splines, x, t, e)
            try:
                posteriors_for_spline = all_posteriors if use_posteriors else None
                splines = _fit_breslow_splines(
                    model,
                    x,
                    t,
                    e,
                    posteriors=posteriors_for_spline,
                    smoothing_factor=smoothing_factor,
                    device=device,
                )
            except Exception:
                pass
    return splines or {}


def train_cox_survival(
    model: CoxSurvivalNetwork,
    train_data,
    val_data,
    epochs: int = 50,
    patience: int = 3,
    vloss: str = "q",
    batch_size: int = 256,
    typ: str = "soft",
    lr: float = 1e-3,
    use_posteriors: bool = True,
    random_seed: int | None = None,
    return_losses: bool = False,
    update_splines_after: int = 10,
    smoothing_factor: float = 1e-2,
    device: str = "cpu",
    progress: bool = True,
):
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    if val_data is None:
        val_data = train_data

    xt, tt, et = train_data
    xv, tv, ev = val_data
    xt = torch.as_tensor(xt, dtype=torch.float32, device=device)
    tt = torch.as_tensor(tt, dtype=torch.float32, device=device)
    et = torch.as_tensor(et, dtype=torch.float32, device=device)
    xv = torch.as_tensor(xv, dtype=torch.float32, device=device)
    tv = torch.as_tensor(tv, dtype=torch.float32, device=device)
    ev = torch.as_tensor(ev, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    previous_val = float("inf")
    bad_epochs = 0
    splines = None
    losses = []

    epoch_range = tqdm(range(epochs), desc="Cox pretrain", disable=not progress)
    for epoch in epoch_range:
        splines = _train_epoch(
            model=model,
            x=xt,
            t=tt,
            e=et,
            splines=splines,
            optimizer=optimizer,
            batch_size=batch_size,
            seed=epoch if random_seed is not None else None,
            typ=typ,
            use_posteriors=use_posteriors,
            update_splines_after=update_splines_after,
            smoothing_factor=smoothing_factor,
            device=device,
        )
        with torch.no_grad():
            val_posteriors = _e_step(model, splines, xv, tv, ev)
            if vloss == "q":
                val_loss = float(_q_loss(model, xv, tv, ev, val_posteriors, typ=typ).detach().cpu() / xv.shape[0])
            else:
                val_loss = float((-val_posteriors.mean()).detach().cpu())
        losses.append(val_loss)
        epoch_range.set_postfix(val_loss=f"{val_loss:.4f}")

        if val_loss > previous_val:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
        else:
            bad_epochs = 0
        previous_val = val_loss

    if return_losses:
        return (model, splines), losses
    return model, splines


@dataclass
class CoxSurvivalConfig:
    input_dim: int
    survival_components: int = 2
    layers: list[int] | None = None
    gamma: float = 10.0
    smoothing_factor: float = 1e-4
    use_activation: bool = False
    random_seed: int | None = None
    device: str = "cpu"
    region_specific: bool = True
    region: int = 148


class CoxSurvivalModel:
    def __init__(
        self,
        input_dim: int,
        survival_components: int = 2,
        layers: list[int] | None = None,
        gamma: float = 10.0,
        smoothing_factor: float = 1e-4,
        use_activation: bool = False,
        random_seed: int | None = None,
        device: str = "cpu",
        region_specific: bool = True,
        region: int = 148,
    ) -> None:
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
        self.config = CoxSurvivalConfig(
            input_dim=input_dim,
            survival_components=survival_components,
            layers=layers,
            gamma=gamma,
            smoothing_factor=smoothing_factor,
            use_activation=use_activation,
            random_seed=random_seed,
            device=device,
            region_specific=region_specific,
            region=region,
        )
        self.device = torch.device(self.config.device)
        self.model = CoxSurvivalNetwork(
            input_dim=self.config.input_dim,
            survival_components=self.config.survival_components,
            gamma=self.config.gamma,
            use_activation=self.config.use_activation,
            layers=self.config.layers,
            device=self.config.device,
            region_specific=self.config.region_specific,
            region=self.config.region,
        )
        self.splines: dict[int, TorchBaselineSpline] = {}
        self.fitted = False

    def parameters(self):
        return self.model.parameters()

    def freeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False

    def to(self, device: str | torch.device):
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.device = self.device
        for key, spline in self.splines.items():
            self.splines[key] = TorchBaselineSpline(spline.x, spline.y, ext=spline.ext, device=self.device)
        return self

    def fit(
        self,
        x,
        t,
        e,
        vsize: float = 0.15,
        val_data=None,
        epochs: int = 1,
        learning_rate: float = 1e-3,
        batch_size: int = 100,
        auto_convert_splines: bool = True,
        patience: int = 3,
        progress: bool = True,
    ):
        x = _to_numpy(x)
        t = _to_numpy(t)
        e = _to_numpy(e)
        indices = np.arange(len(x))
        if self.config.random_seed is None:
            np.random.shuffle(indices)
        else:
            rng = np.random.RandomState(self.config.random_seed)
            rng.shuffle(indices)
        x, t, e = x[indices], t[indices], e[indices]
        if val_data is None:
            val_count = int(vsize * len(x))
            train = (x[:-val_count], t[:-val_count], e[:-val_count]) if val_count > 0 else (x, t, e)
            valid = (x[-val_count:], t[-val_count:], e[-val_count:]) if val_count > 0 else (x, t, e)
        else:
            valid = tuple(_to_numpy(v) for v in val_data)
            train = (x, t, e)

        (model, splines), _ = train_cox_survival(
            self.model,
            train,
            valid,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            typ="soft",
            lr=learning_rate,
            use_posteriors=True,
            random_seed=self.config.random_seed,
            return_losses=True,
            update_splines_after=10,
            smoothing_factor=self.config.smoothing_factor,
            device=str(self.device),
            progress=progress,
        )
        self.model = model.eval()
        self.splines = splines or {}
        self.fitted = True
        if auto_convert_splines:
            self.convert_breslow_splines_to_torch(device=self.device)
        return self

    def convert_breslow_splines_to_torch(self, device: str | torch.device | None = None):
        if device is not None:
            self.device = torch.device(device)
        converted = {}
        for cluster, spline in self.splines.items():
            converted[cluster] = TorchBaselineSpline(spline.x, spline.y, ext=spline.ext, device=self.device)
        self.splines = converted
        return self

    def _check_ready(self):
        if not self.fitted:
            raise RuntimeError("CoxSurvivalModel is not fitted.")

    def predict_survival(self, x, t):
        self._check_ready()
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        gates, log_risks = self.model(x_tensor)
        gate_probs = gates.exp()
        if isinstance(t, (list, tuple)):
            times = list(t)
        else:
            times = [t]

        region_specific = log_risks.ndim == 3
        predictions = []
        for time_value in times:
            if isinstance(time_value, torch.Tensor) and time_value.ndim > 0 and time_value.numel() > 1:
                time_tensor = time_value.to(self.device).float()
            else:
                scalar = float(time_value.detach().cpu().item()) if isinstance(time_value, torch.Tensor) else float(time_value)
                time_tensor = torch.tensor(scalar, device=self.device)

            expert_outputs = []
            for cluster in range(self.config.survival_components):
                cluster_risks = log_risks[:, cluster, :] if region_specific else log_risks[:, cluster]
                spline = self.splines[cluster]
                base = spline(time_tensor)
                if not isinstance(base, torch.Tensor):
                    base = torch.as_tensor(base, device=self.device, dtype=torch.float32)
                if region_specific:
                    if base.ndim == 0:
                        base = base.view(1)
                    if base.ndim == 1 and cluster_risks.ndim == 2:
                        base = base.unsqueeze(-1)
                    expert_output = base.pow(torch.exp(cluster_risks))
                else:
                    expert_output = base.pow(torch.exp(cluster_risks))
                expert_outputs.append(expert_output)

            stacked = torch.stack(expert_outputs, dim=1)
            selected = stacked[torch.arange(stacked.shape[0], device=self.device), gate_probs.argmax(dim=1)]
            predictions.append(selected)

        if len(predictions) == 1:
            return predictions[0].unsqueeze(1)
        return torch.stack(predictions, dim=1)

    def predict_latent_z(self, x):
        self._check_ready()
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        gates, _ = self.model(x_tensor)
        return gates.exp()

    def state_dict(self) -> dict:
        return {
            "config": self.config.__dict__,
            "model": self.model.state_dict(),
            "splines": {str(key): spline.state_dict_payload() for key, spline in self.splines.items()},
            "fitted": self.fitted,
        }

    def load_state_dict(self, payload: dict):
        config = dict(payload["config"])
        config["device"] = str(self.device)
        self.config = CoxSurvivalConfig(**config)
        self.model = CoxSurvivalNetwork(
            input_dim=self.config.input_dim,
            survival_components=self.config.survival_components,
            gamma=self.config.gamma,
            use_activation=self.config.use_activation,
            layers=self.config.layers,
            device=str(self.device),
            region_specific=self.config.region_specific,
            region=self.config.region,
        )
        self.model.load_state_dict(payload["model"])
        self.splines = {int(key): TorchBaselineSpline.from_state_dict_payload(state, device=self.device) for key, state in payload["splines"].items()}
        self.fitted = bool(payload.get("fitted", True))
        self.to(self.device)
        return self

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device = "cpu") -> "CoxSurvivalModel":
        payload = _load_checkpoint(path)
        config = payload["config"]
        config["device"] = str(device)
        model = cls(**config)
        model.model.load_state_dict(payload["model"])
        model.splines = {int(key): TorchBaselineSpline.from_state_dict_payload(state, device=device) for key, state in payload["splines"].items()}
        model.fitted = bool(payload.get("fitted", True))
        model.to(device)
        return model
