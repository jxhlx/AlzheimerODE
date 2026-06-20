from __future__ import annotations

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum
from torch_geometric.nn import GCNConv
from torchdiffeq import odeint


REGION_DIM = 148
ATN_DIM = 3 * REGION_DIM
DEFAULT_HORIZON = 5.0
ODE_STEPS = 21


def exists(value):
    return value is not None


def default(value, fallback):
    if exists(value):
        return value
    return fallback() if callable(fallback) else fallback


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, theta: int = 10000) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        if x.size() == torch.Size([]):
            x = x.unsqueeze(0)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, is_random: bool = False) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("dim must be even")
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad=not is_random)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return torch.cat((x, fouriered), dim=-1)


class Residual(nn.Module):
    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.fn(x, *args, **kwargs) + x


class RMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm(x))


class Block(nn.Module):
    def __init__(self, dim: int, dim_out: int) -> None:
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, scale_shift=None) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim: int, dim_out: int, *, time_emb_dim: int | None = None) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if exists(time_emb_dim) else None
        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor | None = None) -> torch.Tensor:
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1")
            scale_shift = time_emb.chunk(2, dim=1)
        hidden = self.block1(x, scale_shift=scale_shift)
        hidden = self.block2(hidden)
        return hidden + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv1d(hidden_dim, dim, 1), RMSNorm(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) n -> b h c n", h=self.heads), qkv)
        q = q.softmax(dim=-2) * self.scale
        k = k.softmax(dim=-1)
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c n -> b (h c) n", h=self.heads)
        return self.to_out(out)


class Attention1D(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) n -> b h c n", h=self.heads), qkv)
        q = q * self.scale
        similarity = einsum("b h d i, b h d j -> b h i j", q, k)
        attn = similarity.softmax(dim=-1)
        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        return self.to_out(rearrange(out, "b h n d -> b (h d) n"))


class EmbedFC(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(nn.Linear(input_dim, emb_dim), nn.GELU(), nn.Linear(emb_dim, emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class EmbedAge(EmbedFC):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.unsqueeze(-1))


def upsample(idx: int, dim: int, dim_out: int | None = None) -> nn.Module:
    if idx == 0:
        return nn.Sequential(nn.Upsample(size=(37,), mode="nearest"), nn.Conv1d(dim, default(dim_out, dim), 3, padding=1))
    return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv1d(dim, default(dim_out, dim), 3, padding=1))


def downsample(dim: int, dim_out: int | None = None) -> nn.Module:
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)


class InitUNet1D(nn.Module):
    def __init__(
        self,
        dim: int,
        init_dim: int | None = None,
        out_dim: int | None = None,
        dim_mults: tuple[int, ...] = (1, 2, 4, 8),
        channels: int = 1,
        self_condition: bool = False,
        learned_variance: bool = False,
        learned_sinusoidal_cond: bool = False,
        random_fourier_features: bool = False,
        learned_sinusoidal_dim: int = 16,
        sinusoidal_pos_emb_theta: int = 10000,
        attn_dim_head: int = 32,
        attn_heads: int = 4,
        n_classes: int = 5,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding=3)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        time_dim = None

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
        else:
            SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out_inner) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                        ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        downsample(dim_in, dim_out_inner) if not is_last else nn.Conv1d(dim_in, dim_out_inner, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention1D(mid_dim, dim_head=attn_dim_head, heads=attn_heads)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        cond_in, cond_out = 32, 64
        self.age_embed1 = EmbedAge(1, cond_in)
        self.age_embed2 = EmbedAge(1, cond_in // 2)
        self.label_embed1 = EmbedFC(n_classes, cond_in)
        self.label_embed2 = EmbedFC(n_classes, cond_in // 2)
        self.cond_mix1 = EmbedFC(2 * cond_in, 2 * cond_out)
        self.cond_mix2 = EmbedFC(cond_in, cond_out)

        for ind, (dim_in, dim_out_inner) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_out_inner + dim_in + 2 * cond_out, dim_out_inner, time_emb_dim=time_dim),
                        ResnetBlock(dim_out_inner + dim_in + cond_out, dim_out_inner, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out_inner, LinearAttention(dim_out_inner))),
                        upsample(ind, dim_out_inner, dim_in) if not is_last else nn.Conv1d(dim_out_inner, dim_in, 3, padding=1),
                    ]
                )
            )

        default_out_dim = channels * (3 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        self.final_res_block = ResnetBlock(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

    def forward(self, x: torch.Tensor, label: torch.Tensor, age: torch.Tensor, x_self_cond: torch.Tensor | None = None) -> torch.Tensor:
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        age_emb1 = self.age_embed1(age)
        age_emb2 = self.age_embed2(age)
        label_emb1 = self.label_embed1(label.float())
        label_emb2 = self.label_embed2(label.float())
        cond1 = self.cond_mix1(torch.cat((age_emb1, label_emb1), dim=-1))
        cond2 = self.cond_mix2(torch.cat((age_emb2, label_emb2), dim=-1))

        x = self.init_conv(x)
        residual = x.clone()
        skip_connections = []
        for block1, block2, attn, down in self.downs:
            x = block1(x)
            skip_connections.append(x)
            x = block2(x)
            x = attn(x)
            skip_connections.append(x)
            x = down(x)

        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        for block1, block2, attn, up in self.ups:
            cond1_exp = cond1.unsqueeze(2).expand(-1, -1, x.shape[-1])
            x = torch.cat((x, cond1_exp, skip_connections.pop()), dim=1)
            x = block1(x)
            cond2_exp = cond2.unsqueeze(2).expand(-1, -1, x.shape[-1])
            x = torch.cat((x, cond2_exp, skip_connections.pop()), dim=1)
            x = block2(x)
            x = attn(x)
            x = up(x)

        x = torch.cat((x, residual), dim=1)
        x = self.final_res_block(x)
        return self.final_conv(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32) -> None:
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = heads * dim_head
        self.to_q = nn.Conv1d(dim, hidden_dim, 1, bias=False)
        self.to_kv = nn.Conv1d(dim, hidden_dim * 2, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)
        self.attention_scores = None

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        q = self.to_q(a)
        k, v = self.to_kv(b).chunk(2, dim=1)
        q = rearrange(q, "b (h d) n -> b h d n", h=self.heads) * self.scale
        k = rearrange(k, "b (h d) n -> b h d n", h=self.heads)
        v = rearrange(v, "b (h d) n -> b h d n", h=self.heads)
        attn = einsum("b h d i, b h d j -> b h i j", q, k).softmax(dim=-1)
        self.attention_scores = attn.mean(dim=1)
        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        return self.to_out(rearrange(out, "b h n d -> b (h d) n"))


class ReactionBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.cross_attention = CrossAttention(in_dim)
        self.projection = nn.Sequential(
            nn.Linear(2 * out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, target: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        source_to_target = torch.relu(self.cross_attention(source.unsqueeze(-1), target.unsqueeze(-1))).squeeze(-1)
        return self.projection(torch.cat([source_to_target, target], dim=-1))


class GraphDiffusion(nn.Module):
    def __init__(self, edge_index: torch.Tensor, edge_weight: torch.Tensor, node_dim: int = 1) -> None:
        super().__init__()
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", edge_weight)
        self.conv1 = GCNConv(node_dim, 4 * node_dim)
        self.conv2 = GCNConv(4 * node_dim, node_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x.unsqueeze(-1), self.edge_index, self.edge_weight)
        out = F.leaky_relu(out)
        out = self.conv2(out, self.edge_index, self.edge_weight)
        return torch.relu(out)


class AlzheimerODEGenerator(nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        survival_model,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        method: str = "euler",
        atol: float = 1e-6,
        rtol: float = 1e-6,
    ) -> None:
        super().__init__()
        self.method = method
        self.atol = atol
        self.rtol = rtol
        self.survival_model = survival_model
        self.survival_model.freeze()

        self.n_to_a = ReactionBlock(in_dim, out_dim, hidden_dim)
        self.a_to_t = ReactionBlock(in_dim, out_dim, hidden_dim)
        self.t_to_n = ReactionBlock(in_dim, out_dim, hidden_dim)
        self.a_diffusion = GraphDiffusion(edge_index=edge_index, edge_weight=edge_weight)
        self.t_diffusion = GraphDiffusion(edge_index=edge_index, edge_weight=edge_weight)
        self.a_nature_emb = nn.Sequential(nn.Linear(2 * in_dim, out_dim), nn.LeakyReLU(0.2, inplace=True), nn.Linear(out_dim, out_dim))
        self.t_nature_emb = nn.Sequential(nn.Linear(2 * in_dim, out_dim), nn.LeakyReLU(0.2, inplace=True), nn.Linear(out_dim, out_dim))
        self.n_nature_emb = nn.Sequential(nn.Linear(2 * in_dim, out_dim), nn.LeakyReLU(0.2, inplace=True), nn.Linear(out_dim, out_dim))
        self.init_emb = InitUNet1D(dim=64, dim_mults=(1, 2, 4, 8), channels=1, n_classes=5)

    def _split_atn(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return state[:, :REGION_DIM], state[:, REGION_DIM:2 * REGION_DIM], state[:, 2 * REGION_DIM:ATN_DIM]

    def _time_grid(self, times: float | torch.Tensor, device: torch.device) -> torch.Tensor:
        if isinstance(times, torch.Tensor):
            is_default_horizon = bool((times == 0).all().detach().cpu().item())
            horizon = times if not is_default_horizon else DEFAULT_HORIZON
        else:
            is_default_horizon = times == 0
            horizon = DEFAULT_HORIZON if is_default_horizon else times
        return torch.linspace(0, horizon, steps=ODE_STEPS, dtype=torch.float32, device=device)

    def _initial_state(
        self,
        amyloid: torch.Tensor,
        tau: torch.Tensor,
        ctx: torch.Tensor,
        label: torch.Tensor,
        age: torch.Tensor,
        noise: torch.Tensor | int,
        predict: bool,
    ) -> torch.Tensor:
        if predict:
            atn = torch.cat([amyloid, tau, ctx], dim=-1)
        else:
            if not isinstance(noise, torch.Tensor):
                noise = torch.zeros(amyloid.shape[0], REGION_DIM, device=amyloid.device)
            label_onehot = F.one_hot(label.long(), 5).float()
            atn = self.init_emb(noise.view(noise.shape[0], 1, -1), label_onehot, age).view(noise.shape[0], -1)
        return torch.cat([atn, age.unsqueeze(-1)], dim=-1)

    def pde(self, current_time: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        amyloid, tau, ctx = self._split_atn(state)
        age = state[:, -1] + current_time
        atn = state[:, :ATN_DIM]

        survival = self.survival_model.predict_survival(atn, age).view(amyloid.shape[0], -1)
        amyloid_emb = self.a_nature_emb(torch.cat([survival, amyloid], dim=-1))
        tau_emb = self.t_nature_emb(torch.cat([survival, tau], dim=-1))
        ctx_emb = self.n_nature_emb(torch.cat([survival, ctx], dim=-1))

        d_amyloid = amyloid_emb + self.n_to_a(amyloid, ctx) - self.a_diffusion(amyloid).squeeze(-1)
        d_tau = tau_emb + self.a_to_t(tau, amyloid) - self.t_diffusion(tau).squeeze(-1)
        d_ctx = ctx_emb + self.t_to_n(ctx, tau)
        return torch.cat([d_amyloid, d_tau, d_ctx, torch.zeros_like(age).unsqueeze(-1)], dim=-1)

    def forward(
        self,
        amyloid: torch.Tensor,
        tau: torch.Tensor,
        ctx: torch.Tensor,
        label: torch.Tensor,
        age: torch.Tensor,
        times: float | torch.Tensor = 0,
        noise: torch.Tensor | int = 0,
        predict: bool = False,
    ):
        time_steps = self._time_grid(times, amyloid.device)
        initial_state = self._initial_state(amyloid, tau, ctx, label, age, noise, predict)

        trajectory = odeint(self.pde, initial_state, time_steps, method=self.method, atol=self.atol, rtol=self.rtol)
        if torch.isnan(trajectory).any():
            raise ValueError("ODE trajectory contains NaN")
        generated = torch.stack([torch.clamp(step, 0, 1)[:, :ATN_DIM] for step in trajectory], dim=1)
        return generated


class AlzheimerODEDiscriminator(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, num_layers: int = 2) -> None:
        super().__init__()

        def block(in_features: int, out_features: int, use_norm: bool = True):
            layers: list[nn.Module] = [nn.Linear(in_features, out_features)]
            if use_norm:
                layers.append(nn.LayerNorm(out_features, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = block(in_dim, hidden_dim, False)
        for _ in range(num_layers - 1):
            layers += block(hidden_dim, hidden_dim)
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class QNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, num_layers: int = 2) -> None:
        super().__init__()

        def block(in_features: int, out_features: int, use_norm: bool = True):
            layers: list[nn.Module] = [nn.Linear(in_features, out_features)]
            if use_norm:
                layers.append(nn.LayerNorm(out_features, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = block(in_dim, hidden_dim, False)
        for _ in range(num_layers - 1):
            layers += block(hidden_dim, hidden_dim)
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
