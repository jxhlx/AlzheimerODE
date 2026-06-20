from typing import Tuple

import torch
from einops import rearrange
try:
    from ema_pytorch import EMA
except ImportError:
    EMA = None
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.util import lagged_sequence_values
from gluonts.transform.split import InstanceSplitter
try:
    from torchtyping import TensorType, patch_typeguard
except ImportError:
    class TensorType:
        def __class_getitem__(cls, item):
            return torch.Tensor

    def patch_typeguard():
        return None
from typeguard import typechecked

from tsflow.arch import BackboneModel
# from tsflow.arch.backbones import BackboneModelMultivariate
from tsflow.arch import BackboneModel as BackboneModelMultivariate
from tsflow.model._base import PREDICTION_INPUT_NAMES, TSFlowBase
from tsflow.utils.gaussian_process import Q0Dist
from tsflow.utils.util import LongScaler
from tsflow.utils.variables import Prior, Setting

patch_typeguard()


class TSFlowCond(TSFlowBase):
    def __init__(
        self,
        setting: str,
        target_dim: int,
        context_length: int,
        prediction_length: int,
        backbone_params: dict,
        prior_params: dict,
        optimizer_params: dict,
        ema_params: dict,
        frequency: str,
        normalization: str | None = None,
        use_lags: bool = True,
        use_ema: bool = False,
        num_steps: int = 16,
        solver: str = "euler",
        matching: str = "random",
        multi_series_univariate: bool = False,
        static_feature_dim: int = 0,
    ):
        super().__init__(
            context_length=context_length,
            prediction_length=prediction_length,
            prior_params=prior_params,
            optimizer_params=optimizer_params,
            frequency=frequency,
            normalization=normalization,
            use_lags=use_lags,
            use_ema=use_ema,
            num_steps=num_steps,
            solver=solver,
            matching=matching,
        )
        num_features = 2 + (len(self.lags_seq) if use_lags else 0)
        num_features += static_feature_dim

        if setting == Setting.UNIVARIATE and multi_series_univariate:
            target_dim = target_dim
        else:
            target_dim = target_dim if setting == Setting.MULTIVARIATE else 1

        if setting == Setting.UNIVARIATE:
            self.backbone = BackboneModel(
                **backbone_params,
                num_features=num_features,
                target_dim=target_dim,
            )
        else:
            self.backbone = BackboneModelMultivariate(
                **backbone_params,
                num_features=num_features,
                target_dim=target_dim,
            )
        if EMA is None:
            raise ImportError("ema_pytorch is required for TSFlow EMA.")
        self.ema_backbone = EMA(self.backbone, **ema_params)
        self.setting = setting
        self.multi_series_univariate = multi_series_univariate
        self.target_dim = target_dim
        self.guidance_scale = 0
        self.sigmax = self.sigmin
        self.q0 = Q0Dist(
            **prior_params,
            prediction_length=prediction_length,
            freq=self.freq,
            iso=1e-1 if self.prior != Prior.ISO else 0,
        )

    @typechecked
    def _extract_features(
        self, data: dict
    ) -> Tuple[
        TensorType[float, "batch", "length", "num_series"],
        TensorType[float, "batch", "length", "num_series"],
        TensorType[float, "batch", "length", "num_series"],
        TensorType[float, "batch", 1, "num_series"],
        TensorType[float, "batch", 1, "num_series"],
        TensorType[float, "batch", "length", "num_series", "num_features"],
    ]:
        past = data["past_target"]
        future = data["future_target"]
        context_observed = data["past_observed_values"]
        mean = data.get("mean", None)
        # id = data["id"]
        if self.setting == Setting.UNIVARIATE and not self.multi_series_univariate:
            past = rearrange(past, "... -> ... 1")
            future = rearrange(future, "... -> ... 1")
            context_observed = rearrange(context_observed, "... -> ... 1")
            if mean is None:
                mean = torch.ones((past.shape[0], 1), device=past.device)
            mean = rearrange(mean, "... -> ... 1")
        else:
            if mean is None:
                mean = torch.ones((past.shape[0], 1, past.shape[-1]), device=past.device)

        atn_mode = bool(data.get("atn_mode", False))

        context = past[:, -self.context_length :]
        long_context = past[:, : -self.context_length]
        prior_context = past[:, -self.prior_context_length :]

        if atn_mode and self.context_length == 0:
            loc = torch.zeros((future.shape[0], 1, future.shape[-1]), device=future.device)
            scale = torch.ones((future.shape[0], 1, future.shape[-1]), device=future.device)
            scaled_context = torch.empty((future.shape[0], 0, future.shape[-1]), device=future.device)
            scaled_future = torch.nan_to_num(future, nan=0.0)
            if scaled_future.shape[1] == 0:
                scaled_future = torch.zeros(
                    (future.shape[0], self.prediction_length, future.shape[-1]),
                    device=future.device,
                )
            scaled_long_context = torch.empty_like(scaled_context)
            scaled_prior_context = torch.empty_like(scaled_context)
        else:
            if isinstance(self.scaler, LongScaler):
                scaled_context, loc, scale = self.scaler(context, scale=mean)
            else:
                _, loc, scale = self.scaler(past, context_observed)
                scaled_context = context / scale
            scaled_long_context = (long_context - loc) / scale
            scaled_prior_context = (prior_context - loc) / scale
            scaled_future = (future - loc) / scale

        if atn_mode and self.context_length == 0:
            x1 = scaled_future
            x0 = torch.randn_like(x1)

            future_observed = data.get("future_observed_values", None)
            if future_observed is not None:
                observation_mask = future_observed.to(x1.device)
            else:
                observation_mask = torch.ones_like(x1)
            if observation_mask.shape[1] == 0:
                observation_mask = torch.ones_like(x1)

            static_real = data.get("feat_static_real", None)
            static_cat = data.get("feat_static_cat", None)
            if static_real is not None:
                static_real = static_real.to(x1.device)
            if static_cat is not None:
                static_cat = static_cat.to(x1.device)

            time_feats = []
            if static_real is not None:
                sr = static_real.unsqueeze(1).unsqueeze(2).expand(-1, self.prediction_length, x1.shape[-1], -1)
                time_feats.append(sr)

            if static_cat is not None:
                label_val = static_cat.squeeze(-1).long()
                label_oh = torch.nn.functional.one_hot(label_val, num_classes=5).float()
                sc = label_oh.unsqueeze(1).unsqueeze(2).expand(-1, self.prediction_length, x1.shape[-1], -1)
                time_feats.append(sc)

            features = []
            features.append(torch.zeros_like(observation_mask).unsqueeze(-1))
            features.append(observation_mask.unsqueeze(-1))
            if len(time_feats) > 0:
                features.append(torch.cat(time_feats, dim=-1))
            features = torch.cat(features, dim=-1)
            return x1, x0, observation_mask, loc, scale, features

        x1 = torch.cat([scaled_context, scaled_future], dim=-2)
        batch_size, length, c = x1.shape

        observation_mask = torch.zeros_like(x1)
        observation_mask[:, : -self.prediction_length] = context_observed[:, -self.context_length :]

        features = []
        if self.use_lags:
            lags = lagged_sequence_values(
                self.lags_seq,
                scaled_long_context,
                x1,
                dim=1,
            )
            features.append(lags)

        dist = self.q0.gp_regression(rearrange(scaled_prior_context, "b l c -> (b c) l"), self.prediction_length)

        fut = rearrange(dist.sample(), "(b c) l -> b l c", c=c)
        fut_mean = rearrange(dist.mean, "(b c) l -> b l c", c=c)
        fut_std = torch.diagonal(dist.covariance_matrix, dim1=-2, dim2=-1)
        fut_std = rearrange(fut_std, "(b c) ... -> b ... c", c=c)
        features.append(torch.cat([scaled_context, fut_mean], dim=-2).unsqueeze(-1))
        features.append(observation_mask.unsqueeze(-1))
        x0 = torch.cat([scaled_context, fut], dim=-2)

        features = torch.cat(features, dim=-1)
        return x1, x0, observation_mask, loc, scale, features

    @typechecked
    def training_step(self, data: dict, idx: int) -> dict:
        assert self.training is True
        x1, x0, observation_mask, _, _, features = self._extract_features(data)
        use_mask = bool(data.get("atn_mode", False))
        t = torch.rand((x1.shape[0], 1), device=self.device)
        loss = self.p_losses(x1, x0, t, features, observation_mask if use_mask else None)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            batch_size=x1.shape[0],
            on_epoch=True,
            logger=True,
        )
        return {"loss": loss}

    @typechecked
    def p_losses(
        self,
        x1: TensorType[float, "batch", "length", "num_series"],
        x0: TensorType[float, "batch", "length", "num_series"],
        t: TensorType[float, "batch", 1],
        features: TensorType[float, "batch", "length", "num_series", "num_features"] | None = None,
        observation_mask: TensorType[float, "batch", "length", "num_series"] | None = None,
    ) -> TensorType[float]:
        num_dims_to_add = x1.dim() - t.dim()
        t = t.unsqueeze(-1) if num_dims_to_add == 1 else t.unsqueeze(-1).unsqueeze(-1)

        psi, dpsi = self.forward_path(x1, x0, t)
        predicted_flow = self.backbone(t, psi, features)

        if observation_mask is not None:
            loss = torch.nn.functional.mse_loss(dpsi, predicted_flow, reduction="none")
            loss = (loss * observation_mask).sum() / (observation_mask.sum() + 1e-8)
        else:
            loss = torch.nn.functional.mse_loss(dpsi, predicted_flow)
        return loss

    @typechecked
    def forward(
        self,
        past_target: TensorType[float, "batch", "length"] | TensorType[float, "batch", "length", "num_series"],
        past_observed_values: TensorType[float, "batch", "length"] | TensorType[float, "batch", "length", "num_series"],
        mean: TensorType[float, "batch", 1] | TensorType[float, "batch", 1, "num_series"] = None,
        feat_static_real: TensorType[float, "batch", 1] | None = None,
        feat_static_cat: TensorType[float, "batch", 1] | None = None,
    ) -> (
        TensorType[float, "batch", "num_samples", "prediction_length"]
        | TensorType[float, "batch", "num_samples", "prediction_length", "num_series"]
    ):
        # This is only used during prediction
        past_target = past_target.to(self.device).repeat_interleave(self.num_samples, dim=0)
        past_observed_values = past_observed_values.to(self.device).repeat_interleave(self.num_samples, dim=0)
        mean = mean.to(self.device).repeat_interleave(self.num_samples, dim=0)
        if past_target.shape[1] == 0:
            future_target = torch.zeros(
                (past_target.shape[0], self.prediction_length, self.target_dim),
                device=past_target.device,
            )
        else:
            future_target = torch.zeros_like(past_target[:, -self.prediction_length :])

        data = dict(
            past_target=past_target,
            past_observed_values=past_observed_values,
            mean=mean,
            future_target=future_target,
            feat_static_real=feat_static_real,
            feat_static_cat=feat_static_cat,
            atn_mode=bool(feat_static_real is not None or feat_static_cat is not None),
        )
        observation, x0, observation_mask, loc, scale, features = self._extract_features(data)
        x0 = x0 + self.sigmax * torch.randn_like(x0)
        pred = self.sample(
            x0.to(self.device),
            features=features,
            observation=observation,
            observation_mask=observation_mask,
            guidance_scale=self.guidance_scale,
        )
        if self.setting == Setting.UNIVARIATE and not self.multi_series_univariate:
            pred = rearrange(pred * scale + loc, "(b n) l 1 -> b n l", n=self.num_samples)
        else:
            pred = rearrange(pred * scale + loc, "(b n) l k -> b n l k", n=self.num_samples)
        return pred[:, :, observation.shape[1] - self.prediction_length :]

    def forward_atn(
        self,
        past_target,
        past_observed_values,
        mean=None,
        feat_static_real=None,
        feat_static_cat=None,
    ):
        past_target = past_target.to(self.device).repeat_interleave(self.num_samples, dim=0)
        past_observed_values = past_observed_values.to(self.device).repeat_interleave(self.num_samples, dim=0)
        mean = mean.to(self.device).repeat_interleave(self.num_samples, dim=0)

        if past_target.shape[1] == 0:
            future_target = torch.zeros(
                (past_target.shape[0], self.prediction_length, self.target_dim),
                device=past_target.device,
            )
        else:
            future_target = torch.zeros_like(past_target[:, -self.prediction_length :])

        data = dict(
            past_target=past_target,
            past_observed_values=past_observed_values,
            mean=mean,
            future_target=future_target,
            feat_static_real=feat_static_real,
            feat_static_cat=feat_static_cat,
            atn_mode=bool(feat_static_real is not None or feat_static_cat is not None),
        )
        observation, x0, observation_mask, loc, scale, features = self._extract_features(data)
        x0 = x0 + self.sigmax * torch.randn_like(x0)
        pred = self.sample(
            x0.to(self.device),
            features=features,
            observation=observation,
            observation_mask=observation_mask,
            guidance_scale=self.guidance_scale,
        )
        if self.setting == Setting.UNIVARIATE and not self.multi_series_univariate:
            pred = rearrange(pred * scale + loc, "(b n) l 1 -> b n l", n=self.num_samples)
        else:
            pred = rearrange(pred * scale + loc, "(b n) l k -> b n l k", n=self.num_samples)
        return pred[:, :, observation.shape[1] - self.prediction_length :]

    @typechecked
    def get_predictor(self, input_transform: InstanceSplitter, batch_size: int = 40, device: str | torch.device = None):
        return PyTorchPredictor(
            prediction_length=self.prediction_length,
            input_names=PREDICTION_INPUT_NAMES,
            prediction_net=self,
            batch_size=batch_size,
            input_transform=input_transform,
            device=device,
        )
