from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from alzheimer_ode.data.dataset import ATNDataset, ATN_DIM, REGION_DIM, build_subject_groups
from alzheimer_ode.models import ATNSystem, AlzheimerODEDiscriminator, AlzheimerODEGenerator, QNet
from alzheimer_ode.utils import distribution_metrics, root_mean_square_error


DEFAULT_HORIZON = 5.0
ODE_STEPS = 21


@dataclass(frozen=True)
class EvaluationSamples:
    samples: torch.Tensor
    real: torch.Tensor
    sample_trajs: tuple[torch.Tensor, ...]
    real_trajs: tuple[torch.Tensor, ...]


def sample_noise(batch_size: int, noise_dim: int, device: torch.device) -> torch.Tensor:
    return torch.randn(batch_size, noise_dim, device=device)


class AlzheimerODETrainer:
    def __init__(
        self,
        train_dataset: ATNDataset,
        test_dataset: ATNDataset,
        survival_model,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        device: str | torch.device,
        noise_dim: int = 148,
        batch_size: int = 8,
        epochs: int = 2000,
        lr: float = 1e-5,
        q_weight: float = 0.5,
        mse_weight: float = 1.0,
    ) -> None:
        self.device = torch.device(device)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.survival_model = survival_model.to(self.device)
        self.edge_index = edge_index.to(self.device)
        self.edge_weight = edge_weight.to(self.device)
        self.noise_dim = noise_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.q_weight = q_weight
        self.mse_weight = mse_weight
        self.stats = train_dataset.stats
        self.unlabeled_data, self.labeled_data = build_subject_groups(train_dataset)
        self.train_loader = DataLoader(self.labeled_data, batch_size=batch_size, shuffle=True)

        self.generator = AlzheimerODEGenerator(
            in_dim=REGION_DIM,
            out_dim=REGION_DIM,
            hidden_dim=64,
            survival_model=survival_model,
            edge_index=self.edge_index,
            edge_weight=self.edge_weight,
        ).to(self.device)
        self.discriminator = AlzheimerODEDiscriminator(in_dim=ATN_DIM * 2, out_dim=1, hidden_dim=64, num_layers=2).to(self.device)
        self.qnet = QNet(in_dim=ATN_DIM + 6, out_dim=noise_dim, hidden_dim=64).to(self.device)
        self.atn_system = ATNSystem(self.device)

        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.9, 0.999))
        self.generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.9, 0.999))
        self.q_optimizer = torch.optim.Adam(self.qnet.parameters(), lr=lr, betas=(0.9, 0.999))

    def _discriminator_loss(self, real_logits, fake_logits, fake_unlabeled_logits):
        return (
            -torch.mean(torch.log(1.0 - torch.sigmoid(real_logits) + 1e-8))
            - torch.mean(torch.log(torch.sigmoid(fake_logits) + 1e-8))
            - torch.mean(torch.log(torch.sigmoid(fake_unlabeled_logits) + 1e-8))
        )

    def _generator_loss(self, fake_logits, fake_unlabeled_logits):
        return torch.mean(fake_logits) + torch.mean(fake_unlabeled_logits)

    @staticmethod
    def _merge_atn(amyloid: torch.Tensor, tau: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        return torch.cat([amyloid, tau, ctx], dim=-1)

    def _unlabeled_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        atn, age, label = [], [], []
        for subject in self.unlabeled_data:
            atn.append(self._merge_atn(subject[0], subject[1], subject[2]))
            age.append(subject[3])
            label.append(subject[4])
        return torch.stack(atn, dim=0).to(self.device), torch.stack(age, dim=0).to(self.device), torch.stack(label, dim=0).to(self.device)

    def _score_atn(self, atn: torch.Tensor) -> torch.Tensor:
        atn_derivative = self.atn_system.parameter_calculate(
            atn[..., :REGION_DIM],
            atn[..., REGION_DIM:2 * REGION_DIM],
            atn[..., 2 * REGION_DIM:],
        )
        return self.discriminator(torch.cat([atn, atn_derivative], dim=-1))

    def _score_generated_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        logits = [self._score_atn(trajectory[:, time_index]) for time_index in range(trajectory.shape[1])]
        return torch.stack(logits, dim=1).mean(dim=1)

    def _score_real_visits(self, amyloid: torch.Tensor, tau: torch.Tensor, ctx: torch.Tensor, age: torch.Tensor) -> torch.Tensor:
        logits = []
        for idx in range(amyloid.shape[0]):
            valid = age[idx] > 0
            real_atn = self._merge_atn(amyloid[idx, valid], tau[idx, valid], ctx[idx, valid])
            logits.append(self._score_atn(real_atn).mean(dim=0))
        return torch.stack(logits, dim=0)

    def _last_visit_regularizer(self, gen_data, real_data, age):
        unnormalized_gen = self.stats.unnormalize(gen_data)
        unnormalized_real = self.stats.unnormalize(real_data)
        relative_age = torch.where(age == 0, torch.zeros_like(age), age - age[:, 0].unsqueeze(-1))

        max_time = torch.tensor(0.0, device=self.device)
        for idx in range(age.shape[0]):
            span = relative_age[idx, -1] - relative_age[idx, 0]
            if span > max_time:
                max_time = span
        if float(max_time.item()) == 0.0:
            max_time = torch.tensor(DEFAULT_HORIZON, device=self.device)

        generated_time_points = torch.linspace(0, max_time, steps=ODE_STEPS, device=self.device)[1:]
        target_time = torch.max(relative_age, dim=-1).values.unsqueeze(-1)
        closest_indices = torch.argmin(torch.abs(generated_time_points - target_time), dim=1)

        generated_visits = []
        real_visits = []
        for idx in range(age.shape[0]):
            valid_relative = torch.nonzero(relative_age[idx], as_tuple=False).flatten()
            if valid_relative.numel() == 0:
                continue
            last_visit = int(valid_relative.max().item())
            generated_visits.append(unnormalized_gen[idx, closest_indices[idx]])
            real_visits.append(unnormalized_real[idx, last_visit])

        if not generated_visits:
            return torch.zeros((), device=self.device)
        return F.mse_loss(torch.stack(generated_visits), torch.stack(real_visits))

    def _initial_visit_regularizer(self, labeled_gen, labeled_real, unlabeled_gen, unlabeled_real):
        labeled_loss = F.mse_loss(labeled_gen[:, 0, :], labeled_real[:, 0, :])
        unlabeled_loss = F.mse_loss(unlabeled_gen[:, 0, :], unlabeled_real[:, 0, :])
        return labeled_loss + unlabeled_loss

    def _mse_regularizer(self, gen_data, real_data, age, unlabeled_gen, unlabeled_real):
        initial_loss = self._initial_visit_regularizer(gen_data, real_data, unlabeled_gen, unlabeled_real)
        final_loss = self._last_visit_regularizer(gen_data, real_data, age)
        return initial_loss + final_loss

    def _batch_maxtime(self, age: torch.Tensor) -> torch.Tensor:
        max_time = torch.tensor(0.0, device=self.device)
        for idx in range(age.shape[0]):
            valid = age[idx] > 0
            if valid.any():
                span = age[idx][valid][-1] - age[idx][valid][0]
                if span > max_time:
                    max_time = span
        if float(max_time.item()) == 0.0:
            max_time = torch.tensor(DEFAULT_HORIZON, device=self.device)
        return max_time

    def _train_discriminator(self, amyloid, tau, ctx, age, label):
        self.discriminator_optimizer.zero_grad()

        unlabeled_real, unlabeled_age, unlabeled_label = self._unlabeled_batch()

        horizon = self._batch_maxtime(age)
        real_logits = self._score_real_visits(amyloid, tau, ctx, age)

        noise = sample_noise(amyloid.shape[0], self.noise_dim, self.device)
        fake_data = self.generator(
            amyloid[:, 0, :],
            tau[:, 0, :],
            ctx[:, 0, :],
            label[:, 0],
            age[:, 0],
            times=horizon,
            noise=noise,
        )
        fake_logits = self._score_generated_trajectory(fake_data)

        unlabeled_noise = sample_noise(unlabeled_real.shape[0], self.noise_dim, self.device)
        unlabeled_fake = self.generator(
            unlabeled_real[:, 0, :REGION_DIM],
            unlabeled_real[:, 0, REGION_DIM:2 * REGION_DIM],
            unlabeled_real[:, 0, 2 * REGION_DIM:],
            unlabeled_label[:, 0],
            unlabeled_age[:, 0],
            noise=unlabeled_noise,
        )
        unlabeled_fake_logits = self._score_generated_trajectory(unlabeled_fake)

        loss = self._discriminator_loss(real_logits, fake_logits, unlabeled_fake_logits)
        loss.backward()
        self.discriminator_optimizer.step()
        return loss

    def _train_generator(self, amyloid, tau, ctx, age, label):
        self.generator_optimizer.zero_grad()

        unlabeled_real, unlabeled_age, unlabeled_label = self._unlabeled_batch()

        horizon = self._batch_maxtime(age)
        noise = sample_noise(amyloid.shape[0], self.noise_dim, self.device)
        gen_data = self.generator(
            amyloid[:, 0, :],
            tau[:, 0, :],
            ctx[:, 0, :],
            label[:, 0],
            age[:, 0],
            times=horizon,
            noise=noise,
        )
        fake_logits = self._score_generated_trajectory(gen_data)

        unlabeled_noise = sample_noise(unlabeled_real.shape[0], self.noise_dim, self.device)
        unlabeled_fake = self.generator(
            unlabeled_real[:, 0, :REGION_DIM],
            unlabeled_real[:, 0, REGION_DIM:2 * REGION_DIM],
            unlabeled_real[:, 0, 2 * REGION_DIM:],
            unlabeled_label[:, 0],
            unlabeled_age[:, 0],
            noise=unlabeled_noise,
        )
        unlabeled_fake_logits = self._score_generated_trajectory(unlabeled_fake)

        adv_loss = self._generator_loss(fake_logits, unlabeled_fake_logits)
        mse_regularizer = self._mse_regularizer(
            gen_data,
            self._merge_atn(amyloid, tau, ctx),
            age,
            unlabeled_fake,
            unlabeled_real,
        )

        q_generator_loss = torch.zeros((), device=self.device)
        if self.q_weight > 0:
            for time_index in range(gen_data.shape[1]):
                q_input = torch.cat(
                    [
                        F.one_hot(label[:, 0].long(), 5).float(),
                        age[:, 0].unsqueeze(-1),
                        gen_data[:, time_index],
                    ],
                    dim=-1,
                )
                q_pred = self.qnet(q_input)
                q_generator_loss = q_generator_loss + F.mse_loss(q_pred, noise)
            q_generator_loss = q_generator_loss / gen_data.shape[1]

        total_loss = adv_loss + self.mse_weight * mse_regularizer
        if self.q_weight > 0:
            total_loss = total_loss + self.q_weight * q_generator_loss
        total_loss.backward()
        self.generator_optimizer.step()

        self.q_optimizer.zero_grad()
        q_noise = sample_noise(amyloid.shape[0], self.noise_dim, self.device)
        q_data = self.generator(
            amyloid[:, 0, :],
            tau[:, 0, :],
            ctx[:, 0, :],
            label[:, 0],
            age[:, 0],
            times=horizon,
            noise=q_noise,
        )
        q_loss = torch.zeros((), device=self.device)
        for time_index in range(q_data.shape[1]):
            q_input = torch.cat(
                [
                    F.one_hot(label[:, 0].long(), 5).float(),
                    age[:, 0].unsqueeze(-1),
                    q_data[:, time_index],
                ],
                dim=-1,
            )
            q_pred = self.qnet(q_input)
            q_loss = q_loss + F.mse_loss(q_pred, q_noise)
        q_loss = q_loss / q_data.shape[1]
        q_loss.backward()
        self.q_optimizer.step()

        return {
            "g": float(adv_loss.detach().cpu()),
            "d": 0.0,
            "mse": float(mse_regularizer.detach().cpu()),
            "q": float(q_loss.detach().cpu()),
        }

    def fit(self, save_dir: str | Path | None = None, progress: bool = True, save_every: int = 50):
        history = []
        epoch_iter = tqdm(range(self.epochs), desc="AlzheimerODE train", disable=not progress)
        for epoch in epoch_iter:
            d_loss_total = 0.0
            g_loss_total = 0.0
            mse_total = 0.0
            q_total = 0.0
            for batch in self.train_loader:
                amyloid, tau, ctx, age, label, _ = batch
                d_loss = self._train_discriminator(amyloid, tau, ctx, age, label)
                metrics = self._train_generator(amyloid, tau, ctx, age, label)
                d_loss_total += float(d_loss.detach().cpu())
                g_loss_total += metrics["g"]
                mse_total += metrics["mse"]
                q_total += metrics["q"]

            rmse = self.evaluate_rmse(preserve_rng=True)
            summary = {
                "epoch": epoch,
                "d_loss": d_loss_total,
                "g_loss": g_loss_total,
                "mse_regularizer": mse_total,
                "q_loss": q_total,
                "rmse": rmse,
            }
            history.append(summary)
            epoch_iter.set_postfix(
                {
                    "d_loss": f"{d_loss_total:.4f}",
                    "g_loss": f"{g_loss_total:.4f}",
                    "q_loss": f"{q_total:.4f}",
                    "rmse": f"{rmse:.4f}",
                }
            )
            if save_dir is not None and save_every > 0 and (epoch + 1) % save_every == 0:
                self.save_checkpoint(Path(save_dir) / f"alzheimer_ode_epoch_{epoch + 1}.pt", epoch=epoch + 1)
        return history

    def _rng_state(self):
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        return torch.get_rng_state(), cuda_state

    @staticmethod
    def _restore_rng_state(state) -> None:
        cpu_state, cuda_state = state
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)

    def _evaluation_samples(self, preserve_rng: bool = False) -> EvaluationSamples:
        rng_state = self._rng_state() if preserve_rng else None
        generator_training = self.generator.training
        discriminator_training = self.discriminator.training
        qnet_training = self.qnet.training
        self.generator.eval()
        self.discriminator.eval()
        self.qnet.eval()
        test_data = []
        test_labels = []
        test_times = []

        try:
            with torch.no_grad():
                for subject in self.test_dataset:
                    amyloid, tau, ctx, age, label, _ = subject
                    test_data.append(self._merge_atn(amyloid, tau, ctx))
                    test_labels.append(label)
                    test_times.append(age)

                if not test_data:
                    raise ValueError("No test data found for evaluation")

                test_data_tensor = torch.stack(test_data, dim=0).to(self.device)
                test_labels_tensor = torch.stack(test_labels, dim=0).to(self.device)
                test_times_tensor = torch.stack(test_times, dim=0).to(self.device)
                noise = sample_noise(test_data_tensor.shape[0], self.noise_dim, self.device)
                sample_full = self.generator(
                    test_data_tensor[:, 0, :REGION_DIM],
                    test_data_tensor[:, 0, REGION_DIM:2 * REGION_DIM],
                    test_data_tensor[:, 0, 2 * REGION_DIM:],
                    test_labels_tensor[:, 0],
                    test_times_tensor[:, 0],
                    noise=noise,
                )

                time_grid = torch.linspace(0, DEFAULT_HORIZON, steps=ODE_STEPS, device=self.device)
                sample_points = []
                real_points = []
                sample_trajs = []
                real_trajs = []
                for subject_index in range(test_times_tensor.shape[0]):
                    valid_indices = torch.where(test_times_tensor[subject_index] > 0)[0]
                    if valid_indices.numel() == 0:
                        continue
                    first_time = test_times_tensor[subject_index, valid_indices[0]]
                    subject_samples = []
                    subject_real = []
                    for visit_index in valid_indices:
                        relative_time = test_times_tensor[subject_index, visit_index] - first_time
                        closest_index = torch.argmin(torch.abs(time_grid - relative_time))
                        sample_visit = sample_full[subject_index, closest_index]
                        real_visit = test_data_tensor[subject_index, visit_index]
                        sample_points.append(sample_visit)
                        real_points.append(real_visit)
                        subject_samples.append(sample_visit)
                        subject_real.append(real_visit)

                    if subject_samples:
                        sample_trajs.append(self.test_dataset.unnormalize_atn(torch.stack(subject_samples)))
                        real_trajs.append(self.test_dataset.unnormalize_atn(torch.stack(subject_real)))

                if not sample_points:
                    raise ValueError("No valid test visits found for evaluation")

                samples = self.test_dataset.unnormalize_atn(torch.stack(sample_points))
                real = self.test_dataset.unnormalize_atn(torch.stack(real_points))
                return EvaluationSamples(
                    samples=samples,
                    real=real,
                    sample_trajs=tuple(sample_trajs),
                    real_trajs=tuple(real_trajs),
                )
        finally:
            self.generator.train(generator_training)
            self.discriminator.train(discriminator_training)
            self.qnet.train(qnet_training)
            if rng_state is not None:
                self._restore_rng_state(rng_state)

    def _evaluation_points(self, preserve_rng: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        samples = self._evaluation_samples(preserve_rng=preserve_rng)
        return samples.samples, samples.real

    def evaluate_rmse(self, preserve_rng: bool = False) -> float:
        generated_flat, real_flat = self._evaluation_points(preserve_rng=preserve_rng)
        return root_mean_square_error(generated_flat, real_flat)

    def evaluate(self):
        samples = self._evaluation_samples()
        return distribution_metrics(
            samples.samples,
            samples.real,
            sample_trajs=samples.sample_trajs,
            real_trajs=samples.real_trajs,
        )

    def save_checkpoint(self, path: str | Path, epoch: int = 0):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "generator": self.generator.state_dict(),
                "discriminator": self.discriminator.state_dict(),
                "qnet": self.qnet.state_dict(),
                "survival": self.survival_model.state_dict(),
                "survival_class": self.survival_model.__class__.__name__,
            },
            path,
        )

    def load_checkpoint(self, path: str | Path):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        self.generator.load_state_dict(payload["generator"])
        self.discriminator.load_state_dict(payload["discriminator"])
        self.qnet.load_state_dict(payload["qnet"])
        if "survival" in payload:
            self.survival_model.load_state_dict(payload["survival"])
        return payload.get("epoch", 0)
