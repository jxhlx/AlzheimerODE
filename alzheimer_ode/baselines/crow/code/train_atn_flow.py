import argparse
import os
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.optim

from atn_dataset import ATNTrainDataset, ATNTorchDataset, build_atn_samples
from atn_evaluator import ATNEvaluator
from FrEIA.framework import InputNode, OutputNode, Node, ReversibleGraphNet
from FrEIA.modules.coupling_layers_mine import glow_gru_cgate_coupling_layer2
from FrEIA.modules.coeff_functs import F_GRU, F_cgate

ROOT_DIR = Path(__file__).resolve().parents[1]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mmd_multiscale(x, y, device):
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2.0 * xx
    dyy = ry.t() + ry - 2.0 * yy
    dxy = rx.t() + ry - 2.0 * zz

    XX = torch.zeros(xx.shape, device=device)
    YY = torch.zeros(xx.shape, device=device)
    XY = torch.zeros(xx.shape, device=device)

    for a in [0.2, 0.5, 0.9, 1.3]:
        XX += a ** 2 * (a ** 2 + dxx) ** -1
        YY += a ** 2 * (a ** 2 + dyy) ** -1
        XY += a ** 2 * (a ** 2 + dxy) ** -1

    return torch.mean(XX + YY - 2.0 * XY)


def fit_mse(input_tensor, target_tensor):
    return torch.mean((input_tensor - target_tensor) ** 2)


def build_model(ndim_tot, ndim_z):
    inp = InputNode(ndim_tot, name="input")
    t1 = Node(
        [inp.out0],
        glow_gru_cgate_coupling_layer2,
        {"F_class": F_GRU, "F_cgate": F_cgate, "clamp": 2.0, "F_args": {"dropout": 0.0}},
    )
    t2 = Node(
        [t1.out0],
        glow_gru_cgate_coupling_layer2,
        {"F_class": F_GRU, "F_cgate": F_cgate, "clamp": 2.0, "F_args": {"dropout": 0.0}},
    )
    outp = OutputNode([t2.out0], name="output")
    nodes = [inp, t1, t2, outp]
    model = ReversibleGraphNet(nodes)
    return model


def expand_conditions(age, label, num_classes: int | None, onehot: bool, T: int, device):
    if onehot:
        if num_classes is None:
            raise ValueError("num_classes is required when onehot is enabled")
        label_onehot = torch.zeros((label.shape[0], num_classes), device=device)
        label_onehot.scatter_(1, label.view(-1, 1), 1.0)
        cond = torch.cat([age.view(-1, 1), label_onehot], dim=1)
    else:
        cond = torch.cat([age.view(-1, 1), label.float().view(-1, 1)], dim=1)
    cond = cond.unsqueeze(1).repeat(1, T, 1)
    return cond


def filter_train_samples(samples: List, min_timepoints: int):
    filtered = []
    for sample in samples:
        if sample.mask.sum() >= min_timepoints:
            filtered.append(sample)
    return filtered


def train_epoch(
    model,
    train_loader,
    optimizer,
    device,
    ndim_x,
    ndim_y,
    ndim_z,
    ndim_tot,
    y_noise_scale,
    zeros_noise_scale,
    lambd_predict,
    lambd_latent,
    lambd_rev,
    loss_factor,
    num_classes,
    onehot,
    n_its_per_epoch,
):
    model.train()
    l_tot = 0.0
    batch_idx = 0

    for batch in train_loader:
        batch_idx += 1
        if batch_idx > n_its_per_epoch:
            break
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)
        age = batch["age"].to(device)
        label = batch["label"].to(device)

        x = target.permute(0, 2, 1).contiguous()
        x = torch.nan_to_num(x, nan=0.0)

        B, T, _ = x.shape
        y = expand_conditions(age, label, num_classes, onehot, T, device)

        pad_x = zeros_noise_scale * torch.randn(B, T, ndim_tot - ndim_x, device=device)
        pad_yz = zeros_noise_scale * torch.randn(B, T, ndim_tot - ndim_y - ndim_z, device=device)

        y_noisy = y + y_noise_scale * torch.randn(B, T, ndim_y, dtype=torch.float, device=device)

        x_in = torch.cat((x, pad_x), dim=2)
        y_in = torch.cat((torch.randn(B, T, ndim_z, device=device), pad_yz, y_noisy), dim=2)

        optimizer.zero_grad()

        output = model(x_in)

        l = 0.0
        for t in range(T):
            valid_idx = mask[:, t] > 0.5
            if valid_idx.sum() == 0:
                continue
            out_t = output[valid_idx, t, :]
            y_t = y_in[valid_idx, t, :]

            l += lambd_predict * fit_mse(out_t[:, ndim_z:], y_t[:, ndim_z:])

            y_short = torch.cat((y_t[:, :ndim_z], y_t[:, -ndim_y:]), dim=1)
            output_block_grad = torch.cat((out_t[:, :ndim_z], y_t[:, -ndim_y:].data), dim=1)
            l += lambd_latent * mmd_multiscale(output_block_grad, y_short, device)

        l_tot += float(l.detach().item())
        l.backward()

        pad_yz = zeros_noise_scale * torch.randn(B, T, ndim_tot - ndim_y - ndim_z, device=device)
        y_clean = y
        y_rev = y_clean + y_noise_scale * torch.randn(B, T, ndim_y, device=device)
        orig_z_perturbed = output.data[:, :, :ndim_z] + y_noise_scale * torch.randn(B, T, ndim_z, device=device)

        y_rev = torch.cat((orig_z_perturbed, pad_yz, y_rev), dim=2)
        y_rev_rand = torch.cat((torch.randn(B, T, ndim_z, device=device), pad_yz, y_rev[:, :, -ndim_y:]), dim=2)

        output_rev = model(y_rev, rev=True)
        output_rev_rand = model(y_rev_rand, rev=True)

        l_rev = 0.0
        for t in range(T):
            valid_idx = mask[:, t] > 0.5
            if valid_idx.sum() == 0:
                continue
            rev_rand_t = output_rev_rand[valid_idx, t, :ndim_x]
            x_t = x_in[valid_idx, t, :ndim_x]
            l_rev += lambd_rev * loss_factor * mmd_multiscale(rev_rand_t, x_t, device)

            rev_t = output_rev[valid_idx, t, :]
            x_full_t = x_in[valid_idx, t, :]
            l_rev += 0.50 * lambd_predict * fit_mse(rev_t, x_full_t)

        l_tot += float(l_rev.detach().item())
        l_rev.backward()

        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.clamp_(-5.00, 5.00)

        optimizer.step()

    return l_tot / max(batch_idx, 1)


def evaluate_model(
    model,
    test_loader,
    device,
    ndim_x,
    ndim_y,
    ndim_z,
    ndim_tot,
    num_classes,
    onehot,
):
    model.eval()
    evaluator = ATNEvaluator()
    all_samples = []
    all_real = []
    sample_trajectories = []
    real_trajectories = []

    with torch.no_grad():
        for batch in test_loader:
            target = batch["target"].to(device)
            mask = batch["mask"].to(device)
            age = batch["age"].to(device)
            label = batch["label"].to(device)

            x_real = target.permute(0, 2, 1).contiguous()
            B, T, _ = x_real.shape

            y = expand_conditions(age, label, num_classes, onehot, T, device)
            pad_yz = torch.zeros(B, T, ndim_tot - ndim_y - ndim_z, device=device)

            y_rev = torch.cat((torch.randn(B, T, ndim_z, device=device), pad_yz, y), dim=2)
            x_gen = model(y_rev, rev=True)[:, :, :ndim_x]

            for i in range(B):
                valid_idx = mask[i] > 0.5
                if valid_idx.sum() == 0:
                    continue
                sample_traj = x_gen[i][valid_idx].cpu()
                real_traj = x_real[i][valid_idx].cpu()
                sample_trajectories.append(sample_traj)
                real_trajectories.append(real_traj)

                all_samples.append(sample_traj)
                all_real.append(real_traj)

    if not all_samples:
        return None

    all_sample_tensor = torch.cat(all_samples, dim=0)
    all_real_tensor = torch.cat(all_real, dim=0)

    metrics = evaluator.metrics(all_sample_tensor, all_real_tensor)
    traj_metrics = evaluator.trajectory_metrics(sample_trajectories, real_trajectories)

    return {"metrics": metrics, "trajectory_metrics": traj_metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, default=str(ROOT_DIR / "data" / "ALL" / "train"))
    parser.add_argument("--test-path", type=str, default=str(ROOT_DIR / "data" / "ALL" / "test"))
    parser.add_argument("--json-file", type=str, default=str(ROOT_DIR / "data" / "TABLE_Destrieux.json"))
    parser.add_argument("--prediction-length", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ndim-z", type=int, default=4)
    parser.add_argument("--ndim-tot", type=int, default=0)
    parser.add_argument("--label-onehot", action="store_true", default=True)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--output-dir", type=str, default=str(ROOT_DIR / "checkpoints" / "atn"))
    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_samples, test_samples = build_atn_samples(
        train_path=args.train_path,
        test_path=args.test_path,
        json_file=args.json_file,
        prediction_length=args.prediction_length,
    )

    train_samples = filter_train_samples(train_samples, min_timepoints=2)

    if len(train_samples) == 0 or len(test_samples) == 0:
        raise RuntimeError("No train/test samples found after filtering.")

    ndim_x = train_samples[0].target.shape[0]
    if args.label_onehot:
        ndim_y = 1 + args.num_classes
    else:
        ndim_y = 2
    ndim_z = args.ndim_z
    ndim_tot = args.ndim_tot if args.ndim_tot > 0 else (ndim_x + ndim_y + ndim_z)

    model = build_model(ndim_tot, ndim_z).to(device)

    l2_reg = 2e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=l2_reg)
    gamma = 0.01 ** (1.0 / 120)
    meta_epoch = 12
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=meta_epoch, gamma=gamma)
    y_noise_scale = 3e-3
    zeros_noise_scale = 3e-2
    lambd_predict = 300.0
    lambd_latent = 300.0
    lambd_rev = 100.0
    n_its_per_epoch = 10

    batch_size = args.batch_size if args.batch_size > 0 else len(train_samples)
    train_loader = torch.utils.data.DataLoader(
        ATNTrainDataset(train_samples),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    test_loader = torch.utils.data.DataLoader(
        ATNTrainDataset(test_samples),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        loss_factor = 600 ** (float(epoch) / 300) / 600
        if loss_factor > 1:
            loss_factor = 1

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            ndim_x,
            ndim_y,
            ndim_z,
            ndim_tot,
            y_noise_scale,
            zeros_noise_scale,
            lambd_predict,
            lambd_latent,
            lambd_rev,
            loss_factor,
            args.num_classes,
            args.label_onehot,
            n_its_per_epoch=n_its_per_epoch,
        )

        if (epoch + 1) % meta_epoch == 0 or epoch == args.epochs - 1:
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

            if eval_results is not None:
                metrics = eval_results["metrics"]
                traj_metrics = eval_results["trajectory_metrics"]
                print(f"Epoch {epoch+1}: loss={train_loss:.4f}")
                print("Metrics:", metrics)
                print("Trajectory metrics:", traj_metrics)

        if (epoch + 1) % meta_epoch == 0 or epoch == args.epochs - 1:
            ckpt_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
        scheduler.step()


if __name__ == "__main__":
    main()
