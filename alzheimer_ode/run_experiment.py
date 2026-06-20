from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from alzheimer_ode.data.connectome import DEFAULT_CONNECTOME_PATH, load_average_connectome
from alzheimer_ode.data.dataset import ATN_DIM, REGION_DIM, ATNDataset, extract_survival_training_arrays
from alzheimer_ode.survival.cox_survival import CoxSurvivalModel
from alzheimer_ode.training.trainer import AlzheimerODETrainer
from alzheimer_ode.utils.seed import set_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "ATN"
DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT / "checkpoints"
DEFAULT_MODEL_CHECKPOINT_DIR = DEFAULT_CHECKPOINT_ROOT / "model"
DEFAULT_SURVIVAL_CHECKPOINT_DIR = DEFAULT_CHECKPOINT_ROOT / "survival"


def default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AlzheimerODE experiment runner")
    parser.add_argument("--stage", choices=["all", "pretrain", "train", "eval"], default="all")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--connectome-path", type=Path, default=DEFAULT_CONNECTOME_PATH)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_MODEL_CHECKPOINT_DIR)
    parser.add_argument("--survival-checkpoint", type=Path, default=DEFAULT_SURVIVAL_CHECKPOINT_DIR / "cox_survival.pt")
    parser.add_argument("--model-checkpoint", type=Path, default=DEFAULT_MODEL_CHECKPOINT_DIR / "alzheimer_ode_final.pt")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--survival-seed", type=int, default=None)
    parser.add_argument("--train-seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--noise-dim", type=int, default=REGION_DIM)
    parser.add_argument("--q-weight", type=float, default=0.5)
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--survival-epochs", type=int, default=100)
    parser.add_argument("--survival-batch-size", type=int, default=100)
    parser.add_argument("--survival-learning-rate", type=float, default=1e-3)
    parser.add_argument("--survival-patience", type=int, default=3)
    parser.add_argument("--survival-state-k", type=int, default=3)
    parser.add_argument("--retrain-survival", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def resolve_seed(primary: int | None, fallback: int | None) -> int | None:
    return primary if primary is not None else fallback


def build_datasets(args: argparse.Namespace) -> tuple[ATNDataset, ATNDataset]:
    train_dataset = ATNDataset(args.data_dir, "train", device=args.device)
    test_dataset = ATNDataset(args.data_dir, "test", device=args.device)
    return train_dataset, test_dataset


def pretrain_survival(args: argparse.Namespace, train_dataset: ATNDataset) -> CoxSurvivalModel:
    if args.survival_state_k < 2:
        raise ValueError("--survival-state-k must be at least 2")
    survival_components = args.survival_state_k - 1
    x, t, e, _ = extract_survival_training_arrays(train_dataset)
    model = CoxSurvivalModel(
        input_dim=ATN_DIM,
        survival_components=survival_components,
        region_specific=True,
        random_seed=resolve_seed(args.survival_seed, args.seed),
        device=args.device,
        region=REGION_DIM,
    )
    model.fit(
        x,
        t,
        e,
        val_data=(x, t, e),
        epochs=args.survival_epochs,
        learning_rate=args.survival_learning_rate,
        batch_size=args.survival_batch_size,
        patience=args.survival_patience,
        progress=not args.no_progress,
    )
    args.survival_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.survival_checkpoint)
    return model


def load_or_pretrain_survival(args: argparse.Namespace, train_dataset: ATNDataset) -> CoxSurvivalModel:
    if args.survival_checkpoint.exists() and not args.retrain_survival:
        return CoxSurvivalModel.load(args.survival_checkpoint, device=args.device)
    return pretrain_survival(args, train_dataset)


def build_trainer(args: argparse.Namespace, train_dataset: ATNDataset, test_dataset: ATNDataset, survival_model: CoxSurvivalModel) -> AlzheimerODETrainer:
    edge_index, edge_weight = load_average_connectome(args.connectome_path, device=args.device)
    return AlzheimerODETrainer(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        survival_model=survival_model,
        edge_index=edge_index,
        edge_weight=edge_weight,
        device=args.device,
        noise_dim=args.noise_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.learning_rate,
        q_weight=args.q_weight,
        mse_weight=args.mse_weight,
    )


def run_train(args: argparse.Namespace, train_dataset: ATNDataset, test_dataset: ATNDataset, survival_model: CoxSurvivalModel) -> dict:
    trainer = build_trainer(args, train_dataset, test_dataset, survival_model)
    if args.resume_checkpoint is not None:
        trainer.load_checkpoint(args.resume_checkpoint)
    history = trainer.fit(save_dir=args.checkpoint_dir, progress=not args.no_progress, save_every=args.save_every)
    trainer.save_checkpoint(args.model_checkpoint, epoch=args.epochs)
    metrics = trainer.evaluate()
    metrics_path = args.checkpoint_dir / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({"history": history, "metrics": metrics}, indent=2), encoding="utf-8")
    return metrics


def run_eval(args: argparse.Namespace, train_dataset: ATNDataset, test_dataset: ATNDataset, survival_model: CoxSurvivalModel) -> dict:
    trainer = build_trainer(args, train_dataset, test_dataset, survival_model)
    trainer.load_checkpoint(args.model_checkpoint)
    return trainer.evaluate()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    set_seed(resolve_seed(args.survival_seed, args.seed))
    train_dataset, test_dataset = build_datasets(args)

    if args.stage == "pretrain":
        pretrain_survival(args, train_dataset)
        print(f"saved survival checkpoint: {args.survival_checkpoint}")
        return

    survival_model = load_or_pretrain_survival(args, train_dataset)
    set_seed(resolve_seed(args.train_seed, args.seed))
    if args.stage in {"all", "train"}:
        metrics = run_train(args, train_dataset, test_dataset, survival_model)
        print(json.dumps(metrics, indent=2))
    elif args.stage == "eval":
        metrics = run_eval(args, train_dataset, test_dataset, survival_model)
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
