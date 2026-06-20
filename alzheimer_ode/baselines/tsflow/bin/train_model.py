import argparse
import logging
import sys
import tempfile
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from torch.utils.data import DataLoader
try:
    from aim.pytorch_lightning import AimLogger
except ImportError:
    AimLogger = None
from pytorch_lightning.loggers import CSVLogger
from gluonts.dataset.loader import TrainDataLoader
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.dataset.split import OffsetSplitter
from gluonts.evaluation import (
    Evaluator,
    MultivariateEvaluator,
    make_evaluation_predictions,
)
from gluonts.itertools import Cached
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.torch.batchify import batchify
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm.auto import tqdm

from tsflow.callback import EvaluateCallback
from tsflow.dataset import get_gts_dataset
from tsflow.model import TSFlowCond
from tsflow.utils import (
    create_multivariate_transforms,
    create_transforms,
    create_atn_transforms,
)
from tsflow.utils.util import ConcatDataset, add_config_to_argparser, create_splitter, filter_metrics
from tsflow.data_custom import build_atn_datasets, ATNTorchDataset
from tsflow.data_custom import ATNTrainDataset, atn_train_collate
from tsflow.metrics.atn_evaluator import ATNEvaluator
# from tsflow.utils.util import create_atn_splitter
from tsflow.utils.variables import get_season_length

try:
    import pykeops
except ImportError:
    pykeops = None

if pykeops is not None:
    temp_build_folder = tempfile.mkdtemp(prefix="pykeops_build_")
    pykeops.set_build_folder(temp_build_folder)
    pykeops.clean_pykeops()


def create_model(setting, target_dim, model_params):
    model = TSFlowCond(
        setting=setting,
        target_dim=target_dim,
        context_length=model_params["context_length"],
        prediction_length=model_params["prediction_length"],
        backbone_params=model_params["backbone_params"],
        prior_params=model_params["prior_params"],
        optimizer_params=model_params["optimizer_params"],
        ema_params=model_params["ema_params"],
        frequency=model_params["freq"],
        normalization=model_params["normalization"],
        use_lags=model_params["use_lags"],
        use_ema=model_params["use_ema"],
        num_steps=model_params["num_steps"],
        solver=model_params["solver"],
        matching=model_params["matching"],
        multi_series_univariate=model_params.get("multi_series_univariate", False),
        static_feature_dim=model_params.get("static_feature_dim", 0),
    )
    model.to(model_params["device"])
    return model


def evaluate_conditional(
    model_params,
    model: TSFlowCond,
    test_dataset,
    transformation,
    trainer,
    num_samples=100,
):
    logging.info(f"Evaluating with {num_samples} samples.")
    results = {}

    transformed_testdata = transformation.apply(test_dataset, is_train=False)
    test_splitter = create_splitter(
        past_length=max(
            model_params["context_length"] + max(model.lags_seq),
            model.prior_context_length,
        ),
        future_length=model_params["prediction_length"],
        mode="test",
    )

    test_transform = test_splitter
    if model.setting == "univariate":
        batch_size = 1024 * 64 // num_samples
        evaluator = Evaluator(num_workers=1)
    elif model.setting == "multivariate":
        batch_size = 1
        evaluator = MultivariateEvaluator(target_agg_funcs={"sum": np.sum})
    predictor = model.get_predictor(
        test_transform,
        batch_size=batch_size,
        device=model_params["device"],
    )
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=transformed_testdata,
        predictor=predictor,
        num_samples=num_samples,
    )
    forecasts = list(tqdm(forecast_it, total=len(transformed_testdata)))
    tss = list(ts_it)
    metrics, _ = evaluator(tss, forecasts)
    metrics["CRPS"] = metrics["mean_wQuantileLoss"]
    select = ["CRPS", "ND", "NRMSE"]
    if model.setting == "multivariate":
        metrics["m_sum_CRPS"] = metrics["m_sum_mean_wQuantileLoss"]
        select = select + ["m_sum_CRPS"]
    metrics = filter_metrics(metrics, select)
    [
        logger.log_metrics(
            {f"test_{key}": val for key, val in metrics.items()},
            step=trainer.global_step + 1,
        )
        for logger in trainer.loggers
    ]
    results["test"] = dict(**metrics)
    return results


def evaluate_atn(
    model_params,
    model: TSFlowCond,
    test_samples,
    trainer,
    num_samples=1,
):
    device = model_params["device"]
    batch_size = model_params.get("eval_batch_size", 32)
    evaluator = ATNEvaluator()

    test_ds = ATNTorchDataset(test_samples)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_gen_points = []
    all_true_points = []
    all_gen_trajs = []
    all_true_trajs = []

    model.num_samples = num_samples
    _t_eval_start = time.time()
    model.eval()
    with torch.no_grad():
        for batch_x, batch_mask, batch_age, batch_label in test_loader:
            # batch_x: [B, K, T]
            batch_x = batch_x.to(device)
            batch_mask = batch_mask.to(device)
            batch_age = batch_age.to(device)
            batch_label = batch_label.to(device)

            batch_x = batch_x.transpose(1, 2)  # [B, T, K]
            B, T, K = batch_x.shape

            past_target = torch.zeros((B, 0, K), device=device)
            past_observed_values = torch.zeros((B, 0, K), device=device)
            mean = torch.zeros((B, 1, K), device=device)

            feat_static_real = batch_age.unsqueeze(-1)
            feat_static_cat = batch_label.unsqueeze(-1)

            sample_paths = model.forward_atn(
                past_target=past_target,
                past_observed_values=past_observed_values,
                mean=mean,
                feat_static_real=feat_static_real,
                feat_static_cat=feat_static_cat,
            )
            sample_mean = sample_paths.mean(dim=1).cpu()
            batch_x = batch_x.cpu()
            batch_mask = batch_mask.cpu()

            for i in range(B):
                valid = batch_mask[i] > 0
                if valid.sum() == 0:
                    continue
                sample_i = sample_mean[i][valid]
                true_i = batch_x[i][valid]
                all_gen_trajs.append(sample_i)
                all_true_trajs.append(true_i)
                all_gen_points.append(sample_i)
                all_true_points.append(true_i)

    if len(all_gen_points) == 0:
        return {"test": {}}

    all_gen_tensor = torch.cat(all_gen_points, dim=0)
    all_true_tensor = torch.cat(all_true_points, dim=0)

    dist_metrics = evaluator.metrics(all_gen_tensor, all_true_tensor)
    traj_metrics = evaluator.trajectory_metrics(all_gen_trajs, all_true_trajs)

    metrics = {**dist_metrics, **traj_metrics}
    metrics["TIME_SEC"] = float(time.time() - _t_eval_start)
    metrics = {k: (float(v) if v is not None else v) for k, v in metrics.items()}
    if trainer is not None:
        [
            logger.log_metrics(
                {f"test_{key}": val for key, val in metrics.items()},
                step=trainer.global_step + 1,
            )
            for logger in trainer.loggers
        ]
    return {"test": metrics}


def main(
    model,
    setting,
    model_params,
    dataset_params,
    trainer_params,
    evaluation_params,
    eval_only=False,
    eval_ckpt_path=None,
    logdir=None,
    seed=None,
    config=None,
    loggers=[],
):
    if logdir:
        Path(logdir).mkdir(parents=True, exist_ok=True)
    pl.seed_everything(seed)
    # Load parameters
    dataset_name = dataset_params["dataset"]
    freq = model_params["freq"]
    prediction_length = model_params["prediction_length"]

    if dataset_name == "atn_custom":
        _t0 = time.time()
        dataset, train_samples, test_samples = build_atn_datasets(
            train_path=dataset_params["train_path"],
            test_path=dataset_params["test_path"],
            json_file=dataset_params["json_file"],
            prediction_length=prediction_length,
            freq=freq,
        )
        logging.info(f"ATN build time: {time.time() - _t0:.2f}s, train={len(train_samples)}, test={len(test_samples)}")
        target_dim = dataset_params.get("target_dim", 444)
        model = create_model(setting, target_dim, model_params)

        num_rolling_evals = 1
        # _t1 = time.time()
        # transformation = create_atn_transforms(
        #     prediction_length=model_params["prediction_length"],
        # )
        # training_data = dataset.train
        # test_data = dataset.test
        transformation = None
        training_data = None
        test_data = None
    else:
        dataset = get_gts_dataset(dataset_name)
        target_dim = min(2000, int(dataset.metadata.feat_static_cat[0].cardinality))
        # Create model
        model = create_model(setting, target_dim, model_params)

        # Setup dataset and data loading
        assert dataset.metadata.freq == freq
        assert dataset.metadata.prediction_length == prediction_length

        num_rolling_evals = int(len(dataset.test) / len(dataset.train))
        time_features = time_features_from_frequency_str(freq)
        if setting == "univariate":
            transformation = create_transforms(
                time_features=time_features,
                prediction_length=model_params["prediction_length"],
                freq=get_season_length(freq),
                train_length=len(dataset.train),
            )
            training_data = dataset.train
            test_data = dataset.test
        elif setting == "multivariate":
            train_grouper = MultivariateGrouper(max_target_dim=target_dim)
            test_grouper = MultivariateGrouper(
                num_test_dates=num_rolling_evals,
                max_target_dim=target_dim,
            )
            transformation = create_multivariate_transforms(
                time_features=time_features,
                prediction_length=model_params["prediction_length"],
                target_dim=target_dim,
                freq=get_season_length(freq),
                train_length=len(dataset.train),
            )
            training_data = train_grouper(dataset.train)
            test_data = test_grouper(dataset.test)

    if dataset_name == "atn_custom":
        training_splitter = None
    else:
        training_splitter = create_splitter(
            past_length=max(
                model_params["context_length"] + max(model.lags_seq),
                model.prior_context_length,
            ),
            future_length=model_params["prediction_length"],
            mode="train",
        )
    callbacks = []
    if evaluation_params["use_validation_set"] and dataset_name != "atn_custom":
        train_val_splitter = OffsetSplitter(offset=-model_params["prediction_length"] * num_rolling_evals)
        training_data, val_gen = train_val_splitter.split(training_data)
        _t2 = time.time()
        transformed_data = transformation.apply(training_data, is_train=True)
        logging.info(f"Transform apply time: {time.time() - _t2:.2f}s")
        val_data = val_gen.generate_instances(model_params["prediction_length"], num_rolling_evals)
        transformed_valdata = transformation.apply(ConcatDataset(val_data), is_train=False)
        callbacks = [
            EvaluateCallback(
                context_length=model_params["context_length"],
                prediction_length=model_params["prediction_length"],
                model=model,
                datasets={"val": transformed_valdata},
                logdir=logdir,
                **evaluation_params,
            )
        ]
    else:
        if dataset_name == "atn_custom":
            transformed_data = None
        else:
            _t2 = time.time()
            transformed_data = transformation.apply(training_data, is_train=True)
            logging.info(f"Transform apply time: {time.time() - _t2:.2f}s")

    log_monitor = "train_loss"
    filename = dataset_name + "-{epoch:03d}-{train_loss:.3f}"

    if dataset_name == "atn_custom":
        train_ds = ATNTrainDataset(train_samples, model_params["prediction_length"])
        data_loader = DataLoader(
            train_ds,
            batch_size=dataset_params["batch_size"],
            shuffle=True,
            num_workers=0,
            collate_fn=atn_train_collate,
        )
    else:
        shuffle_buffer_length = 0 if dataset_name == "atn_custom" else 10000
        _train_source = Cached(transformed_data)
        data_loader = TrainDataLoader(
            _train_source,
            batch_size=dataset_params["batch_size"],
            stack_fn=batchify,
            transform=training_splitter,
            num_batches_per_epoch=dataset_params["num_batches_per_epoch"],
            shuffle_buffer_length=shuffle_buffer_length,
        )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor=f"{log_monitor}",
        mode="min",
        filename=filename,
        save_last=True,
        save_weights_only=True,
    )

    callbacks.append(checkpoint_callback)
    requested_device = str(model_params["device"])
    use_gpu = requested_device.startswith("cuda") and torch.cuda.is_available()
    trainer = pl.Trainer(
        accelerator="gpu" if use_gpu else "cpu",
        devices=[int(requested_device.split(":")[-1])] if use_gpu and ":" in requested_device else 1,
        default_root_dir=logdir,
        logger=loggers,
        enable_progress_bar=False,
        callbacks=callbacks,
        **trainer_params,
    )

    logging.info(f"Logging to {logdir}")
    if eval_only:
        ckpt_path = Path(eval_ckpt_path) if eval_ckpt_path else Path(logdir) / "best_checkpoint.ckpt"
        if not ckpt_path.exists():
            raise RuntimeError(f"Eval-only requested but checkpoint not found: {ckpt_path}")
        logging.info(f"Loading {ckpt_path}.")
        best_state_dict = torch.load(ckpt_path)
        if isinstance(best_state_dict, dict) and "state_dict" in best_state_dict:
            best_state_dict = best_state_dict["state_dict"]
        model.load_state_dict(best_state_dict, strict=True)
    else:
        trainer.fit(model, train_dataloaders=data_loader)
        logging.info("Training completed.")

        best_ckpt_path = Path(logdir) / "best_checkpoint.ckpt"
        if not best_ckpt_path.exists():
            torch.save(
                torch.load(checkpoint_callback.best_model_path)["state_dict"],
                best_ckpt_path,
            )
        logging.info(f"Loading {best_ckpt_path}.")
        best_state_dict = torch.load(best_ckpt_path)
        model.load_state_dict(best_state_dict, strict=True)

    if evaluation_params.get("do_final_eval", True):
        if dataset_name == "atn_custom":
            metrics = evaluate_atn(
                model_params,
                model,
                test_samples,
                trainer,
                num_samples=evaluation_params.get("num_samples", 1),
            )
        else:
            metrics = evaluate_conditional(model_params, model, test_data, transformation, trainer)
    else:
        metrics = "Final eval not performed"
    # metrics = (
    #     evaluate_conditional(model_params, model, test_data, transformation, trainer)
    #     if evaluation_params.get("do_final_eval", True)
    #     else "Final eval not performed"
    # )

    with open(Path(logdir) / "results.yaml", "w") as fp:
        yaml.dump(
            {
                "config": config,
                "version": trainer.logger.version,
                "metrics": metrics,
            },
            fp,
        )
    return metrics


if __name__ == "__main__":
    # Setup Logger
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Setup argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to yaml config")
    parser.add_argument("--logdir", type=str, default="./logs", help="Path to results dir")
    args, _ = parser.parse_known_args()

    with open(args.config, "r") as fp:
        config = yaml.safe_load(fp)

    # Update config from command line
    parser = add_config_to_argparser(config=config, parser=parser)
    parser.add_argument("--eval-only", action="store_true", default=False)
    parser.add_argument("--eval-ckpt-path", type=str, default="")
    args = parser.parse_args()
    config_updates = vars(args)
    for k in config.keys() & config_updates.keys():
        orig_val = config[k]
        updated_val = config_updates[k]
        if updated_val != orig_val:
            logging.info(f"Updated key '{k}': {orig_val} -> {updated_val}")
    config.update(config_updates)
    if AimLogger is not None:
        logger = AimLogger()
    else:
        logger = CSVLogger(save_dir=args.logdir, name="tsflow")
    logger.log_hyperparams(config)
    config["logdir"] = str(Path(args.logdir) / str(logger.version))
    main(**config, loggers=[logger])
