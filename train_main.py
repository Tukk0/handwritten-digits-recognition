"""Entrypoint for training - placed at project root for Hydra compatibility."""

from __future__ import annotations

import sys
from pathlib import Path

import git
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger, TensorBoardLogger

_ROOT = Path(__file__).resolve().parent


def _get_git_commit() -> str:
    try:
        repo = git.Repo(search_parent_directories=True)
        return repo.head.object.hexsha
    except git.InvalidGitRepositoryError:
        return "unknown"


def _load_config(overrides: list[str]) -> DictConfig:
    """Load Hydra config with CLI overrides."""
    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="configs", job_name="train"):
        return compose(config_name="train", overrides=overrides)


def _parse_cli_overrides() -> list[str]:
    """Extract Hydra-style overrides from sys.argv (skip script name)."""
    return [arg for arg in sys.argv[1:] if "=" in arg and not arg.startswith("-")]


def train(cfg: DictConfig) -> None:
    """Training loop using PyTorch Lightning with full logging."""
    git_sha = _get_git_commit()
    print(f"Training started — git commit: {git_sha}")

    log_dir = str(_ROOT / "logs")

    params: dict[str, str | int | float] = {
        "optimizer_name": str(cfg.training.optimizer.name),
        "optimizer_lr": cfg.training.optimizer.lr,
        "optimizer_weight_decay": cfg.training.optimizer.weight_decay,
        "scheduler_name": str(cfg.training.scheduler.name),
        "max_epochs": cfg.training.max_epochs,
        "gradient_clip_val": cfg.training.gradient_clip_val,
        "devices": str(cfg.training.devices),
        "batch_size": cfg.data.batch_size,
        "num_workers": cfg.data.num_workers,
        "git_commit": git_sha,
    }

    mlflow_logger: MLFlowLogger | None = None
    try:
        import mlflow

        mlflow_logger = MLFlowLogger(
            experiment_name=cfg.logging.mlflow_experiment_name,
            save_dir=str(_ROOT / "mlruns"),
            prefix="",
        )
        with mlflow.start_run(run_id=mlflow_logger.run_id):
            for key, value in params.items():
                mlflow.log_param(key, value)
        del mlflow
    except Exception as error:
        print(f"MLFlow logging failed ({error}) — using TensorBoard-only")

    tb_logger = TensorBoardLogger(save_dir=log_dir, name="", version=0)

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=cfg.training.early_stopping.patience,
        mode="min",
        verbose=True,
    )
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(_ROOT / "checkpoints"),
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch}-{val_loss:.2f}",
    )

    logger_list = [tb_logger] if mlflow_logger is None else [mlflow_logger, tb_logger]

    trainer = Trainer(
        max_epochs=cfg.training.max_epochs,
        devices=cfg.training.devices,
        accelerator=cfg.training.accelerator,
        logger=logger_list,
        callbacks=[early_stop, checkpoint_cb],
        gradient_clip_val=cfg.training.gradient_clip_val,
        precision=cfg.training.precision,
        default_root_dir=log_dir,
    )

    from digit_recognition.data_module import MNISTDataModule
    from digit_recognition.model import DigitModel

    data_module = MNISTDataModule(
        data_dir=str(_ROOT / "data"),
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        image_size=tuple(cfg.data.image_size),
        mean=cfg.data.mean,
        std=cfg.data.std,
    )

    model = DigitModel(
        name=cfg.model.name,
        optimizer_cfg=cfg.training.optimizer,
        scheduler_cfg=cfg.training.scheduler,
        conv1_out=cfg.model.layers[0].out_channels,
        conv2_out=cfg.model.layers[1].out_channels,
        fc1_features=cfg.model.layers[2].out_features,
        fc2_features=cfg.model.layers[3].out_features,
        dropout_prob=cfg.model.dropout_prob,
        batch_norm=cfg.model.batch_norm,
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)

    print(f"Training complete. Logs saved to: {tb_logger.log_dir}")
    print(f"Metrics logged to TensorBoard: {tb_logger.log_dir}")


def main() -> None:
    """Entry point — loads config from CLI args and starts training."""
    config_overrides = _parse_cli_overrides()
    cfg = _load_config(config_overrides) if config_overrides else _load_config([])
    train(cfg)


if __name__ == "__main__":
    main()
