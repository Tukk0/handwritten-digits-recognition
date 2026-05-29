"""Lightning callback to generate training plots and confusion matrix."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import ConfusionMatrixDisplay


class PlotsCallback(Callback):
    """Generate and save training plots after training completes."""

    def __init__(self, save_dir: str = "plots") -> None:
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._epoch: list[int] = []
        self._train_loss: list[float] = []
        self._val_loss: list[float] = []
        self._train_acc: list[float] = []
        self._val_acc: list[float] = []
        self._val_precision: list[float] = []
        self._val_recall: list[float] = []

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        epoch = trainer.current_epoch
        self._epoch.append(epoch)
        metrics = pl_module.trainer.callback_metrics
        self._train_loss.append(float(metrics.get("train_loss_epoch", 0)))
        self._val_loss.append(float(metrics.get("val_loss", 0)))
        self._train_acc.append(float(metrics.get("train_accuracy", 0)))
        self._val_acc.append(float(metrics.get("val_accuracy", 0)))
        self._val_precision.append(float(metrics.get("val_precision", 0)))
        self._val_recall.append(float(metrics.get("val_recall", 0)))

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._plot_metrics()
        self._plot_confusion_matrix(trainer, pl_module)

    def _plot_metrics(self) -> None:
        if not self._epoch:
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(self._epoch, self._train_loss, marker="o", label="Train Loss")
        axes[0].plot(self._epoch, self._val_loss, marker="o", label="Val Loss")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(self._epoch, self._train_acc, marker="o", label="Train Accuracy")
        axes[1].plot(self._epoch, self._val_acc, marker="o", label="Val Accuracy")
        axes[1].plot(self._epoch, self._val_precision, marker="o", label="Val Precision (macro)")
        axes[1].plot(self._epoch, self._val_recall, marker="o", label="Val Recall (macro)")
        axes[1].set_title("Metrics")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Score")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        path = self.save_dir / "training_plots.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Training plots saved to {path}")

    def _plot_confusion_matrix(self, trainer: Trainer, pl_module: LightningModule) -> None:
        try:
            datamodule = trainer.datamodule
            if datamodule is None:
                return
            loader = datamodule.test_dataloader()
        except Exception:
            return

        pl_module.to("cpu")
        pl_module.eval()
        all_preds: list[int] = []
        all_targets: list[int] = []

        with torch.no_grad():
            for images, labels in loader:
                logits = pl_module.model(images)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.tolist())
                all_targets.extend(labels.tolist())

        fig, ax = plt.subplots(figsize=(8, 8))
        ConfusionMatrixDisplay.from_predictions(
            all_targets,
            all_preds,
            ax=ax,
            cmap="Blues",
            colorbar=False,
        )
        ax.set_title("Test Confusion Matrix")
        plt.tight_layout()
        path = self.save_dir / "confusion_matrix.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Confusion matrix saved to {path}")
