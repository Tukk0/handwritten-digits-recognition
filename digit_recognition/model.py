"""
PyTorch Lightning module for MNIST digit classification.
Supports LeNet5 and ResNet18 (fine-tuning via config).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torchmetrics
import torchvision.models as models
from pytorch_lightning import LightningModule


class LeNet5(nn.Module):
    """Modified LeNet-5 with BatchNorm and dropout."""

    def __init__(
        self,
        conv1_out: int = 32,
        conv2_out: int = 64,
        fc1_features: int = 128,
        fc2_features: int = 10,
        dropout_prob: float = 0.5,
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, conv1_out, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(conv1_out) if batch_norm else nn.Identity()
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(conv2_out) if batch_norm else nn.Identity()
        self.fc1 = nn.Linear(conv2_out * 7 * 7, fc1_features)
        self.fc2 = nn.Linear(fc1_features, fc2_features)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class LeNet5WithResNet18(nn.Module):
    """Wraps a ResNet18 for fine-tuning on MNIST 1-channel input."""

    def __init__(self, num_classes: int = 10, freeze_backbone: bool = True) -> None:
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)


def build_model(name: str = "lenet5", **build_kwargs) -> nn.Module:
    """Factory function based on config name."""
    if name == "resnet18":
        return LeNet5WithResNet18(num_classes=build_kwargs.get("num_classes", 10))
    return LeNet5(**build_kwargs)


class DigitModel(LightningModule):
    """Full training loop for digit classification."""

    def __init__(
        self,
        name: str = "lenet5",
        optimizer_cfg: dict | None = None,
        scheduler_cfg: dict | None = None,
        **model_kwargs,
    ) -> None:
        super().__init__()
        self.model = build_model(name=name, **model_kwargs)
        self.save_hyperparameters(ignore=["optimizer_cfg", "scheduler_cfg"])
        self.loss_fn = nn.CrossEntropyLoss()
        self._optimizer_cfg = optimizer_cfg or {"name": "AdamW", "lr": 0.001, "weight_decay": 0.01}
        self._scheduler_cfg = scheduler_cfg or {
            "name": "CosineAnnealingLR",
            "T_max": 30,
            "eta_min": 1e-6,
        }
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.val_precision = torchmetrics.Precision(task="multiclass", num_classes=10, average="macro")
        self.val_recall = torchmetrics.Recall(task="multiclass", num_classes=10, average="macro")
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)

    def forward(self, x: torch.Tensor) -> dict:
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return {"logits": logits, "probs": probs, "predictions": preds}

    def training_step(self, batch, batch_idx):
        images, labels = batch
        out = self(images)
        loss = self.loss_fn(out["logits"], labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.train_accuracy(out["predictions"], labels)
        self.log("train_accuracy", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        out = self(images)
        loss = self.loss_fn(out["logits"], labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_accuracy(out["predictions"], labels)
        self.log("val_accuracy", self.val_accuracy)
        self.val_precision(out["predictions"], labels)
        self.log("val_precision", self.val_precision)
        self.val_recall(out["predictions"], labels)
        self.log("val_recall", self.val_recall)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        out = self(images)
        loss = self.loss_fn(out["logits"], labels)
        self.log("test_loss", loss, on_epoch=True)
        self.test_accuracy(out["predictions"], labels)
        self.log("test_accuracy", self.test_accuracy)
        return loss

    def configure_optimizers(self):
        weight_decay = self._optimizer_cfg.get("weight_decay", 1e-2)
        lr = self._optimizer_cfg.get("lr", 1e-3)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        t_max = self._scheduler_cfg.get("T_max", 30)
        eta_min = self._scheduler_cfg.get("eta_min", 1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
