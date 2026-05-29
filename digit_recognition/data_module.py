"""
Lightning data module for MNIST digit classification.
"""

from __future__ import annotations

import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.v2 as transforms_v2
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST


class MNISTDataModule(LightningDataModule):
    """MNIST data pipeline with train/val/test splits."""

    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 128,
        num_workers: int = 4,
        train_val_test_split: list = (0.6, 0.2, 0.2),
        image_size: tuple = (28, 28),
        mean: float = 0.1307,
        std: float = 0.3081,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        self.image_size = image_size
        self.mean = mean
        self.std = std

        self.train_transform = transforms.Compose(
            [
                transforms_v2.RandomAffine(degrees=10, translate=(0.1, 0.1)),
                transforms_v2.RandomErasing(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize((mean,), (std,)),
            ]
        )

        self.val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((mean,), (std,)),
            ]
        )

        self._train_indices: list[int] = []
        self._val_indices: list[int] = []
        self._test_indices: list[int] = []

    def setup(self, stage: str | None = None) -> None:
        self.dataset = MNIST(
            root=self.data_dir,
            train=True,
            download=True,
        )
        self._split_dataset()

    def _split_dataset(self) -> None:
        from sklearn.model_selection import StratifiedShuffleSplit

        labels = np.array([int(t.item()) for t in self.dataset.targets])
        train_pct, val_pct, _ = self.train_val_test_split
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - train_pct, random_state=42)
        train_idx, valtest_idx = next(splitter.split(np.zeros(len(labels)), labels))

        labels_valtest = labels[valtest_idx]
        splitter2 = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_pct / (1.0 - train_pct),
            random_state=42,
        )
        val_idx, test_idx = next(splitter2.split(np.zeros(len(valtest_idx)), labels_valtest))
        val_idx = valtest_idx[val_idx]
        test_idx = valtest_idx[test_idx]

        self._train_indices = sorted(train_idx)
        self._val_indices = sorted(val_idx)
        self._test_indices = sorted(test_idx)

    def train_dataloader(self) -> DataLoader:
        dataset = MNIST(
            root=self.data_dir,
            train=True,
            download=False,
            transform=self.train_transform,
        )
        return DataLoader(
            Subset(dataset, self._train_indices),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        dataset = MNIST(root=self.data_dir, train=True, download=False, transform=self.val_transform)
        return DataLoader(
            Subset(dataset, self._val_indices), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        dataset = MNIST(root=self.data_dir, train=True, download=False, transform=self.val_transform)
        return DataLoader(
            Subset(dataset, self._test_indices), batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
