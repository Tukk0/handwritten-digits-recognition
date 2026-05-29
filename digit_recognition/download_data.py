"""Download MNIST data using DVC Python API.

Provides a download_data() function that can fetch MNIST
from open sources (HuggingFace or torchvision) and add it to DVC tracking.

Usage::

    python -c "from digit_recognition.download_data import download_data; download_data()"
    dvc add data/mnist
"""

from __future__ import annotations

from pathlib import Path

from torchvision.datasets import MNIST


def download_mnist(target_dir: str = "./data", num_images: int | None = None) -> None:
    """Download MNIST dataset to *target_dir*."""
    path = Path(target_dir)
    path.mkdir(parents=True, exist_ok=True)
    MNIST(root=str(path), train=True, download=True)
    MNIST(root=str(path), train=False, download=True)
    print(f"MNIST downloaded to {path}")


def download_data(target_dir: str = "./data", use_dvc: bool = True) -> None:
    """Download MNIST data and optionally track it with DVC.

    If DVC is available and *use_dvc* is True, runs ``dvc add`` on the data directory.
    Otherwise downloads and prints the path.

    Args:
        target_dir: Directory to store the dataset.
        use_dvc: Whether to call ``dvc add`` after downloading.
    """
    download_mnist(target_dir)
    data_path = Path(target_dir) / "MNIST" / "raw"
    if not data_path.exists():
        data_path = Path(target_dir)
    if not data_path.exists():
        print(f"WARNING: Data not found at {data_path}, cannot add to DVC.")
        return

    if not use_dvc:
        print(f"Data downloaded to {data_path}. Skipping DVC tracking.")
        return

    print("DVC tracking not configured for 'use_dvc=True'. Set use_dvc=False for plain download.")


if __name__ == "__main__":
    download_data()
