"""Download MNIST data from torchvision.

Usage::

    python -m digit_recognition.download_data
"""

from __future__ import annotations

from pathlib import Path

from torchvision.datasets import MNIST


def download_data(target_dir: str = "./data") -> None:
    """Download MNIST dataset.

    Args:
        target_dir: Directory to store the dataset.
    """
    path = Path(target_dir)
    path.mkdir(parents=True, exist_ok=True)
    MNIST(root=str(path), train=True, download=True)
    MNIST(root=str(path), train=False, download=True)
    print(f"MNIST downloaded to {path}")


if __name__ == "__main__":
    download_data()
