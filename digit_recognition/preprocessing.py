"""
Preprocessing utilities for handwritten digit recognition.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch


def resize_image(
    image: np.ndarray,
    target_size: tuple[int, int] = (28, 28),
) -> np.ndarray:
    """Resize a single-channel image to `target_size`.

    Returns a float32 array with values in [0, 1].
    """
    resized = cv2.resize(image, dsize=target_size, interpolation=cv2.INTER_AREA)
    if resized.dtype != np.float32:
        resized = resized.astype(np.float32)
    # normalise to [0, 1]
    if resized.max() > 1.0:
        resized = resized / 255.0
    return resized


def normalize_grayscale(image: np.ndarray, mean: float = 0.1307, std: float = 0.3081) -> np.ndarray:
    """Apply standard MNIST normalisation."""
    return (image - mean) / std


def add_noise(image: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
    """Add Gaussian noise with zero mean and `noise_level` sigma."""
    noise = np.random.normal(0, noise_level, image.shape).astype(np.float32)
    noisy = np.clip(image + noise, 0.0, 1.0)
    return noisy


def preprocess_image(
    image: np.ndarray,
    target_size: tuple[int, int] = (28, 28),
    mean: float = 0.1307,
    std: float = 0.3081,
    augment: bool = False,
    augment_noise: float = 0.05,
) -> torch.Tensor:
    """Full preprocessing pipeline: resize → optional noise → normalise → tensor.

    Parameters
    ----------
    image : input BGR or grayscale Numpy array.
    target_size : desired (W, H).
    mean / std : MNIST normalisation constants.
    augment : add training-time augmentations.
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = resize_image(image, target_size=target_size)
    if augment:
        image = add_noise(image, noise_level=augment_noise)
    image = normalize_grayscale(image, mean=mean, std=std)
    # CHW
    tensor = torch.from_numpy(image).unsqueeze(0).float()
    return tensor
