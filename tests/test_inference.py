"""Test inference module.

run: pytest tests/test_inference.py -v
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture(scope="session")
def checkpoint_path():
    """Create a tiny LeNet5 for testing."""
    from digit_recognition.model import DigitModel, LeNet5

    backbone = LeNet5(conv1_out=8, conv2_out=16, fc1_features=32, fc2_features=10, dropout_prob=0.3, batch_norm=True)
    wrapper = DigitModel(
        name="lenet5",
        optimizer_cfg=None,
        scheduler_cfg=None,
        conv1_out=8,
        conv2_out=16,
        fc1_features=32,
        fc2_features=10,
        dropout_prob=0.3,
        batch_norm=True,
    )
    wrapper.model = backbone
    wrapper.model.eval()

    tmp = Path(tempfile.mkdtemp()) / "test.pt"
    torch.save(wrapper.state_dict(), tmp)
    return str(tmp)


@pytest.fixture(scope="session")
def dummy_image():
    """Create a dummy 28x28 grayscale PNG."""
    tmp = Path(tempfile.mkdtemp()) / "dummy.png"
    arr = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
    Image.fromarray(arr).convert("L").save(tmp)
    return str(tmp)


class TestPreprocessing:
    def test_returns_correct_shape(self, dummy_image):
        from digit_recognition.inference import preprocess_image

        tensor = preprocess_image(dummy_image, device="cpu")
        assert tensor.shape == (1, 1, 28, 28)
        assert tensor.dtype == torch.float32

    def test_values_finite(self, dummy_image):
        from digit_recognition.inference import preprocess_image

        tensor = preprocess_image(dummy_image, device="cpu")
        assert torch.isfinite(tensor).all()


class TestLoadModel:
    def test_model_is_eval(self, checkpoint_path):
        from digit_recognition.inference import load_model

        model = load_model(checkpoint_path, device="cpu")
        assert model.training is False

    def test_returns_correct_structure(self, checkpoint_path):
        from digit_recognition.inference import load_model

        model = load_model(checkpoint_path, device="cpu")
        assert hasattr(model, "forward")


class TestPredict:
    def test_predict_is_int(self, checkpoint_path, dummy_image):
        from digit_recognition.inference import load_model, predict_with_probs

        model = load_model(checkpoint_path, device="cpu")
        probs = predict_with_probs(model, dummy_image, device="cpu")
        label = int(probs.argmax(dim=1).item())
        assert isinstance(label, int | np.integer)
        assert 0 <= label <= 9

    def test_probs_sum_to_one(self, checkpoint_path, dummy_image):
        from digit_recognition.inference import load_model, predict_with_probs

        model = load_model(checkpoint_path, device="cpu")
        probs = predict_with_probs(model, dummy_image, device="cpu")
        assert torch.isclose(probs.sum(), torch.ones((1,)), atol=1e-5)

    def test_probs_valid_range(self, checkpoint_path, dummy_image):
        from digit_recognition.inference import load_model, predict_with_probs

        model = load_model(checkpoint_path, device="cpu")
        probs = predict_with_probs(model, dummy_image, device="cpu")
        assert (probs >= 0).all()
        assert (probs <= 1.0 + 1e-5).all()
