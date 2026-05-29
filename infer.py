"""Public API for inference — CLI entry point.

Usage:
    python infer.py predict --model_path checkpoints/best.pt --image_path test.png
    python infer.py onnx --model checkpoints/best.pt --onnx model.onnx
    python infer.py trt --onnx model.onnx --trt model.trt --batch 1
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import fire


def _ensure_data() -> None:
    """Pull data from DVC remote if not already present."""
    data_dir = Path(__file__).resolve().parent / "data" / "MNIST" / "raw"
    if data_dir.exists() and any(data_dir.iterdir()):
        return
    print("Pulling data from DVC remote...")
    result = subprocess.run(
        [sys.executable, "-m", "dvc", "pull"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(result.stdout)


def predict(
    model_path: str,
    image_path: str,
    device: str = "cpu",
) -> None:
    """Run inference on an image using a trained model."""
    _ensure_data()
    from digit_recognition.inference import load_model, predict_with_probs

    model = load_model(model_path, device=device)
    probs = predict_with_probs(model, image_path, device=device)
    label = int(probs.argmax(dim=1).item())
    confidence = probs[0, label].item()
    print(f"Predicted digit: {label}")
    print(f"Confidence: {confidence:.4f}")


def onnx(
    model: str,
    onnx: str,
) -> None:
    """Export model to ONNX format and verify."""
    _ensure_data()
    from scripts.convert import export_onnx, test_onnx_consistency

    export_onnx(model, onnx)
    test_onnx_consistency(model, onnx)


def trt(
    onnx: str = "model.onnx",
    trt: str = "model.trt",
    batch: int = 1,
) -> None:
    """Convert ONNX to TensorRT engine."""
    env = os.environ.copy()
    env["ONNX_PATH"] = onnx
    env["TRT_ENGINE"] = trt
    env["BATCH_SIZE"] = str(batch)
    script_path = str(Path(__file__).parent / "deploy_tensorrt.sh")
    result = subprocess.run(["bash", script_path], env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    fire.Fire()
