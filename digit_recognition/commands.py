"""Unified CLI entry points for the digit-recognition package.

Usage::

    python -m digit_recognition.commands train [epochs]
    python -m digit_recognition.commands onnx
    python -m digit_recognition.commands test
    python -m digit_recognition.commands lint
"""

from __future__ import annotations

import glob
import subprocess
import sys
from pathlib import Path

import fire

_ROOT = Path(__file__).resolve().parent.parent


def train(epochs: int = 5) -> None:
    """Train the digit recognition model.

    Args:
        epochs: Number of training epochs (default: 5).
    """
    cmd = [
        sys.executable,
        str(_ROOT / "train_main.py"),
        f"training.max_epochs={epochs}",
    ]
    result = subprocess.run(cmd, cwd=str(_ROOT))
    sys.exit(result.returncode)


def onnx(checkpoint: str | None = None) -> None:
    """Export the trained model to ONNX and verify consistency.

    Args:
        checkpoint: Path to checkpoint file. Defaults to latest .ckpt in checkpoints/.
    """
    if checkpoint is None:
        ckpts = sorted(glob.glob(str(_ROOT / "checkpoints" / "*.ckpt")))
        if not ckpts:
            print("No checkpoints found. Run 'train' first (via docker or train command).")
            return
        checkpoint = ckpts[-1]

    onnx_path = str(_ROOT / "model.onnx")
    cmd = [
        sys.executable,
        str(_ROOT / "scripts" / "convert.py"),
        "export_onnx",
        checkpoint,
        onnx_path,
    ]
    result = subprocess.run(cmd, cwd=str(_ROOT))
    if result.returncode != 0:
        sys.exit(result.returncode)

    cmd_verify = [
        sys.executable,
        str(_ROOT / "scripts" / "convert.py"),
        "test_onnx_consistency",
        checkpoint,
        onnx_path,
    ]
    result = subprocess.run(cmd_verify, cwd=str(_ROOT))
    sys.exit(result.returncode)


def test() -> None:
    """Run the pytest test suite."""
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], cwd=str(_ROOT))
    sys.exit(result.returncode)


def lint() -> None:
    """Run ruff linter over the project."""
    result = subprocess.run([sys.executable, "-m", "ruff", "check", "."], cwd=str(_ROOT))
    sys.exit(result.returncode)


def cli(*args, **kwargs) -> None:
    """Main CLI — redirects to appropriate sub-command."""
    fire.Fire(
        {
            "train": train,
            "onnx": onnx,
            "test": test,
            "lint": lint,
        },
        name="",
    )


if __name__ == "__main__":
    cli()
