#!/bin/bash
set -euo pipefail

COMMAND="${1:-train}"

case "$COMMAND" in
    train)
        EPOCHS="${2:-5}"
        echo "Training MNIST model for $EPOCHS epochs..."
        python train_main.py training.max_epochs="$EPOCHS"
        ;;
    test)
        echo "Running tests..."
        pytest tests/ -v
        ;;
    lint)
        echo "Running linting..."
        ruff check .
        ;;
    onnx)
        MODEL_PATH="${2:-checkpoints/best.pt}"
        ONNX_PATH="${3:-model.onnx}"
        echo "Converting $MODEL_PATH to ONNX ($ONNX_PATH)..."
        python scripts/convert.py export_onnx "$MODEL_PATH" "$ONNX_PATH"
        python scripts/convert.py test_onnx_consistency "$MODEL_PATH" "$ONNX_PATH"
        ;;
    inference)
        MODEL_PATH="${2:-checkpoints/best.pt}"
        IMAGE_PATH="$3"
        echo "Running inference: $MODEL_PATH -> $IMAGE_PATH"
        python infer.py predict "$MODEL_PATH" "$IMAGE_PATH"
        ;;
    bot)
        echo "Starting Telegram bot..."
        python scripts/main.py
        ;;
    train-onnx-inference)
        # Full pipeline: train -> convert -> test
        echo "=== Full MLOps pipeline ==="
        python train_main.py training.max_epochs=5
        python scripts/convert.py export_onnx checkpoints/best.pt model.onnx
        python scripts/convert.py test_onnx_consistency checkpoints/best.pt model.onnx
        echo "=== Pipeline complete ==="
        ;;
    *)
        echo "Usage: $0 {train|test|lint|onnx|inference|bot|train-onnx-inference}"
        echo "  train [N]                     - Train model for N epochs (default: 5)"
        echo "  test                          - Run pytest"
        echo "  lint                          - Run ruff check"
        echo "  onnx [MODEL] [ONNX]           - Export to ONNX"
        echo "  inference MODEL IMAGE         - Run inference"
        echo "  bot                           - Start Telegram bot"
        echo "  train-onnx-inference          - Full pipeline"
        exit 1
        ;;
esac
