#!/bin/bash
set -euo pipefail

# Quick project scripts:
#   ./run.sh           - Full pipeline (train -> onnx -> verify)
#   ./run.sh train 10  - Train model (10 epochs)
#   ./run.sh onnx      - Export last checkpoint to ONNX + verify
#   ./run.sh test      - Run pytest
#   ./run.sh lint      - Run ruff check
#   ./run.sh docker    - Build & run via docker
#   ./run.sh docker-all - Full pipeline via docker-compose

case "${1:-run}" in
    run)
        echo "=== Running full MLOps pipeline ==="
        echo "Step 1/3: Training MNIST model (5 epochs)..."
        python train_main.py training.max_epochs=5
        echo "Step 2/3: Exporting to ONNX..."
        LATEST=$(ls -t checkpoints/*.ckpt 2>/dev/null | head -1)
        if [ -z "$LATEST" ]; then
            echo "No checkpoints found. Run ./run.sh train first or use docker."
            exit 1
        fi
        cp -- "$LATEST" checkpoints/best.pt
        python scripts/convert.py export_onnx checkpoints/best.pt model.onnx
        python scripts/convert.py test_onnx_consistency checkpoints/best.pt model.onnx
        echo "=== Pipeline complete ==="
        echo ""
        echo "Next steps:"
        echo "  python infer.py predict checkpoints/best.pt your_image.png"
        echo "  python infer.py onnx --model checkpoints/best.pt --onnx model.onnx"
        ;;
    train)
        EPOCHS="${2:-5}"
        echo "Training MNIST model for $EPOCHS epochs..."
        python train_main.py training.max_epochs="$EPOCHS"
        ;;
    onnx)
        echo "Exporting latest checkpoint to ONNX..."
        if [ ! -d "checkpoints" ]; then
            echo "No checkpoints found. Run ./run.sh train first."
            exit 1
        fi
        LATEST=$(ls -t checkpoints/*.ckpt 2>/dev/null | head -1)
        cp -- "$LATEST" checkpoints/best.pt
        python scripts/convert.py export_onnx "checkpoints/best.pt" "model.onnx"
        python scripts/convert.py test_onnx_consistency "checkpoints/best.pt" "model.onnx"
        ;;
    inference)
        MODEL_PATH="${2:-checkpoints/best.pt}"
        if [ ! -f "$MODEL_PATH" ]; then
            echo "Model not found at $MODEL_PATH. Run ./run.sh train first."
            exit 1
        fi
        echo "Ready for inference."
        echo "Use: python infer.py predict $MODEL_PATH your_image.png"
        ;;
    test)
        echo "Running tests..."
        pytest tests/ -v
        ;;
    lint)
        echo "Running linting..."
        ruff check .
        ;;
    docker)
        echo "Building Docker image..."
        docker build -t digit-recognition .
        echo ""
        echo "Usage: docker run --rm digit-recognition [train|test|lint|onnx|inference|bot|train-onnx-inference] [args...]"
        echo "  docker run --rm digit-recognition train 10  # 10 epochs"
        echo "  docker run --rm digit-recognition test      # run tests"
        echo "  docker run --rm digit-recognition lint      # ruff check"
        echo "  docker run --rm digit-recognition onnx      # export to ONNX"
        ;;
    docker-all)
        echo "Running full pipeline via Docker..."
        docker build -t digit-recognition .
        docker run --rm digit-recognition train-onnx-inference
        ;;
    docker-compose)
        echo "Starting docker-compose services..."
        docker compose up --build -d
        ;;
    *)
        echo "Usage: ./run.sh [command] [args]"
        echo ""
        echo "Commands:"
        echo "  train [N]              Train model for N epochs (default: 5)"
        echo "  onnx                   Export last checkpoint to ONNX + verify"
        echo "  inference [MODEL]      Show inference ready (run inference separately)"
        echo "  test                   Run pytest"
        echo "  lint                   Run ruff check"
        echo "  docker                 Build Docker image and show usage"
        echo "  docker-all             Full pipeline: train -> onnx -> convert via Docker"
        echo "  docker-compose         Start Triton + Telegram bot via docker compose"
        exit 1
        ;;
esac
