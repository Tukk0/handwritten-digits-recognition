#!/usr/bin/env bash
set -euo pipefail

# TensorRT deployment from ONNX model.
# Usage (env):
#   ONNX_PATH=model.onnx TRT_ENGINE=model.trt BATCH_SIZE=1 ./deploy_tensorrt.sh

ONNX_PATH="${ONNX_PATH:-${1:-model.onnx}}"
TRT_ENGINE="${TRT_ENGINE:-${2:-model.trt}}"
BATCH_SIZE="${BATCH_SIZE:-${3:-1}}"

export ONNX_PATH TRT_ENGINE BATCH_SIZE

echo "=== TensorRT deployment ==="
echo "  ONNX  : ${ONNX_PATH}"
echo "  TRT   : ${TRT_ENGINE}"
echo "  batch : ${BATCH_SIZE}"

if [[ ! -f "${ONNX_PATH}" ]]; then
    echo "ERROR: ONNX model not found: ${ONNX_PATH}"
    exit 1
fi

python -c "
import os, sys
from pathlib import Path

try:
    import tensorrt as trt
except ImportError:
    print('tensorrt not installed — skipping TensorRT build.')
    print('Run: pip install tensorrt')
    sys.exit(0)

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    print('pycuda not installed — skipping TensorRT build.')
    print('Run: pip install pycuda')
    sys.exit(0)

try:
    from polygraphy import cuda as poly_cuda
    from polygraphy.backend.onnx import OnnxModel
    from polygraphy.backend.trt import CreateConfig, engine_from_onnx, save_engine
    from polygraphy.logger import set_logging_level
except ImportError:
    print('polygraphy not installed — skipping TensorRT build.')
    print('Run: pip install polygraphy')
    sys.exit(0)

set_logging_level('ERROR')

onnx_path = os.environ['ONNX_PATH']
engine_path = os.environ['TRT_ENGINE']
batch_size = int(os.environ['BATCH_SIZE'])

onnx_bytes = Path(onnx_path).read_bytes()
print(f'Converting ONNX (batch={batch_size})')

engine = engine_from_onnx(
    onnx_bytes,
    CreateConfig(
        precision_flags=trt.PrecisionFlag.FP16,
        builder_optimization_depth=1,
        max_workspace_size=1 << 30,
        int8=False,
    ),
    min_shapes={'input': ([1, 1, 28, 28])},
    opt_shapes={'input': ([batch_size, 1, 28, 28])},
    max_shapes={'input': ([batch_size, 1, 28, 28])},
)

save_engine(engine, engine_path)
print(f'TensorRT engine saved: {engine_path}')
"

if [[ -f "${TRT_ENGINE}" ]]; then
    echo "TensorRT engine created: ${TRT_ENGINE}"
else
    echo "TensorRT may not have been built (check dependencies)."
fi

echo "=== Done ==="
