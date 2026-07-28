#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TRITONSERVER_IMAGE="nvcr.io/nvidia/tritonserver:26.04-py3"

TASK="pp-doclayout-v3"
MODEL_NAME="pp-doclayout-v3"
VENV_NAME="vllm-v0_15_1"

MODEL_WEIGHT_DIR="/home/yrlab/models/docparse/weights/PP-DocLayoutV3_safetensors"

# Config template variables (consumed by config.pbtxt.jinja)
CONFIG_VARS=$(cat <<'JSON'
{
  "MAX_BATCH_SIZE": 8,
  "DOCLAYOUT_THRESHOLD": "0.5"
}
JSON
)

# Stage 1: render config.pbtxt from jinja template
TASK_DIR="${REPO_ROOT}/tasks/${TASK}"
TEMPLATE="${TASK_DIR}/config.pbtxt.jinja"
OUTPUT="${TASK_DIR}/config.pbtxt"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hf

python "${REPO_ROOT}/scripts/render-config.py" "${TEMPLATE}" "${OUTPUT}" --vars "${CONFIG_VARS}"

# Stage 2: launch tritonserver
docker run --rm \
    --gpus all \
    -e TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    -p 8000:8000 \
    -p 8001:8001 \
    -p 8002:8002 \
    -v "${TASK_DIR}:/models/${MODEL_NAME}" \
    -v "${MODEL_WEIGHT_DIR}:/tmp/model:ro" \
    -v "${REPO_ROOT}/venv-builder/envs/${VENV_NAME}.tar.gz:/tmp/venv.tar.gz:ro" \
    "${TRITONSERVER_IMAGE}" \
    tritonserver \
    --model-repository=/models \
    --model-control-mode=explicit \
    --load-model "${MODEL_NAME}"
