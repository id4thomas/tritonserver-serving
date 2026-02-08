#!/bin/bash
set -e

TRITONSERVER_IMAGE="nvcr.io/nvidia/tritonserver:25.06-py3"
# TRITONSERVER_IMAGE="nvcr.io/nvidia/tritonserver:26.01-py3"

TASK="text-clf/vllm"
MODEL_NAME="text-classification"
# VENV_NAME="text-clf-vllm-v1"
VENV_NAME="vllm-v0_11_0"
CONFIG_NAME="v1-qwen3-emb-dgx-spark"

# Qwen3-Embedding-0.6B with classificationhead
MODEL_WEIGHT_DIR="model/weights/clf-qwen3"

docker run --rm \
    --gpus all \
    -e TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    -p 8000:8000 \
    -p 8001:8001 \
    -p 8002:8002 \
    -v "$(pwd)/tasks/${TASK}:/models/${MODEL_NAME}" \
    -v "$(pwd)/${MODEL_WEIGHT_DIR}:/tmp/model:ro" \
    -v "$(pwd)/venv-builder/envs/${VENV_NAME}.tar.gz:/tmp/venv.tar.gz:ro" \
    ${TRITONSERVER_IMAGE} \
    tritonserver \
    --model-repository=/models \
    --model-control-mode=explicit \
    --load-model ${MODEL_NAME} \
    --model-config-name ${CONFIG_NAME}