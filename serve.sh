#!/bin/bash
# Mount model_repository/ read-only and launch tritonserver.
#
# Usage:
#   ./serve.sh                                   # load every model in model_repository/
#   ./serve.sh example                           # load by model name
#   ./serve.sh deployments/example.yaml [...]    # load by deployment spec(s)
#
# Settings come from .env (see .env.example); environment variables override it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
[ -f "${REPO_ROOT}/.env" ] && source "${REPO_ROOT}/.env"

TRITONSERVER_IMAGE="${TRITONSERVER_IMAGE:-nvcr.io/nvidia/tritonserver:26.06-py3}"
MODEL_REPOSITORY="${MODEL_REPOSITORY:-${REPO_ROOT}/model_repository}"
HTTP_PORT="${HTTP_PORT:-8000}"
GRPC_PORT="${GRPC_PORT:-8001}"
METRICS_PORT="${METRICS_PORT:-8002}"
GPUS="${GPUS:-all}"
SHM_SIZE="${SHM_SIZE:-8g}"

if [ ! -d "${MODEL_REPOSITORY}" ]; then
    echo "model repository not found: ${MODEL_REPOSITORY} (run ./deploy.sh first)" >&2
    exit 1
fi

# Resolve arguments to model names: a *.yaml is read as a deployment spec,
# anything else is taken as a model name directly.
MODELS=()
for arg in "$@"; do
    case "${arg}" in
        *.yaml|*.yml)
            name="$(sed -n 's/^model_name:[[:space:]]*//p' "${arg}" | head -1 | tr -d "\"' ")"
            if [ -z "${name}" ]; then
                echo "no model_name in ${arg}" >&2
                exit 1
            fi
            MODELS+=("${name}")
            ;;
        *)
            MODELS+=("${arg}")
            ;;
    esac
done

# No models given: load everything staged in the repository
if [ "${#MODELS[@]}" -eq 0 ]; then
    while IFS= read -r dir; do
        MODELS+=("$(basename "${dir}")")
    done < <(find "${MODEL_REPOSITORY}" -mindepth 1 -maxdepth 1 -type d | sort)
fi

if [ "${#MODELS[@]}" -eq 0 ]; then
    echo "no models found in ${MODEL_REPOSITORY} (run ./deploy.sh first)" >&2
    exit 1
fi

LOAD_ARGS=()
for model in "${MODELS[@]}"; do
    if [ ! -d "${MODEL_REPOSITORY}/${model}" ]; then
        echo "model not staged: ${MODEL_REPOSITORY}/${model} (run ./deploy.sh first)" >&2
        exit 1
    fi
    LOAD_ARGS+=(--load-model "${model}")
done

echo "serving: ${MODELS[*]}"

# Ask for a TTY only when there is one, so the script also works under nohup / CI.
TTY_ARGS=()
[ -t 0 ] && TTY_ARGS=(-it)

# The repository is mounted read-only: tritonserver runs as root in the container and would
# otherwise leave root-owned files (__pycache__) in the host tree. The python backend unpacks
# EXECUTION_ENV_PATH into /tmp inside the container, so it never needs to write to /models.
exec docker run --rm \
    "${TTY_ARGS[@]}" \
    --gpus "${GPUS}" \
    --shm-size "${SHM_SIZE}" \
    -e TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -p "${HTTP_PORT}:8000" \
    -p "${GRPC_PORT}:8001" \
    -p "${METRICS_PORT}:8002" \
    -v "${MODEL_REPOSITORY}:/models:ro" \
    "${TRITONSERVER_IMAGE}" \
    tritonserver \
    --model-repository=/models \
    --model-control-mode=explicit \
    "${LOAD_ARGS[@]}"
