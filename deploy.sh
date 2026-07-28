#!/bin/bash
# Stage deployment spec(s) into model_repository/ for tritonserver.
#
# Usage:
#   ./deploy.sh deployments/example.yaml [more.yaml ...] [--force] [--copy-mode link|copy|symlink]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
[ -f "${REPO_ROOT}/.env" ] && source "${REPO_ROOT}/.env"

CONDA_ENV="${CONDA_ENV:-hf}"

if [ "$#" -eq 0 ]; then
    echo "usage: $0 <deployment.yaml> [...] [--force] [--copy-mode link|copy|symlink]" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

REPO_ARGS=()
if [ -n "${MODEL_REPOSITORY:-}" ]; then
    REPO_ARGS=(--model-repository "${MODEL_REPOSITORY}")
fi

python "${REPO_ROOT}/scripts/deploy.py" "${REPO_ARGS[@]}" "$@"
