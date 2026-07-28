#!/bin/bash
# Stage deployment spec(s) into model_repository/ for tritonserver.
#
# Usage:
#   ./deploy.sh deployments/example.yaml [more.yaml ...] [--force]
#
# Runs with whatever python environment is currently active (needs pyyaml + jinja2).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck disable=SC1091
[ -f "${REPO_ROOT}/.env" ] && source "${REPO_ROOT}/.env"

if [ "$#" -eq 0 ]; then
    echo "usage: $0 <deployment.yaml> [...] [--force]" >&2
    exit 1
fi

REPO_ARGS=()
if [ -n "${MODEL_REPOSITORY:-}" ]; then
    REPO_ARGS=(--model-repository "${MODEL_REPOSITORY}")
fi

python "${REPO_ROOT}/scripts/deploy.py" "${REPO_ARGS[@]}" "$@"
