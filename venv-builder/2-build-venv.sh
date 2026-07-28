#!/usr/bin/env bash
set -euo pipefail

REQUIREMENTS_FILE_NAME="${1:-}"

if [ -z "${REQUIREMENTS_FILE_NAME}" ]; then
  echo "ERROR: requirements file name is required."
  echo "Usage: $0 <requirements_file_name>"
  echo "Example: $0 text-clf-vllm-v1"
  exit 1
fi

REQUIREMENTS_DIR="requirements/${REQUIREMENTS_FILE_NAME}"
OUTPUT_DIR="envs"
CONTAINER_TAR="venv.tar.gz"
OUTPUT_TAR="${REQUIREMENTS_FILE_NAME}.tar.gz"
VERSION=$(tr -d '[:space:]' < VERSION)
if [ -z "${VERSION}" ]; then
  echo "VERSION file is empty or invalid"
  exit 1
fi

IMAGE_NAME="triton-venv-builder:${VERSION}"
CONTAINER_NAME="triton-venv-builder-${VERSION}-$$"

# Validate input
if [ ! -d "${REQUIREMENTS_DIR}" ]; then
  echo "ERROR: ${REQUIREMENTS_DIR} directory not found."
  echo "Usage: $0 <requirements_dir_name>"
  exit 1
fi

# Run container with GPU access, mounting requirements folder
echo "==> Creating environment from ${REQUIREMENTS_DIR}/ ..."
docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
docker run --name "${CONTAINER_NAME}" --gpus all \
  -v "$(pwd)/${REQUIREMENTS_DIR}:/tmp/requirements:ro" \
  "${IMAGE_NAME}"

# Copy the packed environment out
echo "==> Copying packed environment to host ..."
mkdir -p "${OUTPUT_DIR}"
docker cp "${CONTAINER_NAME}:/tmp/${CONTAINER_TAR}" "${OUTPUT_DIR}/${OUTPUT_TAR}"
docker rm "${CONTAINER_NAME}" >/dev/null

echo ""
echo "Done!  ${OUTPUT_DIR}/${OUTPUT_TAR}"
