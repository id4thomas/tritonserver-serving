#!/bin/bash

VERSION=$(tr -d '[:space:]' < VERSION)
if [ -z "${VERSION}" ]; then
  echo "VERSION file is empty or invalid"
  exit 1
fi

docker build -t triton-venv-builder:${VERSION} -f Dockerfile .
