#!/usr/bin/env bash
set -euo pipefail

SERVER_URL="${1:-localhost:8000}"

RESPONSE=$(curl -s "${SERVER_URL}/v2/models/text-classification/infer" \
    -H "Content-Type: application/json" \
    -d '{
  "inputs": [
    {
      "name": "text",
      "shape": [1, 1],
      "datatype": "BYTES",
      "data": ["This movie was absolutely wonderful!"]
    }
  ],
  "outputs": [
    {
      "name": "label"
    },
    {
      "name": "scores"
    }
  ]
}')

echo "==> Response:"
echo "${RESPONSE}"
