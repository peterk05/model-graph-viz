#!/usr/bin/env bash
# Export TinyLlama-1.1B-Chat-v1.0 to OpenVINO IR (fp16 weights), exactly as in the brief.
# Output: models/tinyllama/openvino_model.xml + .bin (~2.2 GB, not committed to git).
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f models/tinyllama/openvino_model.xml ]; then
  echo "models/tinyllama/openvino_model.xml already exists, nothing to do"
  exit 0
fi

python -m pip install --quiet "optimum[openvino]"
optimum-cli export openvino \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --weight-format fp16 \
  models/tinyllama
echo "exported to models/tinyllama/"
