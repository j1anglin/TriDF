#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

MODEL_ID="${1:-Qwen/Qwen3-VL-8B-Instruct}"
MAX_SAMPLES="${2-10}"
QUESTIONS_DIR="${QUESTIONS_DIR:-${TRIDF_ROOT}/3_Benchmark/benchmark_perception_mc}"
OUTPUT_DIR="${OUTPUT_DIR:-${INFERENCE_ROOT}/perception_mc}"
MODALITIES=(${MODALITIES:-img})

CMD=(
  "${PYTHON_BIN}" -m inference.cli.run_perception_mc
  --questions-dir "${QUESTIONS_DIR}"
  --data-root "${DATA_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --model "${MODEL_ID}"
  --modalities "${MODALITIES[@]}"
  --cache-dir "${CACHE_DIR}"
)

if [ -n "${MAX_SAMPLES}" ]; then
  CMD+=(--max-samples "${MAX_SAMPLES}")
fi
if [ "${OFFLINE:-0}" = "1" ]; then
  CMD+=(--offline)
fi

echo "[INFO] model=${MODEL_ID}"
echo "[INFO] questions=${QUESTIONS_DIR}"
echo "[INFO] output=${OUTPUT_DIR}"
exec "${CMD[@]}"
