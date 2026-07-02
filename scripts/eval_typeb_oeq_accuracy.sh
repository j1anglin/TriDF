#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

MODEL_TAG="${1:-Qwen3-VL-8B-Instruct}"
RUN_NAME="${RUN_NAME:-}"
if [ -n "${RUN_NAME}" ]; then
  RUN_TAG="${RUN_NAME//\//-}"
  RUN_TAG="${RUN_TAG//:/_}"
  RUN_TAG="${RUN_TAG// /_}"
  PRED_DIR="${PRED_DIR:-${INFERENCE_ROOT}/typeb_oeq/${RUN_TAG}/models/${MODEL_TAG}}"
  OUT_CSV="${OUT_CSV:-${SCORING_ROOT}/OEQ_score/typeb_oeq_${MODEL_TAG}_${RUN_TAG}_accuracy.csv}"
else
  PRED_DIR="${PRED_DIR:-${INFERENCE_ROOT}/typeb_oeq/models/${MODEL_TAG}}"
  OUT_CSV="${OUT_CSV:-${SCORING_ROOT}/OEQ_score/typeb_oeq_${MODEL_TAG}_accuracy.csv}"
fi

CMD=(
  "${PYTHON_BIN}" -m inference.eval_accuracy
  "${PRED_DIR}"
  --recursive
  --single-record-only
  -o "${OUT_CSV}"
)

if [ "${STRICT:-1}" = "0" ]; then
  CMD+=(--no-strict)
else
  CMD+=(--strict)
fi

exec "${CMD[@]}"
