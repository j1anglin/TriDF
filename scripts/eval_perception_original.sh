#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

BENCHMARK="${1:-tf}"
MODEL_TAG="${2:-Qwen3-VL-8B-Instruct}"

case "${BENCHMARK}" in
  tf)
    DEFAULT_QUESTIONS_DIR="${TRIDF_ROOT}/3_Benchmark/benchmark_perception_tf"
    DEFAULT_PRED_DIR="${INFERENCE_ROOT}/perception_tf/${MODEL_TAG}"
    DEFAULT_OUT_DIR="${SCORING_ROOT}/TFQ_score"
    ;;
  mc)
    DEFAULT_QUESTIONS_DIR="${TRIDF_ROOT}/3_Benchmark/benchmark_perception_mc"
    DEFAULT_PRED_DIR="${INFERENCE_ROOT}/perception_mc/${MODEL_TAG}"
    DEFAULT_OUT_DIR="${SCORING_ROOT}/MCQ_score"
    ;;
  *)
    echo "[ERROR] BENCHMARK must be 'tf' or 'mc'." >&2
    exit 1
    ;;
esac

QUESTIONS_DIR="${QUESTIONS_DIR:-${DEFAULT_QUESTIONS_DIR}}"
PRED_DIR="${PRED_DIR:-${DEFAULT_PRED_DIR}}"
OUT_DIR="${OUT_DIR:-${DEFAULT_OUT_DIR}}"
SUMMARY_OUT="${SUMMARY_OUT:-${OUT_DIR}/${BENCHMARK}_${MODEL_TAG}_summary.json}"
DETAILS_OUT="${DETAILS_OUT:-${OUT_DIR}/${BENCHMARK}_${MODEL_TAG}_details.jsonl}"

exec "${PYTHON_BIN}" -m inference.score_perception_original \
  --benchmark "${BENCHMARK}" \
  --questions-dir "${QUESTIONS_DIR}" \
  --predictions-root "${PRED_DIR}" \
  --summary-out "${SUMMARY_OUT}" \
  --details-out "${DETAILS_OUT}"
