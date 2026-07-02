#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

MODEL_TAG="${1:-Qwen3-VL-8B-Instruct}"
# Space-separated list; set BENCHMARK_TASK to restrict to a single task.
BENCHMARK_TASKS="${BENCHMARK_TASK:-typea_oeq typeb_oeq}"
RUN_NAME="${RUN_NAME:-}"

MAPPER_BACKEND="${MAPPER_BACKEND:-openai}"
if [ "${MAPPER_BACKEND}" = "gemini" ]; then
  MAPPER_MODEL="${MAPPER_MODEL:-gemini-3.1-flash-lite}"
else
  MAPPER_MODEL="${MAPPER_MODEL:-gpt-5-mini}"
fi
MAPPER_MODEL_TAG="${MAPPER_MODEL//\//-}"
MAPPER_MODEL_TAG="${MAPPER_MODEL_TAG//:/_}"

BATCH_SIZE="${BATCH_SIZE:-200}"
POLL_INTERVAL="${POLL_INTERVAL:-30}"
MAX_PARALLEL_BATCHES="${MAX_PARALLEL_BATCHES:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
COMPLETION_WINDOW="${COMPLETION_WINDOW:-24h}"
MODALITIES="${MODALITIES:-image video}"

if [ "${MAPPER_BACKEND}" = "openai" ] \
  && [ -z "${MAPPER_API_KEY:-}" ] \
  && [ -z "${OPENAI_API_KEY:-}" ] \
  && [ -z "${OPENAI_API_KEY_GPT5:-}" ] \
  && [ -z "${OPENAI_API_KEY_BATCH:-}" ]; then
  echo "[ERROR] OpenAI mapper requires OPENAI_API_KEY or MAPPER_API_KEY." >&2
  exit 1
fi

for BENCHMARK_TASK in ${BENCHMARK_TASKS}; do
  echo "[INFO] ===== ${BENCHMARK_TASK} ====="

  if [ -n "${RUN_NAME}" ]; then
    RUN_TAG="${RUN_NAME//\//-}"
    RUN_TAG="${RUN_TAG//:/_}"
    RUN_TAG="${RUN_TAG// /_}"
    SOURCE_ROOT="${INFERENCE_ROOT}/${BENCHMARK_TASK}/${RUN_TAG}/models/${MODEL_TAG}"
    MODEL_TAG_SUFFIX="${MODEL_TAG}_${RUN_TAG}"
  else
    SOURCE_ROOT="${INFERENCE_ROOT}/${BENCHMARK_TASK}/models/${MODEL_TAG}"
    MODEL_TAG_SUFFIX="${MODEL_TAG}"
  fi

  MAPPING_ROOT="${SCORING_ROOT}/mapping_result/${BENCHMARK_TASK}/${MODEL_TAG_SUFFIX}_${MAPPER_MODEL_TAG}"
  EVAL_ROOT="${SCORING_ROOT}/OEQ_score"
  EVAL_MODEL="${MODEL_TAG_SUFFIX}_${MAPPER_MODEL_TAG}"
  METRICS_CSV="${SCORING_ROOT}/OEQ_score/${BENCHMARK_TASK}_${EVAL_MODEL}_metrics.csv"

  MAPPER_CMD=(
    "${PYTHON_BIN}" -m inference.runner.eval_hallucination_from_analysis_text
    --input-root "${SOURCE_ROOT}"
    --output-dir "${MAPPING_ROOT}"
    --analysis-field response
    --modalities ${MODALITIES}
    --backend "${MAPPER_BACKEND}"
    --model-id "${MAPPER_MODEL}"
    --batch-size "${BATCH_SIZE}"
    --poll-interval "${POLL_INTERVAL}"
    --max-parallel-batches "${MAX_PARALLEL_BATCHES}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --completion-window "${COMPLETION_WINDOW}"
    --skip-existing
  )

  if [ -n "${MAPPER_API_KEY:-}" ]; then
    MAPPER_CMD+=(--api-key "${MAPPER_API_KEY}")
  fi

  "${MAPPER_CMD[@]}"

  for _mod in ${MODALITIES}; do
    "${PYTHON_BIN}" -m inference.convert_mapping_result_to_eval_csv \
      --input-root "${MAPPING_ROOT}" \
      --output-root "${EVAL_ROOT}" \
      --modality "${_mod}" \
      --benchmark "${BENCHMARK_TASK}" \
      --model "${EVAL_MODEL}" \
      --prefer-source-id
  done

  "${PYTHON_BIN}" -m inference.cal_score \
    --eval-root "${EVAL_ROOT}" \
    --gt-root "${TRIDF_ROOT}/2_GT_Final" \
    --benchmark-task "${BENCHMARK_TASK}" \
    --model "${EVAL_MODEL}" \
    --combine-modalities \
    --output "${METRICS_CSV}"

done
