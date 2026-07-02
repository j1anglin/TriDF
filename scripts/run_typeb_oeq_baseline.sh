#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

MODEL_ID="${1:-Qwen/Qwen3-VL-8B-Instruct}"
MAX_SAMPLES="${2-10}"
RUN_NAME="${RUN_NAME:-}"
MODALITY="${TYPEB_OEQ_MODALITY_FILTER:-image}"
COLLECT_NAME="${COLLECT_NAME:-collect.csv}"
if [ -n "${RUN_NAME}" ]; then
  RUN_TAG="${RUN_NAME//\//-}"
  RUN_TAG="${RUN_TAG//:/_}"
  RUN_TAG="${RUN_TAG// /_}"
  OUTPUT_ROOT="${OUTPUT_ROOT:-${INFERENCE_ROOT}/typeb_oeq/${RUN_TAG}}"
else
  OUTPUT_ROOT="${OUTPUT_ROOT:-${INFERENCE_ROOT}/typeb_oeq}"
fi

TASKS=()
for task_dir in "${DATA_ROOT}"/*; do
  [ -d "${task_dir}" ] || continue
  [ -f "${task_dir}/${COLLECT_NAME}" ] || continue
  task_name="$(basename "${task_dir}")"
  case "${MODALITY}" in
    image) [[ "${task_name}" == img_* ]] || continue ;;
    video) [[ "${task_name}" == vid_* ]] || continue ;;
    audio) [[ "${task_name}" == aud_* ]] || continue ;;
    all) ;;
    *) echo "[ERROR] Unknown TYPEB_OEQ_MODALITY_FILTER=${MODALITY}" >&2; exit 1 ;;
  esac
  TASKS+=("${task_name}")
done

if [ "${#TASKS[@]}" -eq 0 ]; then
  echo "[ERROR] No TypeB OEQ tasks found under ${DATA_ROOT} for modality=${MODALITY}" >&2
  exit 1
fi

CMD=(
  "${PYTHON_BIN}" -m inference.cli.run_detection
  --benchmark-root "${DATA_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --models "${MODEL_ID}"
  --collect-name "${COLLECT_NAME}"
  --only-tasks "${TASKS[@]}"
  --cache-dir "${CACHE_DIR}"
  --preview-policy skip
)

if [ -n "${MAX_SAMPLES}" ]; then
  CMD+=(--max-samples "${MAX_SAMPLES}")
fi
if [ "${OFFLINE:-0}" = "1" ]; then
  CMD+=(--offline)
fi

echo "[INFO] model=${MODEL_ID}"
echo "[INFO] output=${OUTPUT_ROOT}"
echo "[INFO] tasks=${#TASKS[@]} modality=${MODALITY}"
exec "${CMD[@]}"
