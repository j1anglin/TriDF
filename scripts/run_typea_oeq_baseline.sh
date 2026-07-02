#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

MODEL_ID="${1:-Qwen/Qwen3-VL-8B-Instruct}"
MAX_SAMPLES="${2-10}"
MODALITY="${TYPEA_OEQ_MODALITY_FILTER:-image}"
COLLECT_NAME="${COLLECT_NAME:-collect.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${INFERENCE_ROOT}/typea_oeq}"

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
    *) echo "[ERROR] Unknown TYPEA_OEQ_MODALITY_FILTER=${MODALITY}" >&2; exit 1 ;;
  esac
  TASKS+=("${task_name}")
done

CMD=(
  "${PYTHON_BIN}" -m inference.cli.run_perception
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
exec "${CMD[@]}"
