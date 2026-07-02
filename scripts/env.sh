#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRIDF_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${TRIDF_ROOT}/3_Benchmark${PYTHONPATH:+:${PYTHONPATH}}"
export DATA_ROOT="${DATA_ROOT:-${TRIDF_ROOT}/1_DATA}"
export RUNS_ROOT="${RUNS_ROOT:-${TRIDF_ROOT}/runs}"
export INFERENCE_ROOT="${INFERENCE_ROOT:-${RUNS_ROOT}/inference}"
export SCORING_ROOT="${SCORING_ROOT:-${RUNS_ROOT}/scoring}"
export LOG_ROOT="${LOG_ROOT:-${TRIDF_ROOT}/logs}"
export CACHE_DIR="${CACHE_DIR:-${TRIDF_ROOT}/models}"
export PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "${INFERENCE_ROOT}" "${SCORING_ROOT}" "${LOG_ROOT}" "${CACHE_DIR}"
