#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "CONDA_PREFIX is not set. Run this through the artalk-web conda/mamba environment." >&2
  echo "Example: micromamba run -n artalk-web scripts/run_web_backend.sh" >&2
  exit 1
fi

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

exec uvicorn web_app:app --host "${ARTALK_WEB_HOST:-0.0.0.0}" --port "${ARTALK_WEB_PORT:-8961}"
