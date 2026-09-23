#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -f .env ]]; then set -a; source .env; set +a; fi
main_python="${COSEEK_MAIN_PYTHON:-.venvs/main/bin/python}"
if [[ ! -x "$main_python" ]]; then echo 'Run ./setup.sh first, or set COSEEK_MAIN_PYTHON.' >&2; exit 2; fi
export PATH="$(dirname -- "$main_python"):$PATH"
export LITELLM_LOCAL_MODEL_COST_MAP=True
exec "$main_python" scripts/portable.py "${@:-run}"
