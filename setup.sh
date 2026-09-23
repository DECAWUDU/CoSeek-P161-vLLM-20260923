#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || { echo 'This release requires Linux x86_64 + NVIDIA CUDA.'; exit 2; }
mode="${1:---full}"
[[ "$mode" == --full || "$mode" == --main-only || "$mode" == --download-model ]] || { echo 'Usage: setup.sh [--full|--main-only|--download-model]'; exit 2; }
if ! command -v ffmpeg >/dev/null || ! command -v ffprobe >/dev/null || ! command -v git >/dev/null; then
  if [[ $EUID == 0 ]] && command -v apt-get >/dev/null; then apt-get update; apt-get install -y ffmpeg git ca-certificates;
  else echo 'Install ffmpeg, ffprobe and git first (Ubuntu: sudo apt-get install ffmpeg git).'; exit 2; fi
fi
[[ "$mode" == --main-only ]] || command -v nvidia-smi >/dev/null || { echo 'NVIDIA driver compatible with the selected CUDA 12.8 wheels is required for the local search model.'; exit 2; }
if ! command -v uv >/dev/null; then
  python3 -m venv .bootstrap
  .bootstrap/bin/python -m pip install 'uv>=0.10.9'
  uv_bin="$PWD/.bootstrap/bin/uv"
else uv_bin="$(command -v uv)"; fi
"$uv_bin" python install 3.13.0 3.11.15
[[ -d .venvs/main ]] || "$uv_bin" venv --python 3.13.0 .venvs/main
if [[ "$mode" != --main-only ]]; then
[[ -d .venvs/qwen ]] || "$uv_bin" venv --python 3.11.15 .venvs/qwen
"$uv_bin" pip install --python .venvs/qwen/bin/python 'torch==2.10.0' 'torchvision==0.25.0' --index-url https://download.pytorch.org/whl/cu128
"$uv_bin" pip install --python .venvs/qwen/bin/python -r requirements/qwen.txt
fi
"$uv_bin" pip install --python .venvs/main/bin/python -r requirements/main.lock.txt
.venvs/main/bin/python -c 'import tiktoken; tiktoken.get_encoding("o200k_base")'
mkdir -p runs models
if [[ ! -f .env ]]; then cp .env.example .env; chmod 600 .env; fi
if [[ "${1:-}" == --download-model ]]; then
  .venvs/qwen/bin/python -c 'from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3.5-9B",local_dir="models/Qwen3.5-9B")'
fi
.venvs/main/bin/python scripts/portable.py verify
.venvs/main/bin/python -m unittest discover -s tests -v
printf '\nSetup complete. Fill .env and examples/cases.json, then ./run.sh doctor and ./run.sh run\n'
