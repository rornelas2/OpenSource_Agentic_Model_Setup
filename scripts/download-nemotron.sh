#!/usr/bin/env bash
# Download only the pinned Nemotron checkpoint files needed for serving.
set -euo pipefail
umask 077

if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo 'Run the download in a Slurm CPU allocation (see README).' >&2
  exit 1
fi
root="/data/${USER:?}/pinnacles-agents"
export HF_HOME="$root/huggingface"
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
model="$root/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
mkdir -p "$HF_HOME" "$root/models"
"$root/envs/nemotron-vllm/bin/hf" download \
  nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --revision ff433f5493e25d631c9f12b5d55c674229923d02 \
  --include '*.json' --include '*.jinja' --include '*.safetensors' --include '*.py' \
  --local-dir "$model"
printf '%s\n' 'ff433f5493e25d631c9f12b5d55c674229923d02' \
  > "$model/.pinnacles-revision"
