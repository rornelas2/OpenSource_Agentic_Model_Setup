#!/usr/bin/env bash
# Download only the pinned checkpoint files used by the serving lesson.
set -euo pipefail
umask 077
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo 'Run the download in a Slurm CPU allocation (see README).' >&2
  exit 1
fi
root="/data/${USER:?}/pinnacles-agents"
model="${MUSE_MODEL_DIR:-$root/models/Muse-Glimmer-30B}"
export HF_HOME="$root/huggingface"
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
mkdir -p "$HF_HOME" "$model"
"$root/envs/muse-vllm/bin/hf" download meta-models/Muse-Glimmer-30B \
  --revision a4e59da52a7bc87ae7251dd5545c0dd437c44b68 \
  --include '*.json' --include '*.jinja' --include '*.safetensors' \
  --local-dir "$model"
printf '%s\n' 'a4e59da52a7bc87ae7251dd5545c0dd437c44b68' \
  > "$model/.pinnacles-revision"
