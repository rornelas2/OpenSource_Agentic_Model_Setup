#!/usr/bin/env bash
# Download only the pinned checkpoint files used by the serving lesson.
set -euo pipefail
umask 077
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo 'Run the download in a Slurm CPU allocation (see README).' >&2
  exit 1
fi
root="/data/${USER:?}/pinnacles-agents"
export HF_HOME="$root/huggingface"
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0
mkdir -p "$HF_HOME" "$root/models"
"$root/envs/gemma-vllm/bin/hf" download google/gemma-4-31B-it \
  --revision 842da3794eaa0b77d5f08bae87a17459d91ff475 \
  --include '*.json' --include '*.jinja' --include '*.safetensors' \
  --local-dir "$root/models/gemma-4-31B-it"
printf '%s\n' '842da3794eaa0b77d5f08bae87a17459d91ff475' \
  > "$root/models/gemma-4-31B-it/.pinnacles-revision"
