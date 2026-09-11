#!/usr/bin/env bash
# Foreground server: run inside the H200 allocation shown in the README.
set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo 'Run this script inside a Slurm GPU allocation (see README).' >&2
  exit 1
fi
root="/data/${USER:?}/pinnacles-agents"
scratch="/scratch/$USER/pinnacles-agents"
model="$root/models/Muse-Glimmer-30B"
if [[ ! -f $model/.pinnacles-revision ]] || \
   [[ $(cat "$model/.pinnacles-revision") != a4e59da52a7bc87ae7251dd5545c0dd437c44b68 ]]; then
  echo 'Complete scripts/download-muse.sh before starting the server.' >&2
  exit 1
fi
export HF_HOME="$root/huggingface"
export HF_HUB_OFFLINE=1
export TMPDIR="$scratch/tmp"
export TRITON_CACHE_DIR="$scratch/muse/triton"
export VLLM_CACHE_ROOT="$scratch/muse/vllm"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"
exec "$root/envs/muse-vllm/bin/vllm" serve "$model" \
  --served-model-name muse-glimmer-30b \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 32768 --max-num-seqs 1 \
  --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice --tool-call-parser muse_glimmer --reasoning-parser muse_glimmer \
  --default-chat-template-kwargs '{"reasoning_strength":"high"}' \
  --limit-mm-per-prompt '{"image":0,"video":0}'
