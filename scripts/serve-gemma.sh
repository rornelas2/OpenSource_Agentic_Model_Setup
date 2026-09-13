#!/usr/bin/env bash
# Foreground server: run inside the H200 allocation shown in the README.
set -euo pipefail
if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo 'Run this script inside a Slurm GPU allocation (see README).' >&2
  exit 1
fi
root="/data/${USER:?}/pinnacles-agents"
scratch="/scratch/$USER/pinnacles-agents"
model="$root/models/gemma-4-31B-it"
if [[ ! -f $model/.pinnacles-revision ]] || \
   [[ $(cat "$model/.pinnacles-revision") != 842da3794eaa0b77d5f08bae87a17459d91ff475 ]]; then
  echo 'Complete scripts/download-gemma.sh before starting the server.' >&2
  exit 1
fi
export HF_HOME="$root/huggingface"
export HF_HUB_OFFLINE=1
export TMPDIR="$scratch/tmp"
export TRITON_CACHE_DIR="$scratch/triton"
export VLLM_CACHE_ROOT="$scratch/vllm"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"
exec "$root/envs/gemma-vllm/bin/vllm" serve "$model" \
  --served-model-name gemma4-31b \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 32768 --max-num-seqs 1 \
  --gpu-memory-utilization "${GEMMA_GPU_UTIL:-0.88}" \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --chat-template "$root/tools/tool_chat_template_gemma4.jinja" \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --limit-mm-per-prompt '{"image":0,"video":0}'
