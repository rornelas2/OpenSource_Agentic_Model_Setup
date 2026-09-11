#!/usr/bin/env bash
# Foreground server: run inside the H200 allocation shown in the README.
set -euo pipefail

if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo 'Run this script inside a Slurm GPU allocation (see README).' >&2
  exit 1
fi
root="/data/${USER:?}/pinnacles-agents"
scratch="/scratch/$USER/pinnacles-agents"
model="$root/models/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
if [[ ! -f $model/.pinnacles-revision ]] || \
   [[ $(cat "$model/.pinnacles-revision") != ff433f5493e25d631c9f12b5d55c674229923d02 ]]; then
  echo 'Complete scripts/download-nemotron.sh before starting the server.' >&2
  exit 1
fi
export HF_HOME="$root/huggingface"
export HF_HUB_OFFLINE=1
export TMPDIR="$scratch/nemotron/tmp"
export TRITON_CACHE_DIR="$scratch/nemotron/triton"
export VLLM_CACHE_ROOT="$scratch/nemotron/vllm"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"
exec "$root/envs/nemotron-vllm/bin/vllm" serve "$model" \
  --served-model-name nemotron-3-super \
  --host 127.0.0.1 --port 8000 \
  --dtype auto --max-model-len 32768 --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 \
  --kv-cache-dtype fp8 --mamba-ssm-cache-dtype float16 \
  --async-scheduling --enable-chunked-prefill \
  --max-cudagraph-capture-size 1 --moe-backend marlin \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser nemotron_v3 \
  --default-chat-template-kwargs '{"enable_thinking":true}'
