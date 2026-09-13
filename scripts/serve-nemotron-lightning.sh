#!/usr/bin/env bash
# Foreground single-user server for Nemotron Lightning on A100 (40GB) or L40S (48GB).
# NEMOTRON_GPU=a100|l40s is required.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/nemotron-lightning-common.sh"
nemotron_limits
nemotron_allocation
nemotron_gpu
nemotron_checkpoint
nemotron_runtime_check
nemotron_port_free

export HF_HOME="$nemotron_root/huggingface"
export HF_HUB_OFFLINE=1
export TMPDIR="$nemotron_scratch/tmp"
export TRITON_CACHE_DIR="$nemotron_scratch/triton"
export VLLM_CACHE_ROOT="$nemotron_scratch/vllm"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"

utilization="${NEMOTRON_GPU_UTIL:-0.90}"

extra_args=()
if [[ $NEMOTRON_GPU == "a100" ]]; then
  export VLLM_HUMMING_MOE_GEMM_TYPE=indexed
  extra_args+=(
    --moe-backend "${NEMOTRON_MOE_BACKEND:-humming}"
    --linear-backend "${NEMOTRON_LINEAR_BACKEND:-humming}"
    --quantization modelopt_fp4
    --mamba-ssu-algorithm simple
  )
elif [[ $NEMOTRON_GPU == "l40s" ]]; then
  extra_args+=(
    --moe-backend "${NEMOTRON_MOE_BACKEND:-marlin}"
    --kv-cache-dtype fp8
    --mamba-ssm-cache-dtype float16
  )
fi

exec "$nemotron_env/bin/vllm" serve "$nemotron_model" \
  --served-model-name "$NEMOTRON_LIGHTNING_ALIAS" \
  --host 127.0.0.1 --port "$NEMOTRON_PORT" \
  --max-model-len "$NEMOTRON_CONTEXT" --max-num-seqs "$NEMOTRON_PARALLEL" \
  --gpu-memory-utilization "$utilization" \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 \
  --default-chat-template-kwargs '{"enable_thinking":true}' \
  --mamba-backend flashinfer --mamba-cache-mode align \
  --async-scheduling --enable-prefix-caching \
  "${extra_args[@]}"
