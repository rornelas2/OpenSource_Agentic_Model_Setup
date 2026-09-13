#!/usr/bin/env bash
# Foreground single-user server for Gemma QAT on A100 (40GB) or L40S (48GB).
# GEMMA_GPU=a100|l40s is required.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/gemma-qat-common.sh"
gemma_limits
gemma_allocation
gemma_gpu
gemma_checkpoint
gemma_runtime_check
gemma_port_free

export HF_HOME="$gemma_root/huggingface"
export HF_HUB_OFFLINE=1
export TMPDIR="$gemma_scratch/tmp"
export TRITON_CACHE_DIR="$gemma_scratch/triton"
export VLLM_CACHE_ROOT="$gemma_scratch/vllm"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"

utilization="${GEMMA_GPU_UTIL:-0.90}"

exec "$gemma_env/bin/vllm" serve "$gemma_model" \
  --served-model-name "$GEMMA_QAT_ALIAS" \
  --host 127.0.0.1 --port "$GEMMA_PORT" \
  --dtype bfloat16 \
  --max-model-len "$GEMMA_CONTEXT" --max-num-seqs "$GEMMA_PARALLEL" \
  --gpu-memory-utilization "$utilization" \
  --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --chat-template "$gemma_root/tools/tool_chat_template_gemma4.jinja" \
  --default-chat-template-kwargs '{"enable_thinking":false}' \
  --limit-mm-per-prompt '{"image":0,"video":0}'
