#!/usr/bin/env bash
# Foreground multi-GPU server for Meta Muse Glimmer 30B GGUF across dual GPUs (2x A100 or 2x L40S).
# Uses layer-wise GPU splitting (-sm layer -ts 1,1) and FlashAttention (--flash-attn on) to scale context to 131,072 tokens.
# MUSE_GPU=a100|l40s is required.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/muse-gguf-common.sh"
muse_multigpu_limits
muse_allocation
muse_multigpu_gpu
muse_checkpoint
muse_runtime_check
muse_port_free

version=$("$muse_runtime/bin/llama-server" --version 2>&1)
[[ $version == *"${MUSE_LLAMA_COMMIT:0:7}"* ]] || muse_fail 'Binary reports an unexpected runtime commit.'
printf '%s\n' "$version"

# Prevent inherited llama.cpp environment defaults from changing the profile.
while IFS= read -r name; do unset "$name"; done < <(compgen -v LLAMA_ARG_)
unset LLAMA_API_KEY

args=(-m "$muse_model/$MUSE_GGUF_FILE" -a "$MUSE_ALIAS"
  -sm layer -ts 1,1 -ngl 99
  -c "$((MUSE_CONTEXT * MUSE_PARALLEL))" -np "$MUSE_PARALLEL" -n "$MUSE_OUTPUT"
  --flash-attn on --jinja
  --host 127.0.0.1 --port "$MUSE_PORT" --temp 1.0 --top-p 0.95 --top-k 64
  --chat-template-kwargs '{"reasoning_strength":"high"}'
  --fit off --no-context-shift --no-webui --log-verbosity 4)

exec "$muse_runtime/bin/llama-server" "${args[@]}"
