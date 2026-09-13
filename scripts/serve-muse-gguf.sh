#!/usr/bin/env bash
# Foreground single-user server. MUSE_GPU=a100|l40s is required.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/muse-gguf-common.sh"
muse_limits
muse_allocation
muse_gpu
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
  # llama.cpp divides total context among slots; MUSE_CONTEXT is per request.
  -ngl 99 -c "$((MUSE_CONTEXT * MUSE_PARALLEL))" -np "$MUSE_PARALLEL" -n "$MUSE_OUTPUT" --jinja
  --host 127.0.0.1 --port "$MUSE_PORT" --temp 1.0 --top-p 0.95 --top-k 64
  --chat-template-kwargs '{"reasoning_strength":"high"}'
  --fit off --no-context-shift --no-webui --log-verbosity 4)
exec "$muse_runtime/bin/llama-server" "${args[@]}"
