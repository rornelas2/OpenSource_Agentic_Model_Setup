# Internal helpers for the isolated Gemma QAT profile; source from its scripts.
gemma_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$gemma_repo/env/gemma-qat-pins.sh"
gemma_root="/data/${USER:?}/pinnacles-agents"
gemma_scratch="/scratch/$USER/pinnacles-agents/gemma-qat"
gemma_env="$gemma_root/envs/gemma-vllm"
gemma_model="$gemma_root/models/gemma-4-31B-it-qat-w4a16-ct-$GEMMA_QAT_REVISION"
gemma_fail() { echo "Gemma QAT: $*" >&2; exit 1; }

gemma_allocation() {
  [[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || gemma_fail 'Run inside a Slurm allocation.'
  grep -Eq "/job_${SLURM_JOB_ID}(/|$)" /proc/self/cgroup ||
    gemma_fail 'This process is not in the allocated Slurm compute-node cgroup.'
  local job
  job=$(scontrol show job "$SLURM_JOB_ID" -o) || gemma_fail 'Cannot verify Slurm job.'
  [[ " $job " == *' JobState=RUNNING '* && " $job " == *" UserId=$USER($(id -u)) "* ]] ||
    gemma_fail 'The allocation must be running and owned by the current user.'
}

gemma_uint() {
  [[ $2 =~ ^[1-9][0-9]{0,5}$ ]] || gemma_fail "$1 must be a positive decimal integer."
  (( $2 >= $3 && $2 <= $4 )) || gemma_fail "$1 must be in $3..$4."
}

gemma_limits() {
  GEMMA_CONTEXT=${GEMMA_CONTEXT:-32768}
  GEMMA_OUTPUT=${GEMMA_OUTPUT:-8192}
  GEMMA_PORT=${GEMMA_PORT:-8000}
  GEMMA_PARALLEL=${GEMMA_PARALLEL:-1}
  gemma_uint GEMMA_CONTEXT "$GEMMA_CONTEXT" 1 32768
  [[ $GEMMA_CONTEXT == 16384 || $GEMMA_CONTEXT == 32768 ]] ||
    gemma_fail 'GEMMA_CONTEXT must be 16384 or 32768 (experimental until validated).'
  gemma_uint GEMMA_OUTPUT "$GEMMA_OUTPUT" 1 8192
  (( GEMMA_OUTPUT <= GEMMA_CONTEXT )) || gemma_fail 'Output must be less than or equal to context.'
  gemma_uint GEMMA_PORT "$GEMMA_PORT" 1024 65535
  gemma_uint GEMMA_PARALLEL "$GEMMA_PARALLEL" 1 8
  export GEMMA_CONTEXT GEMMA_OUTPUT GEMMA_PORT GEMMA_PARALLEL
}

gemma_gpu() {
  [[ ${SLURM_GPUS_ON_NODE:-} == 1 ]] || gemma_fail 'Request exactly one GPU on this node.'
  [[ ${GEMMA_GPU:-} == a100 || ${GEMMA_GPU:-} == l40s ]] ||
    gemma_fail 'Set GEMMA_GPU=a100 or GEMMA_GPU=l40s to select the intended GPU.'
  [[ ${CUDA_VISIBLE_DEVICES:-} =~ ^([0-9]+|GPU-[a-fA-F0-9-]+)$ ]] ||
    gemma_fail 'Exactly one Slurm-visible GPU is required.'
  local info
  info=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name,memory.total --format=csv,noheader,nounits) ||
    gemma_fail 'Cannot inspect the allocated GPU.'
  [[ $info != *$'\n'* ]] || gemma_fail 'Multiple GPUs were returned.'
  case $GEMMA_GPU in
    a100) [[ $info == *A100*40GB* ]] || gemma_fail "Expected A100 40GB; found $info" ;;
    l40s) [[ $info == *L40S* ]] || gemma_fail "Expected L40S; found $info" ;;
  esac
  echo "Allocated GPU: $info MiB"
}

gemma_checkpoint() {
  python3 - "$gemma_model" "$GEMMA_QAT_REVISION" "$GEMMA_QAT_FILE" "$GEMMA_QAT_BYTES" "$GEMMA_QAT_SHA256" <<'PY'
import json, pathlib, sys
root, revision, name, size, digest = sys.argv[1:]
try:
    path = pathlib.Path(root) / name
    stamp = path.stat()
    marker = json.loads((path.parent / '.pinnacles-complete.json').read_text())
    expected = dict(revision=revision, file=name, bytes=int(size), sha256=digest,
                    mtime_ns=stamp.st_mtime_ns)
    if marker != expected or stamp.st_size != int(size):
        raise ValueError('checkpoint identity/size/mtime mismatch')
except (OSError, ValueError) as exc:
    sys.exit(f'Gemma QAT: run download-gemma-qat.sh in a CPU allocation: {exc}')
PY
}

gemma_port_free() {
  python3 - "$GEMMA_PORT" <<'PY'
import socket, sys
try:
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', int(sys.argv[1])))
except OSError as exc:
    sys.exit(f'Gemma QAT: port is unavailable: {exc}')
PY
}

gemma_runtime_check() {
  [[ -x $gemma_env/bin/vllm ]] || gemma_fail "vLLM binary missing: $gemma_env/bin/vllm"
  [[ -x $gemma_root/tools/opencode-v1.18.30/opencode ]] ||
    gemma_fail "OpenCode binary missing: $gemma_root/tools/opencode-v1.18.30/opencode"
  [[ -f $gemma_root/tools/tool_chat_template_gemma4.jinja ]] ||
    gemma_fail "Jinja template missing: $gemma_root/tools/tool_chat_template_gemma4.jinja"
  (cd "$gemma_root/tools" && echo 'afdbb2abe3667ccde95cc2f86919f05370339399bab5f750950a4390523b8927  tool_chat_template_gemma4.jinja' | sha256sum --check >/dev/null 2>&1) ||
    gemma_fail "Jinja template SHA256 mismatch: $gemma_root/tools/tool_chat_template_gemma4.jinja"
}

gemma_multigpu_limits() {
  GEMMA_CONTEXT=${GEMMA_CONTEXT:-131072}
  GEMMA_OUTPUT=${GEMMA_OUTPUT:-8192}
  GEMMA_PORT=${GEMMA_PORT:-8000}
  GEMMA_PARALLEL=${GEMMA_PARALLEL:-1}
  gemma_uint GEMMA_CONTEXT "$GEMMA_CONTEXT" 1 131072
  case $GEMMA_CONTEXT in
    16384|32768|65536|131072) ;;
    *) gemma_fail 'GEMMA_CONTEXT must be 16384, 32768, 65536, or 131072.' ;;
  esac
  gemma_uint GEMMA_OUTPUT "$GEMMA_OUTPUT" 1 16384
  (( GEMMA_OUTPUT <= GEMMA_CONTEXT )) || gemma_fail 'Output must be less than or equal to context.'
  gemma_uint GEMMA_PORT "$GEMMA_PORT" 1024 65535
  gemma_uint GEMMA_PARALLEL "$GEMMA_PARALLEL" 1 8
  export GEMMA_CONTEXT GEMMA_OUTPUT GEMMA_PORT GEMMA_PARALLEL
}

gemma_multigpu_gpu() {
  [[ ${SLURM_GPUS_ON_NODE:-} == 2 ]] || gemma_fail 'Request exactly two GPUs on this node (--gres=gpu:a100:2 or --gres=gpu:l40s:2).'
  [[ ${GEMMA_GPU:-} == a100 || ${GEMMA_GPU:-} == l40s ]] ||
    gemma_fail 'Set GEMMA_GPU=a100 or GEMMA_GPU=l40s to select the intended GPU.'
  [[ ${CUDA_VISIBLE_DEVICES:-} =~ ^([0-9]+|GPU-[a-fA-F0-9-]+),([0-9]+|GPU-[a-fA-F0-9-]+)$ ]] ||
    gemma_fail 'Exactly two Slurm-visible GPUs are required.'
  [[ ${CUDA_VISIBLE_DEVICES%,*} != "${CUDA_VISIBLE_DEVICES#*,}" ]] ||
    gemma_fail 'Slurm-visible GPU identifiers must be distinct.'
  local info row
  local -a rows
  info=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name,memory.total --format=csv,noheader,nounits) ||
    gemma_fail 'Cannot inspect allocated GPUs via nvidia-smi.'
  mapfile -t rows <<< "$info"
  [[ ${#rows[@]} == 2 ]] || gemma_fail "Expected 2 allocated GPUs; found ${#rows[@]}."
  for row in "${rows[@]}"; do
    case $GEMMA_GPU in
      a100) [[ $row == *A100*40GB* ]] || gemma_fail "Expected A100 40GB; found $row" ;;
      l40s) [[ $row == *L40S* ]] || gemma_fail "Expected L40S; found $row" ;;
    esac
  done
  echo "Allocated Dual GPUs:"
  echo "$info"
}
