# Internal helpers for the isolated Muse GGUF profile; source from its scripts.
muse_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$muse_repo/env/muse-gguf-pins.sh"
muse_root="/data/${USER:?}/pinnacles-agents"
muse_scratch="/scratch/$USER/pinnacles-agents/muse-gguf"
muse_runtime="$muse_root/tools/llama.cpp-$MUSE_LLAMA_COMMIT"
muse_model="$muse_root/models/Muse-Glimmer-30B-GGUF-$MUSE_GGUF_REVISION"
muse_fail() { echo "Muse GGUF: $*" >&2; exit 1; }

muse_allocation() {
  [[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || muse_fail 'Run inside a Slurm allocation.'
  # A copied job ID or login shell is insufficient: this process must belong
  # to that job's compute-node cgroup (Pinnacles uses cgroup v2).
  grep -Eq "/job_${SLURM_JOB_ID}(/|$)" /proc/self/cgroup ||
    muse_fail 'This process is not in the allocated Slurm compute-node cgroup.'
  local job
  job=$(scontrol show job "$SLURM_JOB_ID" -o) || muse_fail 'Cannot verify Slurm job.'
  [[ " $job " == *' JobState=RUNNING '* && " $job " == *" UserId=$USER($(id -u)) "* ]] ||
    muse_fail 'The allocation must be running and owned by the current user.'
}

muse_uint() {
  # Length bound avoids shell arithmetic overflow; reject signs/expressions.
  [[ $2 =~ ^[1-9][0-9]{0,5}$ ]] || muse_fail "$1 must be a positive decimal integer."
  (( $2 >= $3 && $2 <= $4 )) || muse_fail "$1 must be in $3..$4."
}

muse_limits() {
  MUSE_CONTEXT=${MUSE_CONTEXT:-32768}
  MUSE_OUTPUT=${MUSE_OUTPUT:-8192}
  MUSE_PORT=${MUSE_PORT:-8000}
  MUSE_PARALLEL=${MUSE_PARALLEL:-1}
  muse_uint MUSE_CONTEXT "$MUSE_CONTEXT" 1 32768
  [[ $MUSE_CONTEXT == 16384 || $MUSE_CONTEXT == 32768 ]] ||
    muse_fail 'MUSE_CONTEXT must be 16384 or 32768 (experimental until validated).'
  muse_uint MUSE_OUTPUT "$MUSE_OUTPUT" 1 8192
  (( MUSE_OUTPUT <= MUSE_CONTEXT )) || muse_fail 'Output must be less than or equal to context.'
  muse_uint MUSE_PORT "$MUSE_PORT" 1024 65535
  muse_uint MUSE_PARALLEL "$MUSE_PARALLEL" 1 4
  export MUSE_CONTEXT MUSE_OUTPUT MUSE_PORT MUSE_PARALLEL
}

muse_gpu() {
  [[ ${SLURM_GPUS_ON_NODE:-} == 1 ]] || muse_fail 'Request exactly one GPU on this node.'
  [[ ${MUSE_GPU:-} == a100 || ${MUSE_GPU:-} == l40s ]] ||
    muse_fail 'Set MUSE_GPU=a100 or MUSE_GPU=l40s to select the intended GPU.'
  [[ ${CUDA_VISIBLE_DEVICES:-} =~ ^([0-9]+|GPU-[a-fA-F0-9-]+)$ ]] ||
    muse_fail 'Exactly one Slurm-visible GPU is required.'
  local info
  info=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name,memory.total --format=csv,noheader,nounits) ||
    muse_fail 'Cannot inspect the allocated GPU.'
  [[ $info != *$'\n'* ]] || muse_fail 'Multiple GPUs were returned.'
  case $MUSE_GPU in
    a100) [[ $info == *A100*40GB* ]] || muse_fail "Expected A100 40GB; found $info" ;;
    l40s) [[ $info == *L40S* ]] || muse_fail "Expected L40S; found $info" ;;
  esac
  echo "Allocated GPU: $info MiB"
}

muse_checkpoint() {
  python3 - "$muse_model" "$MUSE_GGUF_REVISION" "$MUSE_GGUF_FILE" "$MUSE_GGUF_BYTES" "$MUSE_GGUF_SHA256" <<'PY'
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
    sys.exit(f'Muse GGUF: run download-muse-gguf.sh in a CPU allocation: {exc}')
PY
}

muse_runtime_check() {
  [[ -f $muse_runtime/.pinnacles-build-profile &&
     $(cat "$muse_runtime/.pinnacles-build-profile") == cuda13-sm80-sm89-no-ui-v1 ]] ||
    muse_fail 'Runtime build profile differs; expected CUDA 13, SM80/89, no downloaded UI.'
  [[ -f $muse_runtime/.pinnacles-commit &&
     $(cat "$muse_runtime/.pinnacles-commit") == "$MUSE_LLAMA_COMMIT" ]] ||
    muse_fail 'Run setup-muse-gguf.sh to install the pinned CUDA runtime.'
  (cd "$muse_runtime" && sha256sum --check --status runtime.sha256) ||
    muse_fail 'Runtime checksum failed; do not use this installation.'
}

muse_port_free() {
  python3 - "$MUSE_PORT" <<'PY'
import socket, sys
try:
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('127.0.0.1', int(sys.argv[1])))
except OSError as exc:
    sys.exit(f'Muse GGUF: port is unavailable: {exc}')
PY
}

muse_multigpu_limits() {
  MUSE_CONTEXT=${MUSE_CONTEXT:-131072}
  MUSE_OUTPUT=${MUSE_OUTPUT:-8192}
  MUSE_PORT=${MUSE_PORT:-8000}
  MUSE_PARALLEL=${MUSE_PARALLEL:-1}
  muse_uint MUSE_CONTEXT "$MUSE_CONTEXT" 1 131072
  case $MUSE_CONTEXT in
    16384|32768|65536|131072) ;;
    *) muse_fail 'MUSE_CONTEXT must be 16384, 32768, 65536, or 131072.' ;;
  esac
  muse_uint MUSE_OUTPUT "$MUSE_OUTPUT" 1 16384
  (( MUSE_OUTPUT <= MUSE_CONTEXT )) || muse_fail 'Output must be less than or equal to context.'
  muse_uint MUSE_PORT "$MUSE_PORT" 1024 65535
  muse_uint MUSE_PARALLEL "$MUSE_PARALLEL" 1 4
  export MUSE_CONTEXT MUSE_OUTPUT MUSE_PORT MUSE_PARALLEL
}

muse_multigpu_gpu() {
  [[ ${SLURM_GPUS_ON_NODE:-} == 2 ]] || muse_fail 'Request exactly two GPUs on this node (--gres=gpu:a100:2 or --gres=gpu:l40s:2).'
  [[ ${MUSE_GPU:-} == a100 || ${MUSE_GPU:-} == l40s ]] ||
    muse_fail 'Set MUSE_GPU=a100 or MUSE_GPU=l40s to select the intended GPU.'
  [[ ${CUDA_VISIBLE_DEVICES:-} =~ ^([0-9]+|GPU-[a-fA-F0-9-]+),([0-9]+|GPU-[a-fA-F0-9-]+)$ ]] ||
    muse_fail 'Exactly two Slurm-visible GPUs are required.'
  [[ ${CUDA_VISIBLE_DEVICES%,*} != "${CUDA_VISIBLE_DEVICES#*,}" ]] ||
    muse_fail 'Slurm-visible GPU identifiers must be distinct.'
  local info row
  local -a rows
  info=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name,memory.total --format=csv,noheader,nounits) ||
    muse_fail 'Cannot inspect allocated GPUs via nvidia-smi.'
  mapfile -t rows <<< "$info"
  [[ ${#rows[@]} == 2 ]] || muse_fail "Expected 2 allocated GPUs; found ${#rows[@]}."
  for row in "${rows[@]}"; do
    case $MUSE_GPU in
      a100) [[ $row == *A100*40GB* ]] || muse_fail "Expected A100 40GB; found $row" ;;
      l40s) [[ $row == *L40S* ]] || muse_fail "Expected L40S; found $row" ;;
    esac
  done
  echo "Allocated Dual GPUs:"
  echo "$info"
}
