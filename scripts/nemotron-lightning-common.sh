# Internal helpers for the isolated Nemotron Lightning profile; source from its scripts.
nemotron_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$nemotron_repo/env/nemotron-lightning-pins.sh"
nemotron_root="/data/${USER:?}/pinnacles-agents"
nemotron_scratch="/scratch/$USER/pinnacles-agents/nemotron-lightning"
nemotron_env="$nemotron_root/envs/nemotron-vllm"
nemotron_model="$nemotron_root/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-$NEMOTRON_LIGHTNING_REVISION"
nemotron_manifest="$nemotron_repo/env/nemotron-lightning-manifest.json"
nemotron_fail() { echo "Nemotron Lightning: $*" >&2; exit 1; }

nemotron_allocation() {
  [[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || nemotron_fail 'Run inside a Slurm allocation.'
  grep -Eq "/job_${SLURM_JOB_ID}(/|$)" /proc/self/cgroup ||
    nemotron_fail 'This process is not in the allocated Slurm compute-node cgroup.'
  local job
  job=$(scontrol show job "$SLURM_JOB_ID" -o) || nemotron_fail 'Cannot verify Slurm job.'
  [[ " $job " == *' JobState=RUNNING '* && " $job " == *" UserId=$USER($(id -u)) "* ]] ||
    nemotron_fail 'The allocation must be running and owned by the current user.'
}

nemotron_uint() {
  [[ $2 =~ ^[1-9][0-9]{0,5}$ ]] || nemotron_fail "$1 must be a positive decimal integer."
  (( $2 >= $3 && $2 <= $4 )) || nemotron_fail "$1 must be in $3..$4."
}

nemotron_limits() {
  NEMOTRON_CONTEXT=${NEMOTRON_CONTEXT:-32768}
  NEMOTRON_OUTPUT=${NEMOTRON_OUTPUT:-8192}
  NEMOTRON_PORT=${NEMOTRON_PORT:-8000}
  NEMOTRON_PARALLEL=${NEMOTRON_PARALLEL:-1}
  nemotron_uint NEMOTRON_CONTEXT "$NEMOTRON_CONTEXT" 1 32768
  [[ $NEMOTRON_CONTEXT == 16384 || $NEMOTRON_CONTEXT == 32768 ]] ||
    nemotron_fail 'NEMOTRON_CONTEXT must be 16384 or 32768 (experimental until validated).'
  nemotron_uint NEMOTRON_OUTPUT "$NEMOTRON_OUTPUT" 1 8192
  (( NEMOTRON_OUTPUT <= NEMOTRON_CONTEXT )) || nemotron_fail 'Output must be less than or equal to context.'
  nemotron_uint NEMOTRON_PORT "$NEMOTRON_PORT" 1024 65535
  nemotron_uint NEMOTRON_PARALLEL "$NEMOTRON_PARALLEL" 1 8
  export NEMOTRON_CONTEXT NEMOTRON_OUTPUT NEMOTRON_PORT NEMOTRON_PARALLEL
}

nemotron_gpu() {
  [[ ${SLURM_GPUS_ON_NODE:-} == 1 ]] || nemotron_fail 'Request exactly one GPU on this node.'
  [[ ${NEMOTRON_GPU:-} == a100 || ${NEMOTRON_GPU:-} == l40s ]] ||
    nemotron_fail 'Set NEMOTRON_GPU=a100 or NEMOTRON_GPU=l40s to select the intended GPU.'
  [[ ${CUDA_VISIBLE_DEVICES:-} =~ ^([0-9]+|GPU-[a-fA-F0-9-]+)$ ]] ||
    nemotron_fail 'Exactly one Slurm-visible GPU is required.'
  local info
  info=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name,memory.total --format=csv,noheader,nounits) ||
    nemotron_fail 'Cannot inspect the allocated GPU.'
  [[ $info != *$'\n'* ]] || nemotron_fail 'Multiple GPUs were returned.'
  case $NEMOTRON_GPU in
    a100) [[ $info == *A100*40GB* ]] || nemotron_fail "Expected A100 40GB; found $info" ;;
    l40s) [[ $info == *L40S* ]] || nemotron_fail "Expected L40S; found $info" ;;
  esac
  echo "Allocated GPU: $info MiB"
}

nemotron_checkpoint() {
  python3 - "$nemotron_model" "$NEMOTRON_LIGHTNING_REVISION" "$NEMOTRON_LIGHTNING_SHARDS" "$NEMOTRON_LIGHTNING_BYTES" <<'PY'
import json, pathlib, sys
root, revision, shards_str, total_bytes_str = sys.argv[1:]
root_path = pathlib.Path(root)
rev_file = root_path / ".pinnacles-revision"
complete_file = root_path / ".pinnacles-complete.json"
if not rev_file.is_file() or rev_file.read_text().strip() != revision:
    sys.exit("Invalid or missing .pinnacles-revision")
if not complete_file.is_file():
    sys.exit("Missing .pinnacles-complete.json")
try:
    with complete_file.open() as f:
        meta = json.load(f)
    if meta.get("revision") != revision:
        sys.exit("Revision mismatch in completion marker")
    if int(meta.get("shards", 0)) != int(shards_str):
        sys.exit("Shard count mismatch in completion marker")
    if int(meta.get("bytes", 0)) != int(total_bytes_str):
        sys.exit("Total bytes mismatch in completion marker")
    shards = meta.get("files", {})
    if len(shards) != int(shards_str):
        sys.exit("Files count mismatch in marker")
    for shard_name, shard_info in shards.items():
        shard_path = root_path / shard_name
        if not shard_path.is_file():
            sys.exit(f"Missing shard file: {shard_name}")
        st = shard_path.stat()
        if st.st_size != shard_info.get("size"):
            sys.exit(f"Size mismatch for {shard_name}: expected {shard_info.get('size')}, got {st.st_size}")
        if shard_info.get("mtime_ns") and st.st_mtime_ns != shard_info.get("mtime_ns"):
            sys.exit(f"mtime mismatch for {shard_name}")
except Exception as e:
    sys.exit(f"Checkpoint verification failed: {e}")
PY
}

nemotron_runtime_check() {
  [[ -x "$nemotron_env/bin/vllm" ]] || nemotron_fail "vLLM not found in $nemotron_env"
}

nemotron_port_free() {
  python3 - "$NEMOTRON_PORT" <<'PY'
import socket, sys
port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(('127.0.0.1', port))
    except OSError:
        sys.exit(f"Nemotron Lightning: port {port} unavailable on 127.0.0.1")
PY
}
