#!/usr/bin/env bash
# Download and verify the official Google Gemma 4 31B IT QAT checkpoint on a CPU node.
set -euo pipefail
umask 077
source "$(dirname -- "${BASH_SOURCE[0]}")/gemma-qat-common.sh"
gemma_allocation
[[ -z ${SLURM_JOB_GPUS:-} ]] || gemma_fail 'Use a CPU allocation for downloads.'

mkdir -p "$gemma_model" "$gemma_root/huggingface"
exec 9>"$gemma_model/.download.lock"
flock -n 9 || gemma_fail 'Another download/verification is running.'

marker="$gemma_model/.pinnacles-complete.json"
file="$gemma_model/$GEMMA_QAT_FILE"

if [[ -f $marker && -f $file ]]; then
  if gemma_checkpoint 2>/dev/null; then
    echo "Verified checkpoint already present: $file"
    exit 0
  fi
fi

rm -f -- "$marker"
export HF_HOME="$gemma_root/huggingface"
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0

"$gemma_env/bin/hf" download "$GEMMA_QAT_REPO" \
  --revision "$GEMMA_QAT_REVISION" \
  --include '*.json' --include '*.jinja' --include '*.safetensors' \
  --local-dir "$gemma_model"

[[ -f $file ]] || gemma_fail "Expected weight file missing: $file"
[[ $(stat -c %s "$file") == "$GEMMA_QAT_BYTES" ]] || gemma_fail 'Checkpoint size mismatch; no completion marker written.'
printf '%s  %s\n' "$GEMMA_QAT_SHA256" "$file" | sha256sum --check
chmod a-w "$file"

python3 - "$file" "$GEMMA_QAT_REVISION" "$GEMMA_QAT_SHA256" <<'PY'
import json, os, pathlib, sys, tempfile
path = pathlib.Path(sys.argv[1])
with path.open('rb') as checkpoint:
    os.fsync(checkpoint.fileno())
stamp = path.stat()
marker = dict(revision=sys.argv[2], file=path.name, bytes=stamp.st_size,
              sha256=sys.argv[3], mtime_ns=stamp.st_mtime_ns)
fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.complete-')
try:
    with os.fdopen(fd, 'w') as out:
        json.dump(marker, out, sort_keys=True)
        out.write('\n')
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, path.parent / '.pinnacles-complete.json')
finally:
    if os.path.exists(tmp):
        os.unlink(tmp)
PY

printf '%s\n' "$GEMMA_QAT_REVISION" > "$gemma_model/.pinnacles-revision"
gemma_checkpoint
echo "Verified checkpoint: $file"
