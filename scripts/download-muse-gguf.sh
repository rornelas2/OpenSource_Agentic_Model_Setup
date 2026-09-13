#!/usr/bin/env bash
# Resume one official text GGUF and hash it on a CPU compute node.
set -euo pipefail
umask 077
source "$(dirname -- "${BASH_SOURCE[0]}")/muse-gguf-common.sh"
muse_allocation
[[ -z ${SLURM_JOB_GPUS:-} ]] || muse_fail 'Use a CPU allocation for downloads.'
mkdir -p "$muse_model"
exec 9>"$muse_model/.download.lock"
flock -n 9 || muse_fail 'Another download/verification is running.'
file="$muse_model/$MUSE_GGUF_FILE"
marker="$muse_model/.pinnacles-complete.json"
# Remove stale success before verification; a failed/interrupted transfer must
# never look complete. Existing completed weights are never overwritten.
rm -f -- "$marker"
if [[ ! -f $file ]]; then
  if [[ ! -f $file.part ]] || [[ $(stat -c %s "$file.part") != "$MUSE_GGUF_BYTES" ]]; then
    curl --fail --location --retry 3 --connect-timeout 30 --max-time 3600 \
      --continue-at - --output "$file.part" \
      "https://huggingface.co/$MUSE_GGUF_REPO/resolve/$MUSE_GGUF_REVISION/$MUSE_GGUF_FILE"
  fi
  candidate="$file.part"
else
  candidate="$file"
fi
[[ $(stat -c %s "$candidate") == "$MUSE_GGUF_BYTES" ]] || muse_fail 'Checkpoint size mismatch; no completion marker written.'
printf '%s  %s\n' "$MUSE_GGUF_SHA256" "$candidate" | sha256sum --check
if [[ $candidate != "$file" ]]; then mv -- "$candidate" "$file"; fi
chmod a-w "$file"
python3 - "$file" "$MUSE_GGUF_REVISION" "$MUSE_GGUF_SHA256" <<'PY'
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
muse_checkpoint
echo "Verified checkpoint: $file"
