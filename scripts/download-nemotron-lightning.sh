#!/usr/bin/env bash
# Download and verify NVIDIA Nemotron 3.5 Lightning 30B NVFP4 on a CPU compute node.
set -euo pipefail
umask 077
source "$(dirname -- "${BASH_SOURCE[0]}")/nemotron-lightning-common.sh"
nemotron_allocation
[[ -z ${SLURM_JOB_GPUS:-} ]] || nemotron_fail 'Use a CPU allocation for downloads.'

mkdir -p "$nemotron_model" "$nemotron_root/huggingface"
exec 9>"$nemotron_model/.download.lock"
flock -n 9 || nemotron_fail 'Another download/verification is running.'

marker="$nemotron_model/.pinnacles-complete.json"
if [[ -f $marker ]]; then
  if nemotron_checkpoint 2>/dev/null; then
    echo "Verified checkpoint already present: $nemotron_model"
    exit 0
  fi
fi

rm -f -- "$marker"
export HF_HOME="$nemotron_root/huggingface"
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0

"$nemotron_env/bin/hf" download "$NEMOTRON_LIGHTNING_REPO" \
  --revision "$NEMOTRON_LIGHTNING_REVISION" \
  --include '*.json' --include '*.jinja' --include '*.safetensors' \
  --local-dir "$nemotron_model"

# Verify all 52 shards and metadata against manifest, then write atomic completion marker
python3 - "$nemotron_model" "$nemotron_manifest" "$NEMOTRON_LIGHTNING_REVISION" <<'PY'
import hashlib, json, os, pathlib, sys, tempfile

model_dir = pathlib.Path(sys.argv[1])
manifest_path = pathlib.Path(sys.argv[2])
revision = sys.argv[3]

with manifest_path.open() as f:
    manifest = json.load(f)

total_bytes = 0
verified_files = {}

for filename, spec in manifest.items():
    shard_path = model_dir / filename
    if not shard_path.is_file():
        sys.exit(f"Missing expected shard: {filename}")
    size = shard_path.stat().st_size
    if size != spec["size"]:
        sys.exit(f"Size mismatch on {filename}: expected {spec['size']}, got {size}")
    expected_sha = spec["sha256"]
    hasher = hashlib.sha256()
    with shard_path.open("rb") as sf:
        while chunk := sf.read(8 * 1024 * 1024):
            hasher.update(chunk)
    actual_sha = hasher.hexdigest()
    if actual_sha != expected_sha:
        sys.exit(f"SHA-256 mismatch on {filename}: expected {expected_sha}, got {actual_sha}")
    st = shard_path.stat()
    total_bytes += size
    verified_files[filename] = {
        "size": size,
        "sha256": actual_sha,
        "mtime_ns": st.st_mtime_ns
    }

# Ensure all shards are read-only
for filename in manifest:
    (model_dir / filename).chmod(0o400)

marker_data = {
    "revision": revision,
    "shards": len(manifest),
    "bytes": total_bytes,
    "files": verified_files
}

fd, tmp = tempfile.mkstemp(dir=model_dir, prefix=".complete-")
try:
    with os.fdopen(fd, "w") as out:
        json.dump(marker_data, out, sort_keys=True)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp, model_dir / ".pinnacles-complete.json")
finally:
    if os.path.exists(tmp):
        os.unlink(tmp)

print(f"Verified all {len(manifest)} shards ({total_bytes} bytes).")
PY

printf '%s\n' "$NEMOTRON_LIGHTNING_REVISION" > "$nemotron_model/.pinnacles-revision"
nemotron_checkpoint
echo "Verified checkpoint: $nemotron_model"
