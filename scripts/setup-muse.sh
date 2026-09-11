#!/usr/bin/env bash
# Install the tested Muse Glimmer runtime and OpenCode without administrator access.
set -euo pipefail
umask 077

if [[ -z ${SLURM_JOB_ID:-} ]]; then
  echo 'Run setup in a Slurm CPU allocation (see README).' >&2
  exit 1
fi
python3 -c 'import sys; assert sys.version_info[:2] == (3, 11), "Load anaconda3/2023.09-0 first"'
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
root="/data/${USER:?}/pinnacles-agents"
scratch="/scratch/$USER/pinnacles-agents"
export TMPDIR="$scratch/tmp"
export UV_CACHE_DIR="$scratch/uv-cache"
export UV_LINK_MODE=copy
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$root/tools/opencode-v1.18.30"

if [[ ! -x $root/bootstrap/bin/python ]]; then
  python3 -m venv "$root/bootstrap"
fi
"$root/bootstrap/bin/python" -m pip install 'uv==0.12.13'
if [[ ! -x $root/envs/muse-vllm/bin/python ]]; then
  "$root/bootstrap/bin/uv" venv --python python3 "$root/envs/muse-vllm"
fi
"$root/bootstrap/bin/uv" pip sync --python "$root/envs/muse-vllm/bin/python" \
  "$repo/env/muse-requirements.lock"
"$root/bootstrap/bin/uv" pip check --python "$root/envs/muse-vllm/bin/python"

opencode_dir="$root/tools/opencode-v1.18.30"
curl --fail --location --retry 3 --connect-timeout 30 \
  https://github.com/anomalyco/opencode/releases/download/v1.18.30/opencode-linux-x64.tar.gz \
  --output "$opencode_dir/opencode-linux-x64.tar.gz"
(cd "$opencode_dir" && echo '55007246858165496ff85ba1c2b648f7421e8e2013bf4189a680c9ff8e699d17  opencode-linux-x64.tar.gz' | sha256sum --check)
tar -xzf "$opencode_dir/opencode-linux-x64.tar.gz" -C "$opencode_dir" opencode

"$opencode_dir/opencode" --version
echo "Setup complete. Runtime: $root/envs/muse-vllm"
