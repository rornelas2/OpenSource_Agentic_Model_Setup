#!/usr/bin/env bash
# Verify/prepare the isolated Nemotron Lightning runtime and OpenCode on a CPU compute node.
set -euo pipefail
umask 077
source "$(dirname -- "${BASH_SOURCE[0]}")/nemotron-lightning-common.sh"
nemotron_allocation
[[ -z ${SLURM_JOB_GPUS:-} ]] || nemotron_fail 'Use a CPU allocation for preparation.'

python3 -c 'import sys; assert sys.version_info[:2] == (3, 11), "Load anaconda3/2023.09-0 first"'

export TMPDIR="$nemotron_scratch/tmp"
export UV_CACHE_DIR="$nemotron_scratch/uv-cache"
export UV_LINK_MODE=copy
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$nemotron_root/tools/opencode-v1.18.30"

exec 9>"$nemotron_root/tools/.nemotron-lightning-install.lock"
flock -n 9 || nemotron_fail 'Another Nemotron Lightning runtime install is running.'

if [[ ! -x $nemotron_root/bootstrap/bin/python ]]; then
  python3 -m venv "$nemotron_root/bootstrap"
fi
if [[ ! -x $nemotron_root/bootstrap/bin/uv ]]; then
  "$nemotron_root/bootstrap/bin/python" -m pip install 'uv==0.12.13'
fi

if [[ ! -x $nemotron_env/bin/python ]]; then
  "$nemotron_root/bootstrap/bin/uv" venv --python python3 "$nemotron_env"
fi
"$nemotron_root/bootstrap/bin/uv" pip sync --python "$nemotron_env/bin/python" \
  "$nemotron_repo/env/nemotron-requirements.lock"
"$nemotron_root/bootstrap/bin/uv" pip check --python "$nemotron_env/bin/python"

opencode_dir="$nemotron_root/tools/opencode-v1.18.30"
if [[ ! -x $opencode_dir/opencode ]]; then
  opencode_stage=$(mktemp -d "$nemotron_root/tools/.opencode-install.XXXXXX")
  trap 'rm -rf -- "$opencode_stage"' EXIT
  curl --fail --location --retry 3 --connect-timeout 30 --max-time 600 \
    https://github.com/anomalyco/opencode/releases/download/v1.18.30/opencode-linux-x64.tar.gz \
    --output "$opencode_stage/opencode-linux-x64.tar.gz"
  (cd "$opencode_stage" && echo '55007246858165496ff85ba1c2b648f7421e8e2013bf4189a680c9ff8e699d17  opencode-linux-x64.tar.gz' | sha256sum --check)
  tar -xzf "$opencode_stage/opencode-linux-x64.tar.gz" -C "$opencode_stage" opencode
  mv -- "$opencode_stage/opencode" "$opencode_dir/opencode"
fi
[[ $("$opencode_dir/opencode" --version) == 1.18.30 ]] || nemotron_fail 'OpenCode 1.18.30 is required.'

nemotron_runtime_check
echo "Setup complete. Runtime: $nemotron_env"
