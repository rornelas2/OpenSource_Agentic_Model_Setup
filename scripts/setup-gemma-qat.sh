#!/usr/bin/env bash
# Verify/prepare the isolated Gemma QAT runtime and OpenCode on a CPU compute node.
set -euo pipefail
umask 077
source "$(dirname -- "${BASH_SOURCE[0]}")/gemma-qat-common.sh"
gemma_allocation
[[ -z ${SLURM_JOB_GPUS:-} ]] || gemma_fail 'Use a CPU allocation for preparation.'

python3 -c 'import sys; assert sys.version_info[:2] == (3, 11), "Load anaconda3/2023.09-0 first"'

export TMPDIR="$gemma_scratch/tmp"
export UV_CACHE_DIR="$gemma_scratch/uv-cache"
export UV_LINK_MODE=copy
export PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p "$TMPDIR" "$UV_CACHE_DIR" "$gemma_root/tools/opencode-v1.18.30"

exec 9>"$gemma_root/tools/.gemma-qat-install.lock"
flock -n 9 || gemma_fail 'Another Gemma QAT runtime install is running.'

if [[ ! -x $gemma_root/bootstrap/bin/python ]]; then
  python3 -m venv "$gemma_root/bootstrap"
fi
if [[ ! -x $gemma_root/bootstrap/bin/uv ]]; then
  "$gemma_root/bootstrap/bin/python" -m pip install 'uv==0.12.13'
fi

if [[ ! -x $gemma_env/bin/python ]]; then
  "$gemma_root/bootstrap/bin/uv" venv --python python3 "$gemma_env"
fi
"$gemma_root/bootstrap/bin/uv" pip sync --python "$gemma_env/bin/python" \
  "$gemma_repo/env/gemma-requirements.lock"
"$gemma_root/bootstrap/bin/uv" pip check --python "$gemma_env/bin/python"

opencode_dir="$gemma_root/tools/opencode-v1.18.30"
if [[ ! -x $opencode_dir/opencode ]]; then
  opencode_stage=$(mktemp -d "$gemma_root/tools/.opencode-install.XXXXXX")
  trap 'rm -rf -- "$opencode_stage"' EXIT
  curl --fail --location --retry 3 --connect-timeout 30 --max-time 600 \
    https://github.com/anomalyco/opencode/releases/download/v1.18.30/opencode-linux-x64.tar.gz \
    --output "$opencode_stage/opencode-linux-x64.tar.gz"
  (cd "$opencode_stage" && echo '55007246858165496ff85ba1c2b648f7421e8e2013bf4189a680c9ff8e699d17  opencode-linux-x64.tar.gz' | sha256sum --check)
  tar -xzf "$opencode_stage/opencode-linux-x64.tar.gz" -C "$opencode_stage" opencode
  mv -- "$opencode_stage/opencode" "$opencode_dir/opencode"
fi
[[ $("$opencode_dir/opencode" --version) == 1.18.30 ]] || gemma_fail 'OpenCode 1.18.30 is required.'

jinja_file="$gemma_root/tools/tool_chat_template_gemma4.jinja"
if [[ ! -f $jinja_file ]]; then
  curl --fail --location --retry 3 --connect-timeout 30 \
    https://raw.githubusercontent.com/vllm-project/vllm/v0.29.0/examples/tool_chat_template_gemma4.jinja \
    --output "$jinja_file"
fi
(cd "$gemma_root/tools" && echo 'afdbb2abe3667ccde95cc2f86919f05370339399bab5f750950a4390523b8927  tool_chat_template_gemma4.jinja' | sha256sum --check)

"$gemma_root/bootstrap/bin/uv" pip list --python "$gemma_env/bin/python" | grep -q 'compressed-tensors' ||
  gemma_fail 'compressed-tensors package missing in gemma-vllm runtime.'

gemma_runtime_check
echo "Setup complete. Runtime: $gemma_env"
