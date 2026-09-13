#!/usr/bin/env bash
# Build an isolated pinned CUDA runtime on a CPU compute node.
set -euo pipefail
umask 077
source "$(dirname -- "${BASH_SOURCE[0]}")/muse-gguf-common.sh"
muse_allocation
[[ -z ${SLURM_JOB_GPUS:-} ]] || muse_fail 'Use a CPU allocation for preparation.'
mkdir -p "$muse_root/tools" "$muse_scratch"
exec 9>"$muse_root/tools/.muse-gguf-install.lock"
flock -n 9 || muse_fail 'Another Muse GGUF runtime install is running.'
opencode_dir="$muse_root/tools/opencode-v1.18.30"
if [[ ! -d $opencode_dir ]]; then
  opencode_stage=$(mktemp -d "$muse_root/tools/.opencode-install.XXXXXX")
  trap 'rm -rf -- "$opencode_stage"' EXIT
  curl --fail --location --retry 3 --connect-timeout 30 --max-time 600 \
    https://github.com/anomalyco/opencode/releases/download/v1.18.30/opencode-linux-x64.tar.gz \
    --output "$opencode_stage/opencode-linux-x64.tar.gz"
  (cd "$opencode_stage" && echo '55007246858165496ff85ba1c2b648f7421e8e2013bf4189a680c9ff8e699d17  opencode-linux-x64.tar.gz' | sha256sum --check)
  tar -xzf "$opencode_stage/opencode-linux-x64.tar.gz" -C "$opencode_stage" opencode
  mv -- "$opencode_stage" "$opencode_dir"
  trap - EXIT
fi
[[ $("$opencode_dir/opencode" --version) == 1.18.30 ]] || muse_fail 'OpenCode 1.18.30 is required.'
if [[ -d $muse_runtime ]]; then
  muse_runtime_check
  echo "Pinned runtime already installed: $muse_runtime"
  exit 0
fi
module load "$MUSE_CUDA_MODULE"
[[ -x $MUSE_CMAKE ]] || muse_fail "Campus CMake missing: $MUSE_CMAKE"
command -v nvcc gcc g++ git make >/dev/null || muse_fail 'Missing CUDA/compiler/build prerequisite.'
nvcc --version
gcc --version
"$MUSE_CMAKE" --version
threads=${SLURM_CPUS_PER_TASK:-8}
muse_uint SLURM_CPUS_PER_TASK "$threads" 1 64
source_dir="$muse_scratch/llama.cpp-$MUSE_LLAMA_COMMIT"
if [[ ! -d $source_dir ]]; then
  git clone --depth 1 --branch "$MUSE_LLAMA_TAG" https://github.com/ggml-org/llama.cpp.git "$source_dir"
fi
[[ $(git -C "$source_dir" rev-parse HEAD) == "$MUSE_LLAMA_COMMIT" ]] || muse_fail 'Unexpected runtime source commit.'
[[ -z $(git -C "$source_dir" status --porcelain --untracked-files=no) ]] || muse_fail 'Runtime source has local edits.'
"$MUSE_CMAKE" -S "$source_dir" -B "$source_dir/build" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES='80;89' \
  -DGGML_NATIVE=OFF -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
  -DCMAKE_BUILD_RPATH="$CUDA_HOME/lib64" \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF
"$MUSE_CMAKE" --build "$source_dir/build" --target llama-server -j "$threads"
stage=$(mktemp -d "$muse_root/tools/.llama.cpp-install.XXXXXX")
trap 'rm -rf -- "$stage"' EXIT
cp -a "$source_dir/build/bin" "$stage/bin"
cp "$source_dir/LICENSE" "$stage/LICENSE"
printf '%s\n' "$MUSE_LLAMA_COMMIT" > "$stage/.pinnacles-commit"
printf '%s\n' 'cuda13-sm80-sm89-no-ui-v1' > "$stage/.pinnacles-build-profile"
(cd "$stage" && find bin -type f -print0 | sort -z | xargs -0 sha256sum > runtime.sha256)
# CPU nodes lack libcuda.so.1. Do not inject driver stubs into the installed
# runtime or execute inference here; inspect --version/--help on the GPU node.
readelf -d "$stage/bin/libggml-cuda.so" | grep -q 'libcudart.so.13' ||
  muse_fail 'Built library does not link the expected CUDA 13 runtime.'
cp "$source_dir/build/CMakeCache.txt" "$stage/build-cache.txt"
mv -- "$stage" "$muse_runtime"
trap - EXIT
muse_runtime_check
echo "Setup complete: $muse_runtime"
