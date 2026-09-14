#!/usr/bin/env bash
# Install the complete pinned Codex package on a Pinnacles CPU allocation.
set -euo pipefail
umask 077
[[ -n ${SLURM_JOB_ID:-} ]] || { echo 'Run in a CPU allocation (see README).' >&2; exit 1; }
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || { echo 'Linux x86-64 is required.' >&2; exit 1; }
root="/data/${USER:?}/pinnacles-agents"
destination="$root/tools/codex-v0.154.0"
mkdir -p "$root/tools"
exec 9>"$root/tools/.codex-install.lock"
flock -n 9 || { echo 'Another Codex installation is running.' >&2; exit 1; }
if [[ -d $destination ]]; then
  (cd "$destination" && sha256sum --quiet --check runtime.sha256)
  [[ $("$destination/bin/codex" --version) == 'codex-cli 0.154.0' ]]
  echo "Codex already installed: $destination"
  exit 0
fi
stage=$(mktemp -d "$root/tools/.codex-install.XXXXXX")
trap 'rm -rf -- "$stage"' EXIT
curl --fail --location --retry 3 --connect-timeout 30 --max-time 600 \
  https://github.com/openai/codex/releases/download/rust-v0.154.0/codex-package-x86_64-unknown-linux-musl.tar.gz \
  --output "$stage/package.tar.gz"
(cd "$stage" && echo 'fc6e3e3b85f2cf7d664520ee5c66a7fe4aa12bae7d46834f47e2f165fd0d6f78  package.tar.gz' | sha256sum --check)
mkdir "$stage/runtime"
tar -xzf "$stage/package.tar.gz" --no-same-owner -C "$stage/runtime"
[[ $("$stage/runtime/bin/codex" --version) == 'codex-cli 0.154.0' ]]
(cd "$stage/runtime" && find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$stage/runtime.sha256"
mv "$stage/runtime.sha256" "$stage/runtime/runtime.sha256"
mv "$stage/runtime" "$destination"
echo "Setup complete: $destination"
