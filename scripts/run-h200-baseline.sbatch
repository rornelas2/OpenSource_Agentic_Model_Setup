#!/usr/bin/env bash
#SBATCH --job-name=gemma-bf16-benchmark
#SBATCH --partition=cenvalarc.gpu
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=results/benchmarks/h200-baseline-%j.out
#SBATCH --error=results/benchmarks/h200-baseline-%j.err

set -euo pipefail

# Slurm copies this script to its spool, so BASH_SOURCE is not the checkout.
# Submit from the repository root after creating results/benchmarks.
REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit this job from the repository root.}"
cd "$REPO_ROOT"
[[ -f benchmarks/server_runner.py ]] || { echo 'Submit from the repository root.' >&2; exit 1; }

echo "Starting Gemma BF16 unquantized baseline on H200 (Job ID: ${SLURM_JOB_ID})..."
nvidia-smi

# 1. Aider Refactor Benchmark
echo "--- Running Aider Refactor on gemma-bf16-h200 ---"
mkdir -p results/benchmarks/aider-gemma-bf16-pilot
export BENCHMARK_PREPARATION_MANIFEST="/data/${USER}/pinnacles-agents/evaluators/aider-refactor-preparation.json"

python3 benchmarks/server_runner.py \
  --profile gemma-bf16-h200 \
  --log-dir results/benchmarks/aider-gemma-bf16-pilot/server -- \
  python3 benchmarks/run.py generate \
    --profile gemma-bf16-h200 \
    --suite aider-refactor \
    --run-dir results/benchmarks/aider-gemma-bf16-pilot \
    --pilot

python3 benchmarks/run.py evaluate \
  --profile gemma-bf16-h200 \
  --suite aider-refactor \
  --run-dir results/benchmarks/aider-gemma-bf16-pilot

# 2. LiveCodeBench Benchmark
echo "--- Running LiveCodeBench on gemma-bf16-h200 ---"
mkdir -p results/benchmarks/lcb-gemma-bf16-pilot
export BENCHMARK_PREPARATION_MANIFEST="/data/${USER}/pinnacles-agents/evaluators/livecodebench-preparation.json"

python3 benchmarks/server_runner.py \
  --profile gemma-bf16-h200 \
  --log-dir results/benchmarks/lcb-gemma-bf16-pilot/server -- \
  python3 benchmarks/run.py generate \
    --profile gemma-bf16-h200 \
    --suite livecodebench \
    --run-dir results/benchmarks/lcb-gemma-bf16-pilot \
    --pilot

python3 benchmarks/run.py evaluate \
  --profile gemma-bf16-h200 \
  --suite livecodebench \
  --run-dir results/benchmarks/lcb-gemma-bf16-pilot

echo "--- Re-generating aggregate benchmark report ---"
python3 benchmarks/run.py report \
  --campaign results/benchmarks \
  --output benchmarks/results/aggregate

python3 benchmarks/run.py report \
  --campaign results/benchmarks \
  --output results/benchmarks/aggregate

echo "Gemma BF16 H200 baseline completed successfully!"
