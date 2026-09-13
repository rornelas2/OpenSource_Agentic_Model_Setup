# Benchmark harness

The catalog contains eight serving profiles and five benchmark suites. Fifteen
measured pilots cover Gemma BF16, Gemma QAT, Muse BF16, Muse GGUF, and Nemotron
Lightning across LiveCodeBench, Aider refactoring, and MMLU-Pro.

Start with the [measured results](results/aggregate/report.md). The
[JSON summary](results/aggregate/summary.json) records each run's actual budgets,
hardware, checkpoint/runtime revisions, timing coverage, and provenance hashes.
Raw responses, code, hidden tests, and journals remain in the ignored
`results/benchmarks/` directory.

## Understand the evaluation

| Suite | Pilot size | Recorded context / output budget | What runs |
|---|---:|---|---|
| LiveCodeBench release_v6 | 10 problems | 32,768 / 8,192 | Direct API generation, then Python test-case evaluation |
| Aider refactoring | 10 tasks | 32,768 / 8,192 | Direct API generation, then local AST checks |
| MMLU-Pro | 28 questions | 16,384 / 4,096 | Five examples per subject, then final-answer extraction |
| SWE-bench Verified | No measured pilot | See suite catalog | OpenCode in task containers; blocked prerequisites |
| Terminal-Bench 2.0 | No measured pilot | See suite catalog | Task containers and verifier; blocked prerequisites |

All protocols remain `candidate`; full campaigns are blocked until the pilot
protocol is reviewed and deliberately frozen. These are local subset results,
not official full-suite leaderboard scores. The three measured suites use no
OpenCode tools during generation.

Aider's local grader checks syntax, a top-level function, and approximate AST
node counts. It does not establish semantic equivalence, verify delegation, or
run repository tests. LiveCodeBench executes generated Python in evaluator
subprocesses. Those subprocesses and Slurm resource limits are **not a security
sandbox**: generated code can inherit the account's filesystem/network access.
Review saved code and use an appropriately isolated execution environment before
grading unfamiliar outputs.

The configured LiveCodeBench evaluator must succeed; a missing or failed
evaluator is recorded as an infrastructure error rather than silently switching
to a different grader. Earlier recorded pilots predate that correction and do
not record which evaluator path was taken, so their evaluator-path provenance
cannot be established from the summaries alone.

## Run lightweight checks

From the repository root:

```bash
python3 -m unittest discover -s benchmarks/tests -v
python3 -m compileall -q benchmarks scripts
python3 benchmarks/run.py catalog
python3 benchmarks/run.py generate --profile gemma-qat-a100 --suite mmlu-pro --run-dir /tmp/mmlu-plan --pilot --dry-run
```

A dry run validates inputs and plans work without submitting jobs, installing
dependencies, contacting a model, or creating a run directory. Real preparation,
generation, and evaluation belong in Slurm compute allocations.

## Prepare and run an MMLU-Pro pilot

Complete the model's setup and download in the main tutorial first. From the
repository root on the login node, prepare the pinned evaluator and dataset
in a CPU allocation:

```bash
export BENCHMARK_PREPARATION_MANIFEST="/data/$USER/pinnacles-agents/evaluators/mmlu-pro-preparation.json"
srun --partition=short --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --mem=32G --time=01:00:00 \
  python3 benchmarks/run.py prepare --suite mmlu-pro \
    --manifest "$BENCHMARK_PREPARATION_MANIFEST"
```

Preparation validates the pinned sources, package lock, dataset hashes, and
private/public input separation. Wait for successful completion. Use a fresh
run directory; existing server logs and immutable run manifests are not
overwritten.

Start Gemma QAT and generate the 28 scheduled answers:

```bash
srun --partition=gpu --gres=gpu:a100:1 --nodes=1 --ntasks=1 \
  --cpus-per-task=8 --mem=128G --time=00:45:00 \
  python3 benchmarks/server_runner.py \
    --profile gemma-qat-a100 \
    --log-dir results/benchmarks/mmlu-gemma-qat-new/server -- \
    python3 benchmarks/run.py generate \
      --profile gemma-qat-a100 --suite mmlu-pro \
      --run-dir results/benchmarks/mmlu-gemma-qat-new --pilot
```

The runner selects the GPU type and serving limits from the profile, owns the
server process group, checks the loopback listener/model alias, and stops its
server when the client exits. The suite can request a smaller token budget than
the server ceiling: MMLU-Pro requests 16K/4K even when a profile serves 32K/8K.

Grade saved answers in a CPU allocation and generate the compact report:

```bash
srun --partition=short --nodes=1 --ntasks=1 --cpus-per-task=4 \
  --mem=32G --time=00:30:00 \
  python3 benchmarks/run.py evaluate \
    --profile gemma-qat-a100 --suite mmlu-pro \
    --run-dir results/benchmarks/mmlu-gemma-qat-new

python3 benchmarks/run.py report \
  --campaign results/benchmarks/mmlu-gemma-qat-new \
  --output benchmarks/results/mmlu-gemma-qat-new
```

Keep `BENCHMARK_PREPARATION_MANIFEST` exported through both stages. For Aider
or LiveCodeBench, prepare that suite separately and change the suite ID,
preparation path, and run directory together. Review LiveCodeBench's execution
boundary above before grading its generated code.

Resume an interrupted generation with identical inputs and `--resume`.
Use a new server log directory for the restarted server while retaining the
original run directory. Terminal failures are not silently retried.

## Match the allocation to the profile

| Profile | Partition | GRES | System RAM |
|---|---|---|---|
| `gemma-bf16-h200` | `cenvalarc.gpu` | `gpu:nvidia_h200_nvl:1` | 128G |
| `muse-bf16-h200` | `cenvalarc.gpu` | `gpu:nvidia_h200_nvl:1` | 128G |
| `super-nvfp4-h200` | `cenvalarc.gpu` | `gpu:nvidia_h200_nvl:1` | 192G |
| `gemma-bf16-a100` | `dept.appliedmath` | `gpu:a100:1` | 128G |
| `muse-bf16-a100` | `dept.appliedmath` | `gpu:a100:1` | 128G |
| `muse-dynamic-a100` | `gpu` | `gpu:a100:1` | 128G |
| `gemma-qat-a100` | `gpu` | `gpu:a100:1` | 128G |
| `lightning-nvfp4-a100` | `gpu` | `gpu:a100:1` | 128G |

The BF16 A100 profiles require 80 GB cards and authorized access to
`dept.appliedmath`; regular sponsored accounts are not assumed to have it.
The two-GPU 128K tutorial launchers are separate from this one-GPU benchmark
catalog. No 128K benchmark results are reported here.

Muse BF16 serving prefers its completed data copy, with scratch as a fallback.
Set `MUSE_MODEL_DIR` explicitly when using a different complete checkpoint
directory. The download script honors the same variable. Scratch is purgeable.

The optional `scripts/run-h200-baseline.sbatch` is an H200 convenience script,
not the source of the recorded A100 baselines. It has not been rerun during
this review. It must be submitted from the repository root after creating
`results/benchmarks` so Slurm can open its output files.

## Interpret scores and timing

Pass rates use all scheduled tasks, including timeout and infrastructure
failures. A terminal record counts toward coverage even if it failed. Wilson
95% intervals express the uncertainty of small counts; they do not account
for task selection, hardware, or evaluator differences.

The `timeout` category includes output-token exhaustion as well as wall-clock
timeouts. Inspect `finish_reason` and the saved generation record to distinguish
them. Earlier generation-timeout trials omitted latency, so each report exposes
the number of measured and missing timings. Medians exclude missing timings,
startup, and offline grading. Future evaluations retain available generation
latency even when generation reaches its limit.

Quantized and BF16 profiles ran on different A100 capacities; Muse also changes
runtime and prompt handling between BF16 and GGUF. These pilots cannot isolate
the causal effect of quantization. A one-task difference in the 10-task coding
pilots is 10 percentage points; one MMLU-Pro answer is about 3.6 points.

## SWE-bench and Terminal-Bench prerequisites

These suites remain unverified on Pinnacles. The inspected account lacks the
subordinate UID/GID mappings needed for rootless Podman, and a working
Docker-compatible task environment has not been established.

The pinned SWE evaluator
`02e7a74ffd0b707aab73d203fe87bdc7c76afc8e` also expects fields absent from the
pinned dataset `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`. Resolve the
evaluator/dataset pair deliberately, regenerate and review task hashes, and
validate the task container, endpoint relay, and known-good/known-bad fixtures
before enabling execution. Existing candidate code and passing unit tests do
not establish an operational recipe for these suites.
