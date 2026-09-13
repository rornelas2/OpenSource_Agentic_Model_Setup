# Benchmark Results Report

Generated from `/home/rornelas5/OpenSource_Agentic_Model_Setup/results/benchmarks`. All evaluations are measured from stored trial records.

Pilot scores describe the scheduled subsets. Latencies include only trials with recorded timing; missing timings are not zero. See summary.json for sample counts, actual token budgets, hardware, and provenance hashes.

## Summary Table

| Profile | Suite | Split | Scheduled | Completed | Coverage | Pass Rate | 95% CI | Reward | Median Lat (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gemma-bf16-a100` | `aider-refactor` | Pilot | 10 | 10 | 100.0% | 90.0% | [59.6%, 98.2%] | 0.900 | 106.162 |
| `gemma-qat-a100` | `aider-refactor` | Pilot | 10 | 10 | 100.0% | 90.0% | [59.6%, 98.2%] | 0.900 | 50.135 |
| `lightning-nvfp4-a100` | `aider-refactor` | Pilot | 10 | 10 | 100.0% | 60.0% | [31.3%, 83.2%] | 0.600 | 32.503 |
| `muse-bf16-a100` | `aider-refactor` | Pilot | 10 | 10 | 100.0% | 80.0% | [49.0%, 94.3%] | 0.800 | 110.743 |
| `muse-dynamic-a100` | `aider-refactor` | Pilot | 10 | 10 | 100.0% | 100.0% | [72.2%, 100.0%] | 1.000 | 85.633 |
| `gemma-bf16-a100` | `livecodebench` | Pilot | 10 | 10 | 100.0% | 90.0% | [59.6%, 98.2%] | 0.900 | 19.747 |
| `gemma-qat-a100` | `livecodebench` | Pilot | 10 | 10 | 100.0% | 80.0% | [49.0%, 94.3%] | 0.800 | 8.613 |
| `lightning-nvfp4-a100` | `livecodebench` | Pilot | 10 | 10 | 100.0% | 60.0% | [31.3%, 83.2%] | 0.600 | 4.16 |
| `muse-bf16-a100` | `livecodebench` | Pilot | 10 | 10 | 100.0% | 90.0% | [59.6%, 98.2%] | 0.900 | 51.599 |
| `muse-dynamic-a100` | `livecodebench` | Pilot | 10 | 10 | 100.0% | 80.0% | [49.0%, 94.3%] | 0.800 | 33.561 |
| `gemma-bf16-a100` | `mmlu-pro` | Pilot | 28 | 28 | 100.0% | 82.1% | [64.4%, 92.1%] | 0.821 | 8.283 |
| `gemma-qat-a100` | `mmlu-pro` | Pilot | 28 | 28 | 100.0% | 50.0% | [32.6%, 67.4%] | 0.500 | 4.5 |
| `lightning-nvfp4-a100` | `mmlu-pro` | Pilot | 28 | 28 | 100.0% | 60.7% | [42.4%, 76.4%] | 0.607 | 7.146 |
| `muse-bf16-a100` | `mmlu-pro` | Pilot | 28 | 28 | 100.0% | 71.4% | [52.9%, 84.8%] | 0.714 | 2.511 |
| `muse-dynamic-a100` | `mmlu-pro` | Pilot | 28 | 28 | 100.0% | 67.9% | [49.3%, 82.1%] | 0.679 | 2.607 |

## Status Breakdown

| Profile | Suite | Pass | Fail | Timeout | Cancelled | Infra Error | Skipped | Not Run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gemma-bf16-a100` | `aider-refactor` | 9 | 0 | 1 | 0 | 0 | 0 | 0 |
| `gemma-qat-a100` | `aider-refactor` | 9 | 0 | 1 | 0 | 0 | 0 | 0 |
| `lightning-nvfp4-a100` | `aider-refactor` | 6 | 1 | 3 | 0 | 0 | 0 | 0 |
| `muse-bf16-a100` | `aider-refactor` | 8 | 1 | 1 | 0 | 0 | 0 | 0 |
| `muse-dynamic-a100` | `aider-refactor` | 10 | 0 | 0 | 0 | 0 | 0 | 0 |
| `gemma-bf16-a100` | `livecodebench` | 9 | 1 | 0 | 0 | 0 | 0 | 0 |
| `gemma-qat-a100` | `livecodebench` | 8 | 1 | 1 | 0 | 0 | 0 | 0 |
| `lightning-nvfp4-a100` | `livecodebench` | 6 | 0 | 4 | 0 | 0 | 0 | 0 |
| `muse-bf16-a100` | `livecodebench` | 9 | 0 | 1 | 0 | 0 | 0 | 0 |
| `muse-dynamic-a100` | `livecodebench` | 8 | 0 | 2 | 0 | 0 | 0 | 0 |
| `gemma-bf16-a100` | `mmlu-pro` | 23 | 5 | 0 | 0 | 0 | 0 | 0 |
| `gemma-qat-a100` | `mmlu-pro` | 14 | 14 | 0 | 0 | 0 | 0 | 0 |
| `lightning-nvfp4-a100` | `mmlu-pro` | 17 | 9 | 2 | 0 | 0 | 0 | 0 |
| `muse-bf16-a100` | `mmlu-pro` | 20 | 8 | 0 | 0 | 0 | 0 | 0 |
| `muse-dynamic-a100` | `mmlu-pro` | 19 | 9 | 0 | 0 | 0 | 0 | 0 |
