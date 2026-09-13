# Benchmark Results Report

Generated from `/home/rornelas5/OpenSource_Agentic_Model_Setup/results/benchmarks/mmlu-gemma-bf16-pilot`. All evaluations are measured from stored trial records.

Pilot scores describe the scheduled subsets. Latencies include only trials with recorded timing; missing timings are not zero. See summary.json for sample counts, actual token budgets, hardware, and provenance hashes.

## Summary Table

| Profile | Suite | Split | Scheduled | Completed | Coverage | Pass Rate | 95% CI | Reward | Median Lat (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gemma-bf16-a100` | `mmlu-pro` | Pilot | 28 | 28 | 100.0% | 82.1% | [64.4%, 92.1%] | 0.821 | 8.283 |

## Status Breakdown

| Profile | Suite | Pass | Fail | Timeout | Cancelled | Infra Error | Skipped | Not Run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gemma-bf16-a100` | `mmlu-pro` | 23 | 5 | 0 | 0 | 0 | 0 | 0 |
