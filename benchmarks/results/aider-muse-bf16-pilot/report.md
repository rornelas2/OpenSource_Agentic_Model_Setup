# Benchmark Results Report

Generated from `/home/rornelas5/OpenSource_Agentic_Model_Setup/results/benchmarks/aider-muse-bf16-pilot`. All evaluations are measured from stored trial records.

Pilot scores describe the scheduled subsets. Latencies include only trials with recorded timing; missing timings are not zero. See summary.json for sample counts, actual token budgets, hardware, and provenance hashes.

## Summary Table

| Profile | Suite | Split | Scheduled | Completed | Coverage | Pass Rate | 95% CI | Reward | Median Lat (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `muse-bf16-a100` | `aider-refactor` | Pilot | 10 | 10 | 100.0% | 80.0% | [49.0%, 94.3%] | 0.800 | 110.743 |

## Status Breakdown

| Profile | Suite | Pass | Fail | Timeout | Cancelled | Infra Error | Skipped | Not Run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `muse-bf16-a100` | `aider-refactor` | 8 | 1 | 1 | 0 | 0 | 0 | 0 |
