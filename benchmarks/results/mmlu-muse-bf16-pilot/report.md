# Benchmark Results Report

Generated from `/home/rornelas5/OpenSource_Agentic_Model_Setup/results/benchmarks/mmlu-muse-bf16-pilot`. All evaluations are measured from stored trial records.

Pilot scores describe the scheduled subsets. Latencies include only trials with recorded timing; missing timings are not zero. See summary.json for sample counts, actual token budgets, hardware, and provenance hashes.

## Summary Table

| Profile | Suite | Split | Scheduled | Completed | Coverage | Pass Rate | 95% CI | Reward | Median Lat (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `muse-bf16-a100` | `mmlu-pro` | Pilot | 28 | 28 | 100.0% | 71.4% | [52.9%, 84.8%] | 0.714 | 2.511 |

## Status Breakdown

| Profile | Suite | Pass | Fail | Timeout | Cancelled | Infra Error | Skipped | Not Run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `muse-bf16-a100` | `mmlu-pro` | 20 | 8 | 0 | 0 | 0 | 0 | 0 |
