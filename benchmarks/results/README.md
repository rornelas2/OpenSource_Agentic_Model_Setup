# Locally measured benchmark results

Fifteen measured pilot runs cover five serving profiles across LiveCodeBench,
Aider refactoring, and MMLU-Pro. Start with the [aggregate report](aggregate/report.md)
and [summary metadata](aggregate/summary.json). Protocols remain candidates;
these are subset measurements, not completed full-suite campaigns.

Raw responses, trajectories, patches, evaluator logs, hidden tests, task images,
and generated workspaces remain outside Git under the campaign's recorded raw
artifact location. Summaries retain coverage, failure classes, hardware/runtime
details, actual token budgets, and provenance hashes. Timing sample counts
make missing latency explicit. Offline recomputation requires the owner's raw
run journals and preparation manifests; these are not included in a fresh clone.

LiveCodeBench and Aider used 32K context / 8K output; MMLU-Pro used 16K / 4K.
BF16 runs used A100 80 GB GPUs with separate partition access; quantized runs
used A100 40 GB GPUs. See the [benchmark guide](../README.md) for grader limits,
the older LiveCodeBench fallback ambiguity, and interpretation cautions.
