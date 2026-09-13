"""Offline report generation from saved trial records without inference or network calls."""

from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from typing import Any

from benchmarks import SCHEMA_VERSION
from benchmarks.contracts import ContractError, load_all_suites, load_json, sha256_file
from benchmarks.state import atomic_write_json, read_trials, validate_trial


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    mid = len(sorted_v) // 2
    if len(sorted_v) % 2 == 1:
        return sorted_v[mid]
    return (sorted_v[mid - 1] + sorted_v[mid]) / 2.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = max(0, min(len(sorted_v) - 1, int(math.ceil(pct / 100.0 * len(sorted_v))) - 1))
    return sorted_v[idx]


def compute_metrics_for_run(
    run_dir: Path,
    suite_tasks: list[dict[str, Any]],
    is_pilot: bool,
    pilot_task_ids: list[str],
    scheduled_task_ids: list[str] | None = None,
    expected_run_id: str | None = None,
    require_complete: bool = False,
) -> dict[str, Any]:
    trials_path = run_dir / "trials.jsonl"
    if not trials_path.exists():
        raise ContractError(f"no trials.jsonl found in run directory: {run_dir}")

    records, recovered_partial = read_trials(trials_path)
    if recovered_partial:
        raise ContractError(f"unresolved partial trailing record in {trials_path}")

    # Validate all records
    seen_attempts: set[str] = set()
    for idx, rec in enumerate(records, 1):
        validate_trial(rec, f"{trials_path}:{idx}")
        if rec["attempt_id"] in seen_attempts:
            raise ContractError(f"duplicate attempt_id {rec['attempt_id']} in {trials_path}")
        seen_attempts.add(rec["attempt_id"])

    canonical_scheduled = pilot_task_ids if is_pilot else [t["task_id"] for t in suite_tasks]
    scheduled_ids = scheduled_task_ids if scheduled_task_ids is not None else canonical_scheduled
    if scheduled_ids != canonical_scheduled:
        raise ContractError("run manifest scheduled_task_ids do not match the frozen suite order")
    scheduled_set = set(scheduled_ids)
    denominator = len(scheduled_ids)
    if denominator == 0:
        raise ContractError("scheduled denominator is zero")

    status_counts = {
        "pass": 0,
        "fail": 0,
        "timeout": 0,
        "cancelled": 0,
        "infra_error": 0,
        "skipped": 0,
        "not_run": 0,
    }

    latencies: list[float] = []
    prompt_tokens_list: list[int] = []
    generated_tokens_list: list[int] = []
    rewards: list[float] = []

    completed_task_ids: set[str] = set()
    for rec in records:
        tid = rec["task_id"]
        if tid not in scheduled_set:
            raise ContractError(f"trial contains task outside the scheduled denominator: {tid}")
        if expected_run_id is not None and rec["run_id"] != expected_run_id:
            raise ContractError(f"stale trial run_id for task {tid}: {rec['run_id']!r}")
        if tid in completed_task_ids:
            raise ContractError(f"multiple terminal trials for scheduled task {tid}")
        completed_task_ids.add(tid)
        st = rec["status"]
        if st in status_counts:
            status_counts[st] += 1
        else:
            status_counts["fail"] += 1

        # Reward / pass
        if st == "pass":
            rewards.append(1.0)
        elif st == "fail":
            rewards.append(rec.get("metrics", {}).get("reward", 0.0))
        else:
            rewards.append(0.0)

        # Latency
        lat = rec.get("metrics", {}).get("elapsed_seconds")
        if lat is None:
            lat = rec.get("metrics", {}).get("latency_seconds")
        if lat is not None and isinstance(lat, (int, float)):
            latencies.append(float(lat))

        # Tokens
        if rec.get("prompt_tokens") is not None:
            prompt_tokens_list.append(rec["prompt_tokens"])
        if rec.get("generated_tokens") is not None:
            generated_tokens_list.append(rec["generated_tokens"])

    missing_count = len(scheduled_set - completed_task_ids)
    if require_complete and missing_count:
        missing = [task_id for task_id in scheduled_ids if task_id not in completed_task_ids]
        raise ContractError(f"incomplete required trial data; missing {len(missing)} tasks, first: {missing[:3]}")
    status_counts["not_run"] += missing_count

    coverage = len(completed_task_ids) / denominator
    pass_rate = status_counts["pass"] / denominator
    mean_reward = sum(rewards) / denominator

    # Compute Wilson 95% score interval for pass rate
    z = 1.96
    p = pass_rate
    denom_ci = 1.0 + (z**2) / denominator
    center = (p + (z**2) / (2 * denominator)) / denom_ci
    half_width = z * math.sqrt((p * (1 - p) / denominator) + (z**2) / (4 * (denominator**2))) / denom_ci
    ci_lower = max(0.0, center - half_width)
    ci_upper = min(1.0, center + half_width)

    return {
        "is_pilot": is_pilot,
        "scheduled_denominator": denominator,
        "completed_count": len(completed_task_ids),
        "coverage": round(coverage, 4),
        "pass_rate": round(pass_rate, 4),
        "accuracy": round(pass_rate, 4),
        "pass_rate_ci95": [round(ci_lower, 4), round(ci_upper, 4)],
        "mean_reward": round(mean_reward, 4),
        "status_counts": status_counts,
        "latency_seconds": {
            "sample_count": len(latencies),
            "missing_count": len(records) - len(latencies),
            "mean": round(mean(latencies), 3) if latencies else None,
            "median": round(median(latencies), 3) if latencies else None,
            "p95": round(percentile(latencies, 95), 3) if latencies else None,
        },
        "tokens": {
            "mean_prompt_tokens": round(mean(prompt_tokens_list), 1) if prompt_tokens_list else None,
            "mean_generated_tokens": round(mean(generated_tokens_list), 1) if generated_tokens_list else None,
            "total_generated_tokens": sum(generated_tokens_list),
        },
    }


def generate_report(
    campaign_or_run_dir: Path,
    output_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Offline aggregation of saved trial records. No inference or network."""
    suites_catalog = load_all_suites(repo_root / "benchmarks" / "suites", repo_root)

    # Find run directories: either directory itself has trials.jsonl or its subdirs do
    run_dirs: list[Path] = []
    if (campaign_or_run_dir / "trials.jsonl").exists():
        run_dirs.append(campaign_or_run_dir)
    else:
        for p in sorted(campaign_or_run_dir.iterdir()):
            if p.is_dir() and (p / "trials.jsonl").exists():
                run_dirs.append(p)

    if not run_dirs:
        raise ContractError(f"no valid benchmark run directories with trials.jsonl found in {campaign_or_run_dir}")

    summaries: list[dict[str, Any]] = []

    for rdir in run_dirs:
        manifest_file = rdir / "manifest.json"
        if not manifest_file.is_file():
            raise ContractError(f"run is missing its immutable manifest: {rdir}")
        meta = load_json(manifest_file)
        suite_id = meta.get("suite_id")
        profile_id = meta.get("profile_id")
        is_pilot = meta.get("pilot")
        run_id = meta.get("run_id")
        if not isinstance(profile_id, str) or not isinstance(run_id, str) or not isinstance(is_pilot, bool):
            raise ContractError(f"run manifest has invalid identity fields: {manifest_file}")
        if not suite_id or suite_id not in suites_catalog:
            raise ContractError(f"run manifest names unknown suite: {suite_id!r}")

        suite_obj, tasks = suites_catalog[suite_id]
        pilot_ids = suite_obj["pilot_task_ids"]

        if meta.get("suite", {}).get("task_manifest_sha256") not in {None, suite_obj["task_manifest"]["sha256"]}:
            raise ContractError(f"stale task manifest hash in {manifest_file}")
        if meta.get("suite", {}).get("protocol_sha256") not in {None, suite_obj["protocol"]["sha256"]}:
            raise ContractError(f"stale protocol hash in {manifest_file}")
        preparation_path = meta.get("preparation_manifest")
        preparation_hash = meta.get("preparation_manifest_sha256")
        if preparation_path is not None or preparation_hash is not None:
            if not isinstance(preparation_path, str) or not isinstance(preparation_hash, str):
                raise ContractError(f"run manifest has incomplete preparation provenance: {manifest_file}")
            if not Path(preparation_path).is_file() or sha256_file(Path(preparation_path)) != preparation_hash:
                raise ContractError(f"preparation manifest is missing or stale for run {run_id}")
        run_metrics = compute_metrics_for_run(
            rdir, tasks, is_pilot, pilot_ids,
            scheduled_task_ids=meta.get("scheduled_task_ids"), expected_run_id=run_id,
            require_complete=True,
        )
        run_metrics["run_id"] = run_id
        run_metrics["metric"] = suite_obj["scoring"]["metric"]
        run_metrics["run_dir"] = str(rdir.resolve())
        run_metrics["suite_id"] = suite_id
        run_metrics["profile_id"] = profile_id
        # Preserve the conditions that actually produced these scores.
        run_metrics["provenance"] = {key: meta.get(key) for key in (
            "model", "runtime", "hardware", "budgets", "sampling", "reasoning", "suite",
        )}
        run_metrics["manifest_sha256"] = sha256_file(manifest_file)
        run_metrics["trials_sha256"] = sha256_file(rdir / "trials.jsonl")
        summaries.append(run_metrics)

    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

    # Write summary.json
    summary_data = {
        "schema_version": SCHEMA_VERSION,
        "runs_count": len(summaries),
        "summaries": summaries,
    }
    atomic_write_json(output_dir / "summary.json", summary_data)

    # Write summary.csv
    csv_buffer = io.StringIO(newline="")
    writer = csv.writer(csv_buffer, lineterminator="\n")
    writer.writerow([
        "profile_id", "suite_id", "pilot", "scheduled", "completed", "coverage",
        "pass", "fail", "timeout", "cancelled", "infra_error", "skipped", "not_run",
        "pass_rate", "ci95_lower", "ci95_upper",
        "mean_reward", "median_latency_s", "mean_generated_tokens",
    ])
    for s in summaries:
        sc = s["status_counts"]
        ci = s["pass_rate_ci95"]
        lat = s["latency_seconds"]
        tok = s["tokens"]
        writer.writerow([
            s["profile_id"],
            s["suite_id"],
            s["is_pilot"],
            s["scheduled_denominator"],
            s["completed_count"],
            s["coverage"],
            sc["pass"],
            sc["fail"],
            sc["timeout"],
            sc["cancelled"],
            sc["infra_error"],
            sc["skipped"],
            sc["not_run"],
            s["pass_rate"],
            ci[0],
            ci[1],
            s["mean_reward"],
            lat["median"],
            tok["mean_generated_tokens"],
        ])
    _atomic_write_text(output_dir / "summary.csv", csv_buffer.getvalue())

    # Write report.md
    md_lines = [
        "# Benchmark Results Report",
        "",
        f"Generated from `{campaign_or_run_dir}`. All evaluations are measured from stored trial records.",
        "",
        "Pilot scores describe the scheduled subsets. Latencies include only trials with recorded timing; missing timings are not zero. See summary.json for sample counts, actual token budgets, hardware, and provenance hashes.",
        "",
        "## Summary Table",
        "",
        "| Profile | Suite | Split | Scheduled | Completed | Coverage | Pass Rate | 95% CI | Reward | Median Lat (s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in summaries:
        ci = s["pass_rate_ci95"]
        lat = s["latency_seconds"]
        split_label = "Pilot" if s["is_pilot"] else "Full Campaign"
        median_label = "N/A" if lat["median"] is None else str(lat["median"])
        md_lines.append(
            f"| `{s['profile_id']}` | `{s['suite_id']}` | {split_label} | {s['scheduled_denominator']} | "
            f"{s['completed_count']} | {s['coverage'] * 100:.1f}% | {s['pass_rate'] * 100:.1f}% | "
            f"[{ci[0]*100:.1f}%, {ci[1]*100:.1f}%] | {s['mean_reward']:.3f} | {median_label} |"
        )
    md_lines.extend([
        "",
        "## Status Breakdown",
        "",
        "| Profile | Suite | Pass | Fail | Timeout | Cancelled | Infra Error | Skipped | Not Run |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ])
    for s in summaries:
        sc = s["status_counts"]
        md_lines.append(
            f"| `{s['profile_id']}` | `{s['suite_id']}` | {sc['pass']} | {sc['fail']} | "
            f"{sc['timeout']} | {sc['cancelled']} | {sc['infra_error']} | {sc['skipped']} | "
            f"{sc['not_run']} |"
        )
    _atomic_write_text(output_dir / "report.md", "\n".join(md_lines) + "\n")
    return summary_data


def _atomic_write_text(path: Path, content: str) -> None:
    """Use the JSON atomic writer's durability contract for arbitrary text."""
    import os
    import tempfile

    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            output.write(content.encode("utf-8"))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
