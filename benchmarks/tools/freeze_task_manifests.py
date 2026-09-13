#!/usr/bin/env python3
"""Freeze public task identities and content hashes at reviewed upstream pins.

This maintainer tool writes no prompts, labels, patches, tests, or solutions to
the repository. It requires the already reviewed dataset loader and explicit
local snapshots for the very large LiveCodeBench JSONL files and Terminal-Bench.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SWE_DATASET = "princeton-nlp/SWE-bench_Verified"
SWE_REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
LCB_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
MMLU_DATASET = "TIGER-Lab/MMLU-Pro"
MMLU_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
TERMINAL_REVISION = "2fd12b88aafdd04a52c298e3940bcb189f9766d6"
SEED = 20260911


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def tree_hash(files: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(files):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def write_records(path: Path, records: list[dict[str, str]]) -> str:
    if len(records) != len({record["task_id"] for record in records}):
        raise ValueError(f"duplicate task IDs for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
            output.write("\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose_stratified(records: list[dict[str, str]], count_per_stratum: int, limit: int | None = None) -> list[str]:
    groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        groups[record["stratum"]].append(record["task_id"])
    chosen: list[str] = []
    for stratum in sorted(groups):
        ids = sorted(groups[stratum], key=lambda task_id: canonical_hash([SEED, task_id]))
        chosen.extend(ids[:count_per_stratum])
    if limit is not None:
        chosen = sorted(chosen, key=lambda task_id: canonical_hash([SEED, task_id]))[:limit]
    return chosen


def freeze_swe() -> tuple[list[dict[str, str]], list[str]]:
    from datasets import load_dataset

    dataset = load_dataset(SWE_DATASET, revision=SWE_REVISION, split="test")
    records = []
    for row in dataset:
        records.append({
            "task_id": row["instance_id"],
            "input_sha256": canonical_hash({
                "repo": row["repo"], "base_commit": row["base_commit"],
                "problem_statement": row["problem_statement"],
            }),
            "grader_sha256": canonical_hash({
                "patch": row["patch"], "test_patch": row["test_patch"],
                "FAIL_TO_PASS": row["FAIL_TO_PASS"], "PASS_TO_PASS": row["PASS_TO_PASS"],
                "environment_setup_commit": row["environment_setup_commit"],
            }),
            "stratum": f"{row['repo']}|{row['difficulty']}",
        })
    repositories: dict[str, list[str]] = defaultdict(list)
    for record, row in zip(records, dataset):
        repositories[row["repo"]].append(record["task_id"])
    ranked_repositories = sorted(repositories, key=lambda repo: canonical_hash([SEED, repo]))[:5]
    pilot = [
        min(repositories[repo], key=lambda task_id: canonical_hash([SEED, repo, task_id]))
        for repo in ranked_repositories
    ]
    return records, pilot


def freeze_lcb(paths: list[Path]) -> tuple[list[dict[str, str]], list[str]]:
    records: list[dict[str, str]] = []
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
                records.append({
                    "task_id": row["question_id"],
                    "input_sha256": canonical_hash({
                        key: row[key] for key in (
                            "question_title", "question_content", "platform", "question_id",
                            "contest_id", "contest_date", "starter_code", "public_test_cases",
                        )
                    }),
                    "grader_sha256": canonical_hash({
                        "private_test_cases": row["private_test_cases"], "metadata": row["metadata"],
                    }),
                    "stratum": row["difficulty"],
                })
    groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        groups[record["stratum"]].append(record["task_id"])
    target = {"easy": 3, "medium": 4, "hard": 3}
    pilot: list[str] = []
    for stratum, count in target.items():
        ids = sorted(groups[stratum], key=lambda task_id: canonical_hash([SEED, task_id]))
        pilot.extend(ids[:count])
    return records, pilot


def freeze_mmlu() -> tuple[list[dict[str, str]], list[str], dict[str, list[str]]]:
    from datasets import load_dataset

    test = load_dataset(MMLU_DATASET, revision=MMLU_REVISION, split="test")
    validation = load_dataset(MMLU_DATASET, revision=MMLU_REVISION, split="validation")
    records = []
    for row in test:
        records.append({
            "task_id": str(row["question_id"]),
            "input_sha256": canonical_hash({
                "question_id": row["question_id"], "question": row["question"],
                "options": row["options"], "category": row["category"], "src": row["src"],
            }),
            "grader_sha256": canonical_hash({
                "answer": row["answer"], "answer_index": row["answer_index"],
            }),
            "stratum": row["category"],
        })
    pilot = choose_stratified(records, 2)
    shots: dict[str, list[str]] = defaultdict(list)
    for row in validation:
        shots[row["category"]].append(str(row["question_id"]))
    normalized_shots = {category: ids[:5] for category, ids in sorted(shots.items())}
    if any(len(ids) != 5 for ids in normalized_shots.values()):
        raise ValueError("MMLU-Pro validation does not provide five shots per subject")
    return records, pilot, normalized_shots


def freeze_terminal(root: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not (root / ".git").is_dir():
        raise ValueError("Terminal-Bench root must be a Git checkout")
    import subprocess

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        stdout=subprocess.PIPE, text=True,
    ).stdout.strip()
    if revision != TERMINAL_REVISION:
        raise ValueError(f"Terminal-Bench checkout is {revision}, expected {TERMINAL_REVISION}")
    records = []
    for task_file in sorted(root.glob("*/task.toml")):
        task_root = task_file.parent
        input_files = [task_file, task_root / "instruction.md", *task_root.glob("environment/**/*")]
        grader_files = [task_file, *task_root.glob("tests/**/*")]
        records.append({
            "task_id": task_root.name,
            "input_sha256": tree_hash(input_files, task_root),
            "grader_sha256": tree_hash(grader_files, task_root),
            "stratum": "terminal-bench-2.0",
        })
    pilot = [
        "cancel-async-tasks", "dna-assembly", "fix-code-vulnerability",
        "qemu-alpine-ssh", "raman-fitting",
    ]
    return records, pilot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--terminal-root", required=True, type=Path)
    parser.add_argument("--lcb-file", required=True, action="append", type=Path)
    parser.add_argument("--output", default=Path("benchmarks/suites"), type=Path)
    args = parser.parse_args()
    if len(args.lcb_file) != 6:
        parser.error("exactly six --lcb-file arguments are required for release_v6")

    suites: list[tuple[str, list[dict[str, str]], list[str]]] = []
    swe_records, swe_pilot = freeze_swe()
    suites.append(("swe-verified", swe_records, swe_pilot))
    terminal_records, terminal_pilot = freeze_terminal(args.terminal_root.resolve())
    suites.append(("terminal-bench", terminal_records, terminal_pilot))
    lcb_records, lcb_pilot = freeze_lcb([path.resolve() for path in args.lcb_file])
    suites.append(("livecodebench", lcb_records, lcb_pilot))
    mmlu_records, mmlu_pilot, mmlu_shots = freeze_mmlu()
    suites.append(("mmlu-pro", mmlu_records, mmlu_pilot))

    report: dict[str, Any] = {"seed": SEED, "suites": {}}
    for suite_id, records, pilot in suites:
        path = args.output / suite_id / "tasks.jsonl"
        digest = write_records(path, records)
        report["suites"][suite_id] = {
            "count": len(records), "sha256": digest, "pilot_task_ids": pilot,
        }
    report["mmlu_five_shot_ids"] = mmlu_shots
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
