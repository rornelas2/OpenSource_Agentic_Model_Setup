"""Resolve immutable benchmark work without launching models or jobs."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from benchmarks import SCHEMA_VERSION
from benchmarks.contracts import ContractError, sha256_file
from benchmarks.state import pending_task_ids, read_trials


def repository_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10,
    )
    return result.stdout.strip()


def input_hashes(profile: dict[str, Any], suite: dict[str, Any], profiles_path: Path) -> dict[str, str]:
    files: list[Path] = [
        profiles_path.resolve(), suite["_manifest_path"], suite["_task_path"],
        suite["_protocol_path"],
    ]
    files.extend(path for path in profile["_resolved_files"].values() if path is not None)
    benchmark_root = profiles_path.resolve().parent
    files.extend(
        path for path in benchmark_root.rglob("*.py")
        if "tests" not in path.relative_to(benchmark_root).parts
        and "__pycache__" not in path.parts
    )
    files.extend((benchmark_root / "schemas").glob("*.json"))
    return {str(path): sha256_file(path) for path in sorted(set(files))}


def build_plan(
    *, repo_root: Path, profiles_path: Path, profile: dict[str, Any], suite: dict[str, Any],
    tasks: list[dict[str, Any]], run_dir: Path, pilot: bool, resume: bool,
) -> dict[str, Any]:
    selected_ids = suite["pilot_task_ids"] if pilot else [task["task_id"] for task in tasks]
    recovered_partial = False
    prior_records = []
    if resume:
        prior_records, recovered_partial = read_trials(run_dir / "trials.jsonl")
        known_task_ids = {task["task_id"] for task in tasks}
        unknown = sorted({record["task_id"] for record in prior_records} - known_task_ids)
        if unknown:
            raise ContractError(f"resume state contains task IDs outside {suite['id']}: {unknown[:3]}")
        run_ids = {record["run_id"] for record in prior_records}
        if len(run_ids) > 1:
            raise ContractError(f"resume state contains multiple run IDs: {sorted(run_ids)}")
        selected_ids = pending_task_ids(selected_ids, prior_records)
    common_context = int(profile["deployment"]["context_tokens"])
    common_output = int(profile["deployment"]["output_tokens"])
    execution_allowed = pilot or suite["protocol_status"] == "frozen"
    return {
        "schema_version": SCHEMA_VERSION,
        "operation": "generate",
        "dry_run": True,
        "profile_id": profile["id"],
        "suite_id": suite["id"],
        "protocol_status": suite["protocol_status"],
        "pilot": pilot,
        "resume": resume,
        "execution_allowed_after_dry_run": execution_allowed,
        "execution_block_reason": None if execution_allowed else "full execution requires a pilot-validated frozen protocol",
        "run_dir": str(run_dir.resolve()),
        "repository_commit": repository_commit(repo_root),
        "input_sha256": input_hashes(profile, suite, profiles_path),
        "endpoint": profile["endpoint"],
        "resource_request": {
            "partition": profile["deployment"]["partition"],
            "gres": profile["deployment"]["gres"],
            "cpus": 8,
            "system_ram_gib": profile["deployment"]["system_ram_gib"],
        },
        "request_budget": {
            **suite["budget"],
            "effective_context_tokens": common_context,
            "effective_output_tokens": common_output,
        },
        "scheduled_count": len(suite["pilot_task_ids"] if pilot else tasks),
        "prior_terminal_count": len(prior_records),
        "pending_count": len(selected_ids),
        "pending_task_ids": selected_ids,
        "recovered_partial_trailing_record": recovered_partial,
        "side_effects": [],
    }


def ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)
