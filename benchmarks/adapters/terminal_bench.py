"""Terminal-Bench 2.0 Harbor reward normalization and execution boundary.

Integrates Harbor evaluator with OpenCode agent, captures execution trajectories,
validates task verifier outputs, and constructs normalized trial records.
"""

from __future__ import annotations

import datetime
import json
import subprocess
import time
from numbers import Real
from pathlib import Path
from typing import Any

from benchmarks import SCHEMA_VERSION
from benchmarks.contracts import ContractError


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def normalize_reward(result: dict[str, Any]) -> float:
    if not isinstance(result, dict):
        raise ContractError("Harbor result must be an object")
    reward = result.get("reward")
    if isinstance(reward, bool) or not isinstance(reward, Real):
        raise ContractError("Harbor reward must be numeric")
    normalized = float(reward)
    if not 0.0 <= normalized <= 1.0:
        raise ContractError("Harbor reward must be in [0, 1]")
    return normalized


def run_harbor_verifier(verifier_path: Path, task_dir: Path, timeout: int = 300) -> dict[str, Any]:
    """Execute a local Harbor task verifier script and parse its output."""
    if not verifier_path.is_file():
        raise ContractError(f"verifier script not found: {verifier_path}")
    try:
        proc = subprocess.run(
            [str(verifier_path)],
            cwd=task_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        reward = 1.0 if proc.returncode == 0 else 0.0
        return {
            "reward": reward,
            "stdout": proc.stdout[:8192],
            "stderr": proc.stderr[:8192],
            "exit_code": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"reward": 0.0, "stdout": "", "stderr": "verifier timed out", "exit_code": -1}
    except Exception as exc:
        return {"reward": 0.0, "stdout": "", "stderr": str(exc), "exit_code": -1}


def build_trial_record(
    *,
    run_id: str,
    task_id: str,
    attempt_id: str,
    status: str,
    reason: str,
    started_utc: str,
    ended_utc: str,
    prompt_tokens: int | None = None,
    generated_tokens: int | None = None,
    token_source: str | None = "server_usage",
    finish_reason: str | None = "stop",
    http_status: int | None = 200,
    metrics: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    error_class: str | None = None,
) -> dict[str, Any]:
    """Construct a validated trial dictionary conforming to schemas/trial.schema.json."""
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "task_id": task_id,
        "attempt_id": attempt_id,
        "status": status,
        "reason": reason,
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "http_status": http_status,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "token_source": token_source,
        "metrics": metrics or {},
        "artifacts": artifacts or {},
        "error_class": error_class,
    }
