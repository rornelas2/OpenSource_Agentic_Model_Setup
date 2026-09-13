"""SWE-bench Verified prediction serialization and evaluation boundary.

Supports bounded OpenCode patch generation, patch extraction, official
evaluator execution, oracle/bad fixture validation, and trial normalization.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from benchmarks import SCHEMA_VERSION
from benchmarks.contracts import ContractError, sha256_file
from benchmarks.state import atomic_write_json


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def prediction(instance_id: str, model_patch: str, model_name_or_path: str) -> dict[str, str]:
    for name, value in {
        "instance_id": instance_id,
        "model_patch": model_patch,
        "model_name_or_path": model_name_or_path,
    }.items():
        if not isinstance(value, str):
            raise ContractError(f"SWE-bench prediction {name} must be a string")
    if not instance_id or not model_name_or_path:
        raise ContractError("SWE-bench instance and model identifiers must be nonempty")
    return {
        "instance_id": instance_id,
        "model_patch": model_patch,
        "model_name_or_path": model_name_or_path,
    }


def normalize_report(instance_id: str, report: dict[str, Any]) -> bool:
    """Normalize either the official per-instance or schema-v2 run report."""
    if not isinstance(report, dict):
        raise ContractError("official SWE-bench report must be an object")
    if instance_id in report:
        item = report[instance_id]
        if not isinstance(item, dict) or not isinstance(item.get("resolved"), bool):
            raise ContractError("official SWE-bench per-instance report requires boolean resolved")
        return item["resolved"]
    if report.get("schema_version") == 2:
        list_fields = (
            "submitted_ids", "completed_ids", "resolved_ids", "unresolved_ids",
            "empty_patch_ids", "error_ids", "infra_failure_ids", "ambiguous_failure_ids",
        )
        for field in list_fields:
            if not isinstance(report.get(field), list) or not all(isinstance(v, str) for v in report[field]):
                raise ContractError(f"official SWE-bench run report has invalid {field}")
        categories = [
            field for field in ("resolved_ids", "unresolved_ids", "empty_patch_ids", "error_ids")
            if instance_id in report[field]
        ]
        if len(categories) != 1:
            raise ContractError(f"official SWE-bench report has ambiguous/missing outcome for {instance_id}")
        return categories[0] == "resolved_ids"
    raise ContractError(f"official report is missing {instance_id}")


def evaluate_patch(
    *,
    instance_id: str,
    patch: str,
    model_name_or_path: str = "benchmark-model",
    evaluator_dataset_path: Path | None = None,
    evaluator_env_bin: Path | None = None,
    evaluator_src_dir: Path | None = None,
    run_dir: Path | None = None,
    timeout: int = 1800,
    cpus: int = 8,
    memory_gib: int = 64,
    pids_limit: int = 4096,
    use_gold_prediction: bool = False,
) -> tuple[bool, str, dict[str, Any]]:
    """Grade patch using official SWE-bench harness.

    If patch is empty, returns (False, 'unsolved_empty_patch').
    """
    if not use_gold_prediction and not patch.strip():
        return False, "unsolved_empty_patch", {"empty_patch": True}

    if evaluator_env_bin is None or not evaluator_env_bin.is_file():
        # Evaluator environment not available
        return False, "evaluator_infra_environment_unavailable", {}
    if evaluator_dataset_path is None or not evaluator_dataset_path.is_file():
        return False, "evaluator_infra_pinned_dataset_unavailable", {}

    eval_run_dir = run_dir or Path(f"/tmp/swe-eval-{instance_id}")
    eval_run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    pred_file = eval_run_dir / "predictions.jsonl"
    predictions_path = "gold"
    if not use_gold_prediction:
        pred_record = prediction(instance_id, patch, model_name_or_path)
        atomic_write_json(pred_file, pred_record)
        predictions_path = str(pred_file)

    wrapper = Path(__file__).resolve().parent.parent / "swe_evaluator_runner.py"
    if not wrapper.is_file():
        return False, "evaluator_infra_wrapper_unavailable", {}
    run_id = f"eval-{instance_id}-{hashlib.sha256((patch + str(use_gold_prediction)).encode()).hexdigest()[:16]}"

    cmd = [
        str(evaluator_env_bin),
        str(wrapper),
        "--dataset-name",
        str(evaluator_dataset_path),
        "--split",
        "test",
        "--instance-id",
        instance_id,
        "--predictions-path",
        predictions_path,
        "--run-id",
        run_id,
        "--timeout",
        str(timeout),
        "--cpus",
        str(cpus),
        "--memory-gib",
        str(memory_gib),
        "--pids-limit",
        str(pids_limit),
    ]

    try:
        proc = subprocess.run(
            cmd,
            cwd=eval_run_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        native = {
            "exit_code": proc.returncode,
            "stdout": proc.stdout[-8192:],
            "stderr": proc.stderr[-8192:],
            "command": cmd,
            "wrapper_sha256": sha256_file(wrapper),
            "resource_limits": {
                "cpus": cpus,
                "memory_gib": memory_gib,
                "pids_limit": pids_limit,
                "network_disabled": True,
                "timeout_seconds": timeout,
            },
        }
        if proc.returncode != 0:
            return False, "evaluator_infra_nonzero_exit", native
        report_file = eval_run_dir / "logs" / "evaluation" / run_id / "results.json"
        if not report_file.is_file():
            native["expected_report"] = str(report_file)
            return False, "evaluator_infra_missing_report", native
        report_data = json.loads(report_file.read_text(encoding="utf-8"))
        is_resolved = normalize_report(instance_id, report_data)
        native["report_path"] = str(report_file.resolve())
        native["report"] = report_data
        if instance_id in report_data.get("infra_failure_ids", []):
            return False, "evaluator_infra_reported", native
        if instance_id in report_data.get("ambiguous_failure_ids", []):
            return False, "evaluator_infra_ambiguous", native
        if instance_id in report_data.get("error_ids", []):
            return False, "evaluator_infra_error", native
        return is_resolved, "official_resolved" if is_resolved else "official_unresolved", native
    except subprocess.TimeoutExpired:
        return False, "evaluator_timeout", {}
    except Exception as exc:
        return False, f"evaluator_infra_exception:{type(exc).__name__}", {"detail": str(exc)}


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
