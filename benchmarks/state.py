"""Append-only trial state with crash-tolerant trailing-record recovery."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

from benchmarks import SCHEMA_VERSION
from benchmarks.contracts import ContractError, canonical_json_bytes


TERMINAL_STATUSES = frozenset({
    "pass", "fail", "timeout", "cancelled", "infra_error", "skipped", "not_run",
})
MAX_TRIAL_RECORD_BYTES = 1024 * 1024
TRIAL_FIELDS = frozenset({
    "schema_version", "run_id", "task_id", "attempt_id", "status", "reason",
    "started_utc", "ended_utc", "http_status", "finish_reason", "prompt_tokens",
    "generated_tokens", "token_source", "metrics", "artifacts", "error_class",
})


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_trial(record: Any, where: str = "trial") -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ContractError(f"{where} must be an object")
    unknown = sorted(record.keys() - TRIAL_FIELDS)
    missing = sorted(TRIAL_FIELDS - record.keys())
    if missing:
        raise ContractError(f"{where} missing fields: {', '.join(missing)}")
    if unknown:
        raise ContractError(f"{where} has unknown fields: {', '.join(unknown)}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ContractError(f"{where} has unsupported schema_version")
    for key in ("run_id", "task_id", "attempt_id", "status", "reason", "started_utc", "ended_utc"):
        if not isinstance(record[key], str) or not record[key]:
            raise ContractError(f"{where}.{key} must be nonempty")
    if record["status"] not in TERMINAL_STATUSES:
        raise ContractError(f"{where}.status is not terminal")
    for key in ("http_status", "prompt_tokens", "generated_tokens"):
        if record[key] is not None and (isinstance(record[key], bool) or not isinstance(record[key], int) or record[key] < 0):
            raise ContractError(f"{where}.{key} must be null or a nonnegative integer")
    for key in ("finish_reason", "token_source", "error_class"):
        if record[key] is not None and not isinstance(record[key], str):
            raise ContractError(f"{where}.{key} must be null or a string")
    if not isinstance(record["metrics"], dict) or not isinstance(record["artifacts"], dict):
        raise ContractError(f"{where}.metrics and artifacts must be objects")
    return record


def read_trials(path: Path) -> tuple[list[dict[str, Any]], bool]:
    """Return complete records and whether a partial final record was ignored."""
    if not path.exists():
        return [], False
    records: list[dict[str, Any]] = []
    recovered = False
    seen_attempts: set[str] = set()
    try:
        with path.open("rb") as source:
            index = 0
            while True:
                raw_line = source.readline(MAX_TRIAL_RECORD_BYTES + 1)
                if not raw_line:
                    break
                index += 1
                if len(raw_line) > MAX_TRIAL_RECORD_BYTES:
                    raise ContractError(f"{path}:{index} exceeds the 1 MiB trial-record limit")
                if not raw_line.endswith((b"\n", b"\r")):
                    recovered = True
                    break
                try:
                    record = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ContractError(f"{path}:{index} is invalid JSON: {exc}") from exc
                record = validate_trial(record, f"{path}:{index}")
                if record["attempt_id"] in seen_attempts:
                    raise ContractError(f"duplicate attempt_id in {path}: {record['attempt_id']}")
                seen_attempts.add(record["attempt_id"])
                records.append(record)
    except OSError as exc:
        raise ContractError(f"cannot read {path}: {exc}") from exc
    return records, recovered


def completed_task_ids(records: Iterable[dict[str, Any]]) -> set[str]:
    return {record["task_id"] for record in records}


def pending_task_ids(scheduled: Iterable[str], records: Iterable[dict[str, Any]]) -> list[str]:
    completed = completed_task_ids(records)
    return [task_id for task_id in scheduled if task_id not in completed]


def append_trial(path: Path, record: dict[str, Any]) -> None:
    """Append one fsynced record under an advisory exclusive lock."""
    validate_trial(record)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "ab", closefd=False) as output:
            fcntl.flock(output.fileno(), fcntl.LOCK_EX)
            existing, recovered = read_trials(path)
            if recovered:
                raise ContractError(f"cannot append after partial trailing record in {path}; preserve and recover it first")
            if any(item["attempt_id"] == record["attempt_id"] for item in existing):
                raise ContractError(f"attempt_id already exists: {record['attempt_id']}")
            output.write(canonical_json_bytes(record))
            output.flush()
            os.fsync(output.fileno())
            fsync_directory(path.parent)
            fcntl.flock(output.fileno(), fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, value: Any, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as output:
            output.write(canonical_json_bytes(value))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def append_json_record(
    path: Path,
    record: dict[str, Any],
    *,
    unique_field: str,
    max_record_bytes: int = MAX_TRIAL_RECORD_BYTES,
) -> None:
    """Append a generic fsynced JSON object with duplicate-key rejection.

    This is used for request/generation journals whose records are intentionally
    distinct from terminal grading trials.
    """
    if not isinstance(record, dict) or unique_field not in record:
        raise ContractError(f"record must be an object containing {unique_field}")
    key = record[unique_field]
    if not isinstance(key, str) or not key:
        raise ContractError(f"record.{unique_field} must be a nonempty string")
    payload = canonical_json_bytes(record)
    if len(payload) > max_record_bytes:
        raise ContractError(f"record exceeds {max_record_bytes} bytes")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = path.with_name(path.name + ".lock")
    lock_descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        if path.exists():
            with path.open("rb") as existing:
                for line_number, raw in enumerate(existing, 1):
                    if len(raw) > max_record_bytes:
                        raise ContractError(f"{path}:{line_number} exceeds record limit")
                    if not raw.endswith(b"\n"):
                        raise ContractError(f"cannot append after partial trailing record in {path}")
                    try:
                        item = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        raise ContractError(f"{path}:{line_number} is invalid JSON: {exc}") from exc
                    if isinstance(item, dict) and item.get(unique_field) == key:
                        raise ContractError(f"{unique_field} already exists: {key}")
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
            os.write(descriptor, payload)
            os.fsync(descriptor)
            fsync_directory(path.parent)
        finally:
            os.close(descriptor)
    finally:
        os.close(lock_descriptor)


def read_json_records(
    path: Path,
    *,
    unique_field: str,
    max_record_bytes: int = MAX_TRIAL_RECORD_BYTES,
) -> tuple[list[dict[str, Any]], bool]:
    if not path.exists():
        return [], False
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    recovered = False
    with path.open("rb") as source:
        for line_number, raw in enumerate(source, 1):
            if len(raw) > max_record_bytes:
                raise ContractError(f"{path}:{line_number} exceeds record limit")
            if not raw.endswith(b"\n"):
                recovered = True
                break
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ContractError(f"{path}:{line_number} is invalid JSON: {exc}") from exc
            if not isinstance(item, dict):
                raise ContractError(f"{path}:{line_number} must contain an object")
            key = item.get(unique_field)
            if not isinstance(key, str) or not key:
                raise ContractError(f"{path}:{line_number}.{unique_field} must be nonempty")
            if key in seen:
                raise ContractError(f"duplicate {unique_field} in {path}: {key}")
            seen.add(key)
            result.append(item)
    return result, recovered
