"""Strict, dependency-free contracts for benchmark configuration and state."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from benchmarks import SCHEMA_VERSION


class ContractError(ValueError):
    """Raised when benchmark input does not satisfy its public contract."""


_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHA40_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object, rejecting duplicate keys and non-object roots."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_object)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{path} must contain a JSON object")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ContractError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{where} must be an object")
    return value


def _keys(value: dict[str, Any], where: str, required: Iterable[str], optional: Iterable[str] = ()) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = sorted(required_set - value.keys())
    unknown = sorted(value.keys() - allowed)
    if missing:
        raise ContractError(f"{where} missing fields: {', '.join(missing)}")
    if unknown:
        raise ContractError(f"{where} has unknown fields: {', '.join(unknown)}")


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{where} must be a positive integer")
    return value


def _identifier(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise ContractError(f"{where} must match {_ID_RE.pattern}")
    return value


def _repo_path(repo_root: Path, value: Any, where: str, nullable: bool = False) -> Path | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ContractError(f"{where} must be a nonempty repository-relative path")
    candidate = (repo_root / value).resolve()
    try:
        candidate.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise ContractError(f"{where} escapes the repository: {value}") from exc
    if not candidate.is_file():
        raise ContractError(f"{where} does not name a file: {value}")
    return candidate


def load_profiles(path: Path, repo_root: Path) -> dict[str, dict[str, Any]]:
    document = load_json(path)
    _keys(document, str(path), {"schema_version", "opencode_version", "opencode_sha256", "profiles"})
    if document["schema_version"] != SCHEMA_VERSION:
        raise ContractError(f"unsupported profile schema: {document['schema_version']!r}")
    if document["opencode_version"] != "1.18.30":
        raise ContractError("the profile matrix must pin OpenCode 1.18.30")
    if not _SHA256_RE.fullmatch(document["opencode_sha256"] if isinstance(document["opencode_sha256"], str) else ""):
        raise ContractError("the profile matrix must pin the extracted OpenCode binary SHA-256")
    if not isinstance(document["profiles"], list) or not document["profiles"]:
        raise ContractError("profiles must be a nonempty list")

    profiles: dict[str, dict[str, Any]] = {}
    for index, raw_profile in enumerate(document["profiles"]):
        where = f"profiles[{index}]"
        profile = _object(raw_profile, where)
        _keys(profile, where, {"id", "model", "runtime", "endpoint", "deployment", "files", "pin_variables", "server_pid_env"})
        profile_id = _identifier(profile["id"], f"{where}.id")
        if profile_id in profiles:
            raise ContractError(f"duplicate profile id: {profile_id}")

        model = _object(profile["model"], f"{where}.model")
        _keys(model, f"{where}.model", {"repository", "revision", "weight_format", "weight_bytes"})
        if not isinstance(model["repository"], str) or "/" not in model["repository"]:
            raise ContractError(f"{where}.model.repository must be an owner/name string")
        if not isinstance(model["revision"], str) or not _SHA40_RE.fullmatch(model["revision"]):
            raise ContractError(f"{where}.model.revision must be a 40-character lowercase commit")
        if not isinstance(model["weight_format"], str) or not model["weight_format"]:
            raise ContractError(f"{where}.model.weight_format must be nonempty")
        _positive_int(model["weight_bytes"], f"{where}.model.weight_bytes")

        runtime = _object(profile["runtime"], f"{where}.runtime")
        _keys(runtime, f"{where}.runtime", {"kind", "revision"})
        if runtime["kind"] not in {"vllm", "llama.cpp"}:
            raise ContractError(f"{where}.runtime.kind is unsupported")
        if not isinstance(runtime["revision"], str) or not runtime["revision"]:
            raise ContractError(f"{where}.runtime.revision must be nonempty")

        endpoint = _object(profile["endpoint"], f"{where}.endpoint")
        _keys(endpoint, f"{where}.endpoint", {"base_url", "model_alias"})
        parsed = urlparse(endpoint["base_url"] if isinstance(endpoint["base_url"], str) else "")
        if (parsed.scheme, parsed.hostname, parsed.path.rstrip("/")) != ("http", "127.0.0.1", "/v1"):
            raise ContractError(f"{where}.endpoint.base_url must be loopback HTTP ending in /v1")
        if parsed.port is None or not 1024 <= parsed.port <= 65535:
            raise ContractError(f"{where}.endpoint.base_url must use port 1024..65535")
        alias = _identifier(endpoint["model_alias"], f"{where}.endpoint.model_alias")

        deployment = _object(profile["deployment"], f"{where}.deployment")
        _keys(deployment, f"{where}.deployment", {
            "gpu", "partition", "gres", "system_ram_gib", "context_tokens",
            "output_tokens", "sequences", "reasoning",
        })
        for key in ("gpu", "partition", "gres", "reasoning"):
            if not isinstance(deployment[key], str) or not deployment[key]:
                raise ContractError(f"{where}.deployment.{key} must be nonempty")
        for key in ("system_ram_gib", "context_tokens", "output_tokens", "sequences"):
            _positive_int(deployment[key], f"{where}.deployment.{key}")
        if deployment["output_tokens"] > deployment["context_tokens"]:
            raise ContractError(f"{where} output budget exceeds context")

        files = _object(profile["files"], f"{where}.files")
        _keys(files, f"{where}.files", {"serve", "supervisor", "provider", "pins"})
        resolved_files = {
            key: _repo_path(repo_root, files[key], f"{where}.files.{key}", nullable=key in {"supervisor", "pins"})
            for key in files
        }
        pin_variables = profile["pin_variables"]
        if resolved_files["pins"] is None:
            if pin_variables is not None:
                raise ContractError(f"{where}.pin_variables must be null without a pins file")
        else:
            pin_variables = _object(pin_variables, f"{where}.pin_variables")
            _keys(pin_variables, f"{where}.pin_variables", {
                "repository", "revision", "weight_bytes", "model_alias", "runtime_revision",
            })
            assignments = load_shell_assignments(resolved_files["pins"])
            checks = {
                "repository": model["repository"],
                "revision": model["revision"],
                "weight_bytes": str(model["weight_bytes"]),
                "model_alias": alias,
            }
            if pin_variables["runtime_revision"] is not None:
                checks["runtime_revision"] = runtime["revision"]
            for semantic_name, expected in checks.items():
                variable = pin_variables[semantic_name]
                if not isinstance(variable, str) or assignments.get(variable) != expected:
                    raise ContractError(
                        f"{files['pins']} {variable!r} does not match profile {semantic_name}={expected!r}"
                    )
        provider = load_json(resolved_files["provider"])
        try:
            provider_base = provider["provider"]["pinnacles"]["options"]["baseURL"]
            provider_models = provider["provider"]["pinnacles"]["models"]
            provider_context = provider_models[alias]["limit"]["context"]
            provider_output = provider_models[alias]["limit"]["output"]
        except (KeyError, TypeError) as exc:
            raise ContractError(f"{files['provider']} does not expose pinnacles/{alias}") from exc
        if provider.get("model") != f"pinnacles/{alias}" or provider.get("small_model") != f"pinnacles/{alias}":
            raise ContractError(f"{files['provider']} model aliases do not match {alias}")
        if provider_base != endpoint["base_url"]:
            raise ContractError(f"{files['provider']} baseURL does not match profile endpoint")
        if (provider_context, provider_output) != (deployment["context_tokens"], deployment["output_tokens"]):
            raise ContractError(f"{files['provider']} token limits do not match profile deployment")
        if not isinstance(profile["server_pid_env"], str) or not profile["server_pid_env"].endswith("SERVER_PID"):
            raise ContractError(f"{where}.server_pid_env must name a SERVER_PID variable")

        profile["_resolved_files"] = resolved_files
        profile["_opencode_sha256"] = document["opencode_sha256"]
        profiles[profile_id] = profile
    return profiles


def load_shell_assignments(path: Path) -> dict[str, str]:
    """Read plain NAME=value pin files without executing shell code."""
    assignments: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ContractError(f"cannot read pins from {path}: {exc}") from exc
    pattern = re.compile(r"^([A-Z][A-Z0-9_]*)=([^\s'\"`$;]+)$")
    for line_number, line in enumerate(lines, 1):
        if not line or line.startswith("#"):
            continue
        match = pattern.fullmatch(line)
        if not match:
            raise ContractError(f"{path}:{line_number} is not a plain NAME=value pin")
        name, value = match.groups()
        if name in assignments:
            raise ContractError(f"duplicate pin variable in {path}: {name}")
        assignments[name] = value
    return assignments


def load_suite(path: Path, repo_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    suite = load_json(path)
    _keys(suite, str(path), {
        "schema_version", "id", "protocol_status", "release", "source",
        "task_manifest", "pilot_task_ids", "protocol", "execution", "budget", "scoring",
    })
    if suite["schema_version"] != SCHEMA_VERSION:
        raise ContractError(f"unsupported suite schema: {suite['schema_version']!r}")
    suite_id = _identifier(suite["id"], f"{path}.id")
    if suite["protocol_status"] not in {"candidate", "frozen"}:
        raise ContractError(f"{path}.protocol_status must be candidate or frozen")
    if not isinstance(suite["release"], str) or not suite["release"]:
        raise ContractError(f"{path}.release must be nonempty")

    source = _object(suite["source"], f"{path}.source")
    _keys(source, f"{path}.source", {"evaluator_repository", "evaluator_revision", "dataset_repository", "dataset_revision", "split"}, {"task_repository", "task_revision"})
    for key, value in source.items():
        if not isinstance(value, str) or not value:
            raise ContractError(f"{path}.source.{key} must be nonempty")
    for key in ("evaluator_revision", "dataset_revision", "task_revision"):
        if key in source and not _SHA40_RE.fullmatch(source[key]):
            raise ContractError(f"{path}.source.{key} must be an immutable 40-character revision")

    manifest = _object(suite["task_manifest"], f"{path}.task_manifest")
    _keys(manifest, f"{path}.task_manifest", {"path", "sha256", "expected_count"})
    task_path = _repo_path(repo_root, manifest["path"], f"{path}.task_manifest.path")
    if not _SHA256_RE.fullmatch(manifest["sha256"] if isinstance(manifest["sha256"], str) else ""):
        raise ContractError(f"{path}.task_manifest.sha256 must be lowercase SHA-256")
    if sha256_file(task_path) != manifest["sha256"]:
        raise ContractError(f"{manifest['path']} SHA-256 does not match suite manifest")
    expected_count = _positive_int(manifest["expected_count"], f"{path}.task_manifest.expected_count")
    tasks = load_task_records(task_path)
    if len(tasks) != expected_count:
        raise ContractError(f"{manifest['path']} has {len(tasks)} tasks, expected {expected_count}")

    pilot = suite["pilot_task_ids"]
    if not isinstance(pilot, list) or not pilot:
        raise ContractError(f"{path}.pilot_task_ids must be a nonempty list")
    if len(pilot) != len(set(pilot)):
        raise ContractError(f"{path}.pilot_task_ids contains duplicates")
    task_ids = {task["task_id"] for task in tasks}
    unknown_pilot = [task_id for task_id in pilot if task_id not in task_ids]
    if unknown_pilot:
        raise ContractError(f"{path}.pilot_task_ids are absent from task manifest: {unknown_pilot[:3]}")

    protocol_ref = _object(suite["protocol"], f"{path}.protocol")
    _keys(protocol_ref, f"{path}.protocol", {"path", "sha256"})
    protocol_path = _repo_path(repo_root, protocol_ref["path"], f"{path}.protocol.path")
    if not _SHA256_RE.fullmatch(protocol_ref["sha256"] if isinstance(protocol_ref["sha256"], str) else ""):
        raise ContractError(f"{path}.protocol.sha256 must be lowercase SHA-256")
    if sha256_file(protocol_path) != protocol_ref["sha256"]:
        raise ContractError(f"{protocol_ref['path']} SHA-256 does not match suite manifest")
    protocol = load_json(protocol_path)
    _keys(protocol, str(protocol_path), {
        "schema_version", "suite_id", "status", "agent", "prompt", "extraction",
        "sampling", "reasoning", "limits", "notes",
    })
    if protocol["schema_version"] != SCHEMA_VERSION or protocol["suite_id"] != suite_id:
        raise ContractError(f"{protocol_path} identity does not match its suite")
    if protocol["status"] != suite["protocol_status"]:
        raise ContractError(f"{protocol_path} status does not match its suite")
    if protocol["agent"] is not None and not isinstance(protocol["agent"], dict):
        raise ContractError(f"{protocol_path}.agent must be null or an object")
    for key in ("prompt", "extraction", "sampling", "limits"):
        if not isinstance(protocol[key], dict):
            raise ContractError(f"{protocol_path}.{key} must be an object")
    if not isinstance(protocol["reasoning"], str) or not isinstance(protocol["notes"], list):
        raise ContractError(f"{protocol_path} reasoning/notes have invalid types")

    execution = _object(suite["execution"], f"{path}.execution")
    _keys(execution, f"{path}.execution", {"adapter", "requires_code_execution", "requires_container", "primary_concurrency"})
    _repo_path(repo_root, execution["adapter"], f"{path}.execution.adapter")
    if not isinstance(execution["requires_code_execution"], bool) or not isinstance(execution["requires_container"], bool):
        raise ContractError(f"{path}.execution boolean fields must be booleans")
    _positive_int(execution["primary_concurrency"], f"{path}.execution.primary_concurrency")

    budget = _object(suite["budget"], f"{path}.budget")
    _keys(budget, f"{path}.budget", {"attempts", "request_timeout_seconds", "max_model_requests", "max_tool_events", "max_generated_tokens", "max_output_tokens", "context_tokens"})
    for key, value in budget.items():
        _positive_int(value, f"{path}.budget.{key}")
    if budget["max_output_tokens"] > budget["context_tokens"]:
        raise ContractError(f"{path}.budget output exceeds context")

    scoring = _object(suite["scoring"], f"{path}.scoring")
    _keys(scoring, f"{path}.scoring", {"metric", "unit", "denominator"})
    if scoring["denominator"] != "scheduled":
        raise ContractError(f"{path}.scoring.denominator must be scheduled")
    suite["_task_path"] = task_path
    suite["_protocol_path"] = protocol_path
    suite["_manifest_path"] = path.resolve()
    return suite, tasks


def load_task_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ContractError(f"cannot read task manifest {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line:
            raise ContractError(f"{path}:{line_number} is blank")
        try:
            record = json.loads(line, object_pairs_hook=_no_duplicate_object)
        except json.JSONDecodeError as exc:
            raise ContractError(f"{path}:{line_number} is invalid JSON: {exc}") from exc
        record = _object(record, f"{path}:{line_number}")
        _keys(record, f"{path}:{line_number}", {"task_id", "input_sha256", "grader_sha256", "stratum"})
        task_id = record["task_id"]
        if not isinstance(task_id, str) or not task_id or len(task_id) > 256:
            raise ContractError(f"{path}:{line_number}.task_id must be 1..256 characters")
        if task_id in seen:
            raise ContractError(f"duplicate task id in {path}: {task_id}")
        seen.add(task_id)
        for key in ("input_sha256", "grader_sha256"):
            if not _SHA256_RE.fullmatch(record[key] if isinstance(record[key], str) else ""):
                raise ContractError(f"{path}:{line_number}.{key} must be lowercase SHA-256")
        if not isinstance(record["stratum"], str) or not record["stratum"]:
            raise ContractError(f"{path}:{line_number}.stratum must be nonempty")
        records.append(record)
    if not records:
        raise ContractError(f"task manifest is empty: {path}")
    return records


def load_all_suites(suites_dir: Path, repo_root: Path) -> dict[str, tuple[dict[str, Any], list[dict[str, Any]]]]:
    suites: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    for path in sorted(suites_dir.glob("*/suite.json")):
        suite, tasks = load_suite(path, repo_root)
        if suite["id"] in suites:
            raise ContractError(f"duplicate suite id: {suite['id']}")
        suites[suite["id"]] = (suite, tasks)
    if not suites:
        raise ContractError(f"no suite manifests found under {suites_dir}")
    return suites
