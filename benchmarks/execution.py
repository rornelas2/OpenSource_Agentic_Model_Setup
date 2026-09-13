"""Live benchmark execution with immutable manifests and append-only journals."""

from __future__ import annotations

import datetime
import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from benchmarks import SCHEMA_VERSION
from benchmarks.adapters.aider_refactor import (
    build_trial_record as build_aider_trial_record,
    extract_code as extract_aider_code,
    format_prompt as format_aider_prompt,
    verify_refactor_ast,
)
from benchmarks.adapters.livecodebench import (
    build_trial_record as build_lcb_trial_record,
    evaluate_python_code,
    extract_code as extract_lcb_code,
    format_prompt as format_lcb_prompt,
)
from benchmarks.adapters.mmlu_pro import (
    build_trial_record as build_mmlu_trial_record,
    format_five_shot_prompt,
    parse_answer as parse_mmlu_answer,
    score as score_mmlu,
)
from benchmarks.adapters.swe_verified import (
    build_trial_record as build_swe_trial_record,
    evaluate_patch,
)
from benchmarks.adapters.terminal_bench import (
    build_trial_record as build_terminal_trial_record,
    normalize_reward as normalize_terminal_reward,
    run_harbor_verifier,
)
from benchmarks.contracts import ContractError, load_json, sha256_file
from benchmarks.opencode_runner import build_benchmark_config, run_opencode_in_container
from benchmarks.planning import ensure_private_directory, input_hashes, repository_commit
from benchmarks.preflight import run_preflight
from benchmarks.state import append_json_record, append_trial, atomic_write_json, read_json_records, read_trials
from benchmarks.telemetry import TelemetrySampler
from benchmarks.transport import ModelTransportClient, ModelTransportConfig


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _slurm_end_epoch() -> float:
    """Resolve the current allocation deadline so tasks never straddle it."""
    value = os.environ.get("SLURM_JOB_END_TIME")
    if value and value.isascii() and value.isdecimal():
        return float(value)
    if os.environ.get("BENCHMARK_ALLOW_LOCAL"):
        return time.time() + 86400.0
    job_id = os.environ.get("SLURM_JOB_ID", "")
    try:
        result = subprocess.run(
            ["scontrol", "show", "job", "--oneliner", job_id],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ContractError(f"cannot resolve Slurm allocation end time: {exc}") from exc
    match = re.search(r"(?:^|\s)EndTime=(\S+)", result.stdout)
    if not match or match.group(1) in {"Unknown", "N/A"}:
        raise ContractError("Slurm did not provide a finite allocation EndTime")
    try:
        end = datetime.datetime.fromisoformat(match.group(1))
    except ValueError as exc:
        raise ContractError(f"Slurm returned malformed EndTime={match.group(1)!r}") from exc
    if end.tzinfo is None:
        end = end.astimezone()
    return end.timestamp()


def _has_task_time(deadline_epoch: float, timeout_seconds: int) -> bool:
    return time.time() + timeout_seconds + 300 < deadline_epoch


def _read_cmdline(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError as exc:
        raise ContractError(f"cannot read owned server command line for PID {pid}: {exc}") from exc
    argv = [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]
    if not argv:
        raise ContractError(f"owned server PID {pid} has an empty command line")
    return argv


def _repository_state(repo_root: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=repo_root,
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30,
    )
    return {"commit": repository_commit(repo_root), "dirty": bool(result.stdout)}


def _load_preparation(suite: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    value = os.environ.get("BENCHMARK_PREPARATION_MANIFEST")
    if not value:
        raise ContractError("BENCHMARK_PREPARATION_MANIFEST must name the immutable successful preparation manifest")
    path = Path(value).expanduser().resolve()
    manifest = load_json(path)
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("operation") != "prepare":
        raise ContractError(f"invalid preparation manifest: {path}")
    prepared = manifest.get("suites", {}).get(suite["id"])
    if not isinstance(prepared, dict) or prepared.get("status") != "ready":
        raise ContractError(f"preparation manifest has no ready entry for {suite['id']}")
    evaluator = prepared.get("evaluator", {})
    dataset = prepared.get("dataset", {})
    source = suite["source"]
    if evaluator.get("revision") != source["evaluator_revision"]:
        raise ContractError("preparation evaluator revision is stale")
    if dataset.get("revision") != source["dataset_revision"]:
        raise ContractError("preparation dataset revision is stale")
    checkout = Path(evaluator.get("checkout_path", ""))
    python_bin = Path(evaluator.get("python_bin", ""))
    lock = evaluator.get("lock", {})
    lock_path = Path(lock.get("path", ""))
    if not checkout.is_dir() or not python_bin.is_file() or not lock_path.is_file():
        raise ContractError("prepared evaluator files are missing")
    if sha256_file(lock_path) != lock.get("sha256"):
        raise ContractError("prepared evaluator lock hash is stale")
    public_path_text = dataset.get("validation", {}).get("public_inputs_path")
    public_hash = dataset.get("validation", {}).get("public_inputs_sha256")
    if suite["id"] == "swe-verified":
        if not isinstance(public_path_text, str) or not isinstance(public_hash, str):
            raise ContractError("SWE preparation is missing staged public inputs")
        if sha256_file(Path(public_path_text)) != public_hash:
            raise ContractError("staged SWE public inputs hash is stale")
        if prepared.get("container", {}).get("status") != "available":
            raise ContractError("SWE preparation has no validated container runtime")
        if not isinstance(prepared.get("images"), list) or not prepared["images"]:
            raise ContractError("SWE preparation has no immutable official task-image records")
        evaluator_dataset_path = dataset.get("validation", {}).get("evaluator_dataset_path")
        evaluator_dataset_hash = dataset.get("validation", {}).get("evaluator_dataset_sha256")
        if not isinstance(evaluator_dataset_path, str) or not isinstance(evaluator_dataset_hash, str):
            raise ContractError("SWE preparation has no pinned evaluator-compatible dataset artifact")
        if sha256_file(Path(evaluator_dataset_path)) != evaluator_dataset_hash:
            raise ContractError("pinned SWE evaluator dataset artifact is stale")
    else:
        if isinstance(public_path_text, str) and isinstance(public_hash, str):
            if sha256_file(Path(public_path_text)) != public_hash:
                raise ContractError(f"staged {suite['id']} public inputs hash is stale")
        if suite["execution"]["requires_container"]:
            if prepared.get("container", {}).get("status") != "available":
                raise ContractError(f"{suite['id']} preparation has no validated container runtime")
    from benchmarks.preparation import revalidate_prepared_suite
    revalidate_prepared_suite(prepared, suite)
    return prepared, path


def _load_public_inputs(prepared: dict[str, Any]) -> dict[str, dict[str, Any]]:
    path_str = prepared.get("dataset", {}).get("validation", {}).get("public_inputs_path")
    if not path_str or not Path(path_str).is_file():
        raise ContractError("preparation manifest does not have valid public_inputs_path")
    path = Path(path_str)
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(f"{path}:{line_number} is invalid JSON: {exc}") from exc
            task_id = (
                row.get("task_id")
                or row.get("instance_id")
                or (str(row.get("question_id")) if "question_id" in row else None)
            )
            if not isinstance(task_id, str) or not task_id or task_id in rows:
                raise ContractError(f"{path}:{line_number} has an invalid or duplicate task ID")
            rows[task_id] = row
    return rows


def _load_private_grader(prepared: dict[str, Any]) -> dict[str, dict[str, Any]]:
    val = prepared.get("dataset", {}).get("validation", {})
    path_str = val.get("private_grader_path")
    if not path_str or not Path(path_str).is_file():
        return {}
    path = Path(path_str)
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(f"{path}:{line_number} is invalid JSON: {exc}") from exc
            task_id = (
                row.get("task_id")
                or row.get("instance_id")
                or (str(row.get("question_id")) if "question_id" in row else None)
            )
            if not isinstance(task_id, str) or not task_id or task_id in rows:
                raise ContractError(f"{path}:{line_number} has an invalid or duplicate task ID")
            rows[task_id] = row
    return rows


def _load_mmlu_validation_shots(prepared: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    val = prepared.get("dataset", {}).get("validation", {})
    path_str = val.get("validation_shots_path")
    if path_str and Path(path_str).is_file():
        return load_json(Path(path_str))
    return {}


def _validate_generation_records(
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
    tasks: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Reject stale, duplicate, or noncanonical saved generation attempts."""
    by_task: dict[str, dict[str, Any]] = {}
    task_meta = {task["task_id"]: task for task in tasks}
    scheduled = set(manifest["scheduled_task_ids"])
    for record in records:
        task_id = record.get("task_id")
        if record.get("run_id") != manifest.get("run_id") or task_id not in scheduled:
            raise ContractError("generation journal contains stale or out-of-schedule records")
        if task_id in by_task:
            raise ContractError(f"generation journal contains multiple attempts for {task_id}")
        if record.get("attempt_id") != f"{manifest['run_id']}:{task_id}:1":
            raise ContractError(f"generation journal has a noncanonical attempt ID for {task_id}")
        if record.get("input_sha256") != task_meta[task_id]["input_sha256"]:
            raise ContractError(f"generation journal has a stale input hash for {task_id}")
        if record.get("status") not in {"generated", "unsolved", "timeout", "cancelled", "infra_error"}:
            raise ContractError(f"generation journal has an invalid status for {task_id}")
        by_task[task_id] = record
    return by_task


def _load_or_create_run_manifest(
    *, args: Any, profile: dict[str, Any], suite: dict[str, Any], tasks: list[dict[str, Any]],
    run_dir: Path, repo_root: Path, profiles_path: Path, preparation_path: Path,
    preflight_report: dict[str, Any],
) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    scheduled = suite["pilot_task_ids"] if args.pilot else [task["task_id"] for task in tasks]
    if path.exists():
        if not args.resume:
            raise ContractError(f"run manifest already exists; use --resume: {path}")
        manifest = load_json(path)
        expected = (profile["id"], suite["id"], bool(args.pilot), scheduled)
        actual = (manifest.get("profile_id"), manifest.get("suite_id"), manifest.get("pilot"), manifest.get("scheduled_task_ids"))
        if actual != expected:
            raise ContractError("resume arguments do not match the immutable run manifest")
        if manifest.get("input_sha256") != input_hashes(profile, suite, profiles_path):
            raise ContractError("resume inputs differ from the immutable run manifest")
        if manifest.get("preparation_manifest") != str(preparation_path):
            raise ContractError("resume preparation path differs from the immutable run manifest")
        if manifest.get("preparation_manifest_sha256") != sha256_file(preparation_path):
            raise ContractError("resume preparation content differs from the immutable run manifest")
        return manifest

    server_pid = int(os.environ[profile["server_pid_env"]])
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{profile['id']}-{suite['id']}-{stamp}-{uuid.uuid4().hex[:8]}"
    manifest = {
        "schema_version": SCHEMA_VERSION, "run_id": run_id,
        "campaign_id": os.environ.get("BENCHMARK_CAMPAIGN_ID", f"campaign-{stamp}"),
        "profile_id": profile["id"], "suite_id": suite["id"], "pilot": bool(args.pilot),
        "protocol_status": suite["protocol_status"], "repository": _repository_state(repo_root),
        "input_sha256": input_hashes(profile, suite, profiles_path),
        "preparation_manifest": str(preparation_path),
        "preparation_manifest_sha256": sha256_file(preparation_path),
        "model": profile["model"], "runtime": profile["runtime"],
        "opencode": {"version": "1.18.30", "sha256": profile["_opencode_sha256"]},
        "endpoint": profile["endpoint"], "hardware": preflight_report["gpu"],
        "allocation": {"slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"), "node": os.environ.get("SLURMD_NODENAME", "localhost")},
        "suite": {"release": suite["release"], "source": suite["source"],
                  "task_manifest_sha256": suite["task_manifest"]["sha256"],
                  "protocol_sha256": suite["protocol"]["sha256"]},
        "sampling": {"temperature": 0, "top_p": 1, "seed": 20260911},
        "reasoning": {"mode": profile["deployment"]["reasoning"]},
        "cache": {"recorded_from_server": False}, "budgets": suite["budget"],
        "execution_policy": {"concurrency": 1, "delegation": False},
        "scheduled_task_ids": scheduled, "raw_artifact_root": str(run_dir),
        "workspace_root": "official-task-containers", "server_argv": _read_cmdline(server_pid),
        "started_utc": utc_now(), "ended_utc": None,
    }
    atomic_write_json(path, manifest)
    return manifest


def execute_generation(
    args: Any, profile: dict[str, Any], suite: dict[str, Any], tasks: list[dict[str, Any]],
    run_dir: Path, repo_root: Path, profiles_path: Path,
) -> dict[str, Any]:
    """Generate saved model artifacts; grading remains a separate offline step."""
    job_id = os.environ.get("SLURM_JOB_ID")
    if not os.environ.get("BENCHMARK_ALLOW_LOCAL"):
        if not job_id or not job_id.isascii() or not job_id.isdecimal():
            raise ContractError("generation execution requires a numeric Slurm allocation")
    prepared, preparation_path = _load_preparation(suite)
    ensure_private_directory(run_dir)
    preflight_report = run_preflight(profile, suite)
    preflight_path = run_dir / "preflight.json"
    if preflight_path.exists():
        if load_json(preflight_path).get("profile_id") != profile["id"]:
            raise ContractError("stale preflight artifact belongs to another profile")
    else:
        atomic_write_json(preflight_path, preflight_report)
    manifest = _load_or_create_run_manifest(
        args=args, profile=profile, suite=suite, tasks=tasks, run_dir=run_dir,
        repo_root=repo_root, profiles_path=profiles_path, preparation_path=preparation_path,
        preflight_report=preflight_report,
    )
    generations_path = run_dir / "generations.jsonl"
    generations, partial = read_json_records(generations_path, unique_field="attempt_id")
    if partial:
        raise ContractError("generation journal has a partial trailing record; preserve it before recovery")
    completed = set(_validate_generation_records(generations, manifest, tasks))
    pending = [task_id for task_id in manifest["scheduled_task_ids"] if task_id not in completed]
    if generations and not args.resume:
        raise ContractError("generation records already exist; use --resume")

    public_inputs = _load_public_inputs(prepared)
    task_meta = {task["task_id"]: task for task in tasks}

    telemetry = TelemetrySampler(
        run_dir / "telemetry.jsonl", server_pid=int(os.environ[profile["server_pid_env"]]),
        gpu_uuid=preflight_report["gpu"]["uuid"], sample_interval_seconds=1.0,
    )
    telemetry.start()

    suite_id = suite["id"]
    if suite_id in {"swe-verified", "terminal-bench"}:
        images = {item.get("task_id"): item for item in prepared.get("images", []) if isinstance(item, dict)}
        network = os.environ.get("BENCHMARK_CONTAINER_NETWORK")
        container_endpoint = os.environ.get("BENCHMARK_CONTAINER_ENDPOINT_BASE_URL")
        if not network or not container_endpoint:
            raise ContractError(
                "BENCHMARK_CONTAINER_NETWORK and BENCHMARK_CONTAINER_ENDPOINT_BASE_URL must identify the "
                "preflight-validated internal network and owned endpoint relay"
            )
        opencode_binary = Path(os.environ.get(
            "BENCHMARK_OPENCODE_BINARY",
            f"/data/{os.environ.get('USER', '')}/pinnacles-agents/tools/opencode-v1.18.30/opencode",
        )).resolve()
        config = build_benchmark_config(
            profile["_resolved_files"]["provider"], endpoint_base_url=container_endpoint,
            model_alias=profile["endpoint"]["model_alias"], context_tokens=suite["budget"]["context_tokens"],
            output_tokens=suite["budget"]["max_output_tokens"],
        )
    elif suite_id in {"livecodebench", "mmlu-pro", "aider-refactor"}:
        transport_config = ModelTransportConfig(
            base_url=profile["endpoint"]["base_url"],
            expected_model_alias=profile["endpoint"]["model_alias"],
            total_timeout=float(suite["budget"]["request_timeout_seconds"]),
        )
        transport_client = ModelTransportClient(transport_config)
        if suite_id == "mmlu-pro":
            validation_shots = _load_mmlu_validation_shots(prepared)
    else:
        raise ContractError(f"unknown suite {suite_id}")

    produced = 0
    allocation_deadline = _slurm_end_epoch()
    try:
        for task_id in pending:
            if not _has_task_time(allocation_deadline, suite["budget"]["request_timeout_seconds"]):
                break
            task = public_inputs.get(task_id)
            meta = task_meta.get(task_id)
            if task is None or meta is None:
                raise ContractError(f"prepared artifacts are incomplete for {task_id}")
            if task.get("input_sha256") != meta["input_sha256"]:
                raise ContractError(f"staged task input hash changed for {task_id}")
            attempt_id = f"{manifest['run_id']}:{task_id}:1"
            artifact_dir = run_dir / "tasks" / task_id.replace("/", "__") / "attempt-1"
            if artifact_dir.exists() and any(artifact_dir.iterdir()):
                raise ContractError(
                    f"orphaned artifact directory exists without a terminal journal record: {artifact_dir}"
                )
            artifact_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            started_utc = utc_now()
            try:
                if suite_id == "swe-verified":
                    image = images.get(task_id)
                    if image is None:
                        raise ContractError(f"prepared SWE artifacts are incomplete for {task_id}")
                    result = run_opencode_in_container(
                        image=image["image"], expected_image_id=image["image_id"], network=network,
                        config=config, opencode_binary=opencode_binary,
                        expected_opencode_sha256=profile["_opencode_sha256"],
                        prompt=("Resolve the issue in the current repository. Inspect the code, make the smallest "
                                "correct change, and run relevant tests.\n\nProblem statement:\n" + task["problem_statement"]),
                        artifact_dir=artifact_dir, model_alias=profile["endpoint"]["model_alias"],
                        timeout_seconds=suite["budget"]["request_timeout_seconds"],
                        max_requests=suite["budget"]["max_model_requests"],
                        max_tool_events=suite["budget"]["max_tool_events"],
                        max_generated_tokens=suite["budget"]["max_generated_tokens"],
                        client_pid_callback=telemetry.set_client_pid,
                    )
                    patch_bytes = Path(result["patch_path"]).read_bytes()
                    termination = result["termination_reason"]
                    if termination == "wall_time_limit":
                        generation_status = "timeout"
                    elif termination in {"client_nonzero_exit", "missing_completion_event", "event_line_size_limit", "trajectory_size_limit",
                                         "truncated_event", "malformed_event"}:
                        generation_status = "infra_error"
                    else:
                        generation_status = "generated" if patch_bytes.strip() else "unsolved"
                    reason = result["termination_reason"] or ("patch_saved" if generation_status == "generated" else "empty_patch")
                    record = {"schema_version": SCHEMA_VERSION, "run_id": manifest["run_id"], "task_id": task_id,
                              "attempt_id": attempt_id, "input_sha256": meta["input_sha256"],
                              "status": generation_status, "reason": reason, "started_utc": started_utc,
                              "ended_utc": utc_now(), **result}

                elif suite_id == "terminal-bench":
                    image = images.get(task_id)
                    image_name = image["image"] if image else f"terminal-bench/{task_id}:latest"
                    image_id = image.get("image_id") if image else None
                    prompt = task.get("instruction") or "Use the terminal to complete the task in the provided environment."
                    result = run_opencode_in_container(
                        image=image_name, expected_image_id=image_id, network=network,
                        config=config, opencode_binary=opencode_binary,
                        expected_opencode_sha256=profile["_opencode_sha256"],
                        prompt=prompt,
                        artifact_dir=artifact_dir, model_alias=profile["endpoint"]["model_alias"],
                        timeout_seconds=suite["budget"]["request_timeout_seconds"],
                        max_requests=suite["budget"]["max_model_requests"],
                        max_tool_events=suite["budget"]["max_tool_events"],
                        max_generated_tokens=suite["budget"]["max_generated_tokens"],
                        client_pid_callback=telemetry.set_client_pid,
                    )
                    termination = result["termination_reason"]
                    if termination == "wall_time_limit":
                        generation_status = "timeout"
                    elif termination in {"client_nonzero_exit", "missing_completion_event", "event_line_size_limit", "trajectory_size_limit",
                                         "truncated_event", "malformed_event"}:
                        generation_status = "infra_error"
                    else:
                        generation_status = "generated"
                    reason = result["termination_reason"] or "agent_finished"
                    record = {"schema_version": SCHEMA_VERSION, "run_id": manifest["run_id"], "task_id": task_id,
                              "attempt_id": attempt_id, "input_sha256": meta["input_sha256"],
                              "status": generation_status, "reason": reason, "started_utc": started_utc,
                              "ended_utc": utc_now(), **result}

                elif suite_id == "livecodebench":
                    messages = format_lcb_prompt(task)
                    telemetry.set_client_pid(os.getpid())
                    chat_res = transport_client.chat_completion(
                        messages=messages,
                        stream=True,
                        max_tokens=suite["budget"]["max_output_tokens"],
                        temperature=0.0,
                        top_p=1.0,
                    )
                    code = extract_lcb_code(chat_res.content)
                    response_path = artifact_dir / "response.txt"
                    response_path.write_text(chat_res.content, encoding="utf-8")
                    solution_path = artifact_dir / "solution.py"
                    solution_path.write_text(code, encoding="utf-8")
                    if chat_res.finish_reason == "length":
                        generation_status = "timeout"
                        reason = "output_length_exceeded"
                    elif code.strip():
                        generation_status = "generated"
                        reason = "code_extracted"
                    else:
                        generation_status = "unsolved"
                        reason = "empty_code"
                    prompt_tok = chat_res.usage.prompt_tokens if chat_res.usage else None
                    gen_tok = chat_res.usage.completion_tokens if chat_res.usage else None
                    record = {
                        "schema_version": SCHEMA_VERSION,
                        "run_id": manifest["run_id"],
                        "task_id": task_id,
                        "attempt_id": attempt_id,
                        "input_sha256": meta["input_sha256"],
                        "status": generation_status,
                        "reason": reason,
                        "started_utc": started_utc,
                        "ended_utc": utc_now(),
                        "prompt_tokens": prompt_tok,
                        "generated_tokens": gen_tok,
                        "token_source": "server_usage" if chat_res.usage else "none",
                        "elapsed_seconds": chat_res.end_to_end_latency,
                        "request_count": 1,
                        "tool_event_count": 0,
                        "termination_reason": chat_res.finish_reason,
                        "artifact_dir": str(artifact_dir.resolve()),
                        "response_path": str(response_path.resolve()),
                        "response_sha256": sha256_file(response_path),
                        "solution_path": str(solution_path.resolve()),
                        "solution_sha256": sha256_file(solution_path),
                    }

                elif suite_id == "mmlu-pro":
                    shots = validation_shots.get(task.get("category", "General"), [])[:5]
                    messages = format_five_shot_prompt(task, shots)
                    telemetry.set_client_pid(os.getpid())
                    chat_res = transport_client.chat_completion(
                        messages=messages,
                        stream=True,
                        max_tokens=suite["budget"]["max_output_tokens"],
                        temperature=0.0,
                        top_p=1.0,
                    )
                    parsed_choice = parse_mmlu_answer(chat_res.content)
                    response_path = artifact_dir / "response.txt"
                    response_path.write_text(chat_res.content, encoding="utf-8")
                    if chat_res.finish_reason == "length":
                        generation_status = "timeout"
                        reason = "output_length_exceeded"
                    elif parsed_choice:
                        generation_status = "generated"
                        reason = f"answer_parsed_{parsed_choice}"
                    else:
                        generation_status = "unsolved"
                        reason = "no_valid_answer_parsed"
                    prompt_tok = chat_res.usage.prompt_tokens if chat_res.usage else None
                    gen_tok = chat_res.usage.completion_tokens if chat_res.usage else None
                    record = {
                        "schema_version": SCHEMA_VERSION,
                        "run_id": manifest["run_id"],
                        "task_id": task_id,
                        "attempt_id": attempt_id,
                        "input_sha256": meta["input_sha256"],
                        "status": generation_status,
                        "reason": reason,
                        "started_utc": started_utc,
                        "ended_utc": utc_now(),
                        "prompt_tokens": prompt_tok,
                        "generated_tokens": gen_tok,
                        "token_source": "server_usage" if chat_res.usage else "none",
                        "elapsed_seconds": chat_res.end_to_end_latency,
                        "request_count": 1,
                        "tool_event_count": 0,
                        "termination_reason": chat_res.finish_reason,
                        "parsed_answer": parsed_choice,
                        "artifact_dir": str(artifact_dir.resolve()),
                        "response_path": str(response_path.resolve()),
                        "response_sha256": sha256_file(response_path),
                    }

                elif suite_id == "aider-refactor":
                    messages = format_aider_prompt(task)
                    telemetry.set_client_pid(os.getpid())
                    chat_res = transport_client.chat_completion(
                        messages=messages,
                        stream=True,
                        max_tokens=suite["budget"]["max_output_tokens"],
                        temperature=0.0,
                        top_p=1.0,
                    )
                    code = extract_aider_code(chat_res.content)
                    response_path = artifact_dir / "response.txt"
                    response_path.write_text(chat_res.content, encoding="utf-8")
                    solution_path = artifact_dir / "solution.py"
                    solution_path.write_text(code, encoding="utf-8")
                    if chat_res.finish_reason == "length":
                        generation_status = "timeout"
                        reason = "output_length_exceeded"
                    elif code.strip():
                        generation_status = "generated"
                        reason = "code_extracted"
                    else:
                        generation_status = "unsolved"
                        reason = "empty_code"
                    prompt_tok = chat_res.usage.prompt_tokens if chat_res.usage else None
                    gen_tok = chat_res.usage.completion_tokens if chat_res.usage else None
                    record = {
                        "schema_version": SCHEMA_VERSION,
                        "run_id": manifest["run_id"],
                        "task_id": task_id,
                        "attempt_id": attempt_id,
                        "input_sha256": meta["input_sha256"],
                        "status": generation_status,
                        "reason": reason,
                        "started_utc": started_utc,
                        "ended_utc": utc_now(),
                        "solution_path": str(solution_path.resolve()),
                        "solution_sha256": sha256_file(solution_path),
                        "response_path": str(response_path.resolve()),
                        "response_sha256": sha256_file(response_path),
                        "elapsed_seconds": chat_res.end_to_end_latency,
                        "request_count": 1,
                        "tool_event_count": 0,
                        "termination_reason": chat_res.finish_reason,
                        "prompt_tokens": prompt_tok,
                        "generated_tokens": gen_tok,
                        "token_source": "server_usage" if chat_res.usage else "none",
                        "http_status": chat_res.http_status,
                        "artifact_dir": str(artifact_dir.resolve()),
                    }
            except KeyboardInterrupt:
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": manifest["run_id"],
                    "task_id": task_id,
                    "attempt_id": attempt_id,
                    "input_sha256": meta["input_sha256"],
                    "status": "cancelled",
                    "reason": "client_interrupted",
                    "error_class": "KeyboardInterrupt",
                    "started_utc": started_utc,
                    "ended_utc": utc_now(),
                    "artifact_dir": str(artifact_dir.resolve()),
                }
                append_json_record(generations_path, record, unique_field="attempt_id")
                raise
            except Exception as exc:
                record = {"schema_version": SCHEMA_VERSION, "run_id": manifest["run_id"], "task_id": task_id,
                          "attempt_id": attempt_id, "input_sha256": meta["input_sha256"],
                          "status": "infra_error", "reason": str(exc), "error_class": type(exc).__name__,
                          "started_utc": started_utc, "ended_utc": utc_now(),
                          "artifact_dir": str(artifact_dir.resolve())}
            append_json_record(generations_path, record, unique_field="attempt_id")
            produced += 1
    finally:
        telemetry.stop()
        atomic_write_json(run_dir / "generation-state.json", {
            "schema_version": SCHEMA_VERSION, "run_id": manifest["run_id"], "updated_utc": utc_now(),
            "telemetry": telemetry.summary(),
        })
    return {"run_id": manifest["run_id"], "suite_id": suite["id"], "profile_id": profile["id"],
            "generated_this_invocation": produced, "remaining": len(pending) - produced}


def execute_evaluate(
    args: Any,
    profile: dict[str, Any],
    suite: dict[str, Any],
    tasks: list[dict[str, Any]],
    run_dir: Path,
) -> dict[str, Any]:
    """Grade saved predictions with the exact prepared official evaluator."""
    job_id = os.environ.get("SLURM_JOB_ID")
    if not os.environ.get("BENCHMARK_ALLOW_LOCAL"):
        if not job_id or not job_id.isascii() or not job_id.isdecimal():
            raise ContractError("evaluation execution requires a numeric Slurm allocation")
    manifest = load_json(run_dir / "manifest.json")
    if (manifest.get("profile_id"), manifest.get("suite_id")) != (profile["id"], suite["id"]):
        raise ContractError("run manifest does not match evaluate arguments")
    prepared, preparation_path = _load_preparation(suite)
    if sha256_file(preparation_path) != manifest.get("preparation_manifest_sha256"):
        raise ContractError("preparation manifest differs from the one frozen for this run")
    generations, partial = read_json_records(run_dir / "generations.jsonl", unique_field="attempt_id")
    if partial:
        raise ContractError("generation journal has an unresolved partial record")
    scheduled = set(manifest["scheduled_task_ids"])
    by_task = _validate_generation_records(generations, manifest, tasks)
    missing = [task_id for task_id in manifest["scheduled_task_ids"] if task_id not in by_task]
    if missing:
        raise ContractError(f"evaluation requires complete generation coverage; missing {missing[:3]}")
    trials_path = run_dir / "trials.jsonl"
    trials, trial_partial = read_trials(trials_path)
    if trial_partial:
        raise ContractError("trial journal has an unresolved partial record")
    already: set[str] = set()
    for record in trials:
        if record["run_id"] != manifest["run_id"] or record["task_id"] not in scheduled:
            raise ContractError("trial journal contains stale or out-of-schedule records")
        if record["task_id"] in already:
            raise ContractError(f"trial journal contains multiple terminal attempts for {record['task_id']}")
        already.add(record["task_id"])

    private_grader = _load_private_grader(prepared)
    evaluator_python = Path(prepared.get("evaluator", {}).get("python_bin", ""))
    suite_id = suite["id"]

    evaluated = 0
    counts = {
        "pass": 0,
        "fail": 0,
        "timeout": 0,
        "cancelled": 0,
        "infra_error": 0,
        "skipped": 0,
        "not_run": 0,
    }
    allocation_deadline = _slurm_end_epoch()
    for task_id in manifest["scheduled_task_ids"]:
        if task_id in already:
            continue
        if not _has_task_time(allocation_deadline, suite["budget"]["request_timeout_seconds"]):
            break
        generation = by_task[task_id]
        started = utc_now()

        if generation["status"] == "infra_error":
            passed, reason, metrics, artifacts = False, "generation_infra_error", {}, {}
            status = "infra_error"
        elif generation["status"] == "timeout":
            passed, reason, metrics, artifacts = False, "generation_timeout", {}, {}
            status = "timeout"
        elif generation["status"] == "cancelled":
            passed, reason, metrics, artifacts = False, "generation_cancelled", {}, {}
            status = "cancelled"
        else:
            if suite_id == "swe-verified":
                patch = ""
                patch_path_text = generation.get("patch_path")
                if isinstance(patch_path_text, str) and Path(patch_path_text).is_file():
                    patch_path = Path(patch_path_text)
                    if sha256_file(patch_path) != generation.get("patch_sha256"):
                        raise ContractError(f"saved patch hash is stale for {task_id}")
                    patch = patch_path.read_text(encoding="utf-8")
                passed, reason, native = evaluate_patch(
                    instance_id=task_id,
                    patch=patch,
                    model_name_or_path=profile["endpoint"]["model_alias"],
                    evaluator_dataset_path=Path(prepared["dataset"]["validation"]["evaluator_dataset_path"]),
                    evaluator_env_bin=evaluator_python,
                    evaluator_src_dir=Path(prepared["evaluator"]["checkout_path"]),
                    run_dir=run_dir / "grading" / task_id.replace("/", "__"),
                    timeout=suite["budget"]["request_timeout_seconds"],
                )
                if reason == "evaluator_timeout":
                    status = "timeout"
                elif reason.startswith("evaluator_infra"):
                    status = "infra_error"
                else:
                    status = "pass" if passed else "fail"
                metrics = {
                    "elapsed_seconds": generation.get("elapsed_seconds"),
                    "request_count": generation.get("request_count"),
                    "tool_event_count": generation.get("tool_event_count"),
                    "reward": 1.0 if passed else 0.0,
                    "official_resolved": passed,
                    "native_evaluator": native,
                }
                artifacts = {key: generation[key] for key in (
                    "patch_path", "patch_sha256", "trajectory_path", "trajectory_sha256", "stderr_path"
                ) if key in generation}
                builder = build_swe_trial_record

            elif suite_id == "livecodebench":
                code = ""
                sol_path_text = generation.get("solution_path")
                if isinstance(sol_path_text, str) and Path(sol_path_text).is_file():
                    sol_path = Path(sol_path_text)
                    if sha256_file(sol_path) != generation.get("solution_sha256"):
                        raise ContractError(f"saved solution hash is stale for {task_id}")
                    code = sol_path.read_text(encoding="utf-8")
                grader_row = private_grader.get(task_id, {})
                test_cases = grader_row.get("private_test_cases", [])
                metadata = grader_row.get("metadata", {})
                evaluator_src = Path(prepared.get("evaluator", {}).get("checkout_path", ""))
                passed, reason = evaluate_python_code(
                    code,
                    test_cases,
                    metadata=metadata,
                    evaluator_python=evaluator_python,
                    evaluator_src_dir=evaluator_src,
                    timeout_per_case=5,
                )
                if reason.startswith("timeout") or reason.startswith("evaluator_timeout"):
                    status = "timeout"
                elif reason.startswith("evaluator_infra"):
                    status = "infra_error"
                elif reason.startswith("exec_error") or reason.startswith("runtime_error"):
                    status = "fail"
                else:
                    status = "pass" if passed else "fail"
                metrics = {
                    "elapsed_seconds": generation.get("elapsed_seconds"),
                    "request_count": generation.get("request_count"),
                    "tool_event_count": 0,
                    "reward": 1.0 if passed else 0.0,
                    "eval_reason": reason,
                }
                artifacts = {key: generation[key] for key in (
                    "solution_path", "solution_sha256", "response_path", "response_sha256"
                ) if key in generation}
                builder = build_lcb_trial_record

            elif suite_id == "mmlu-pro":
                resp_text = ""
                resp_path_text = generation.get("response_path")
                if isinstance(resp_path_text, str) and Path(resp_path_text).is_file():
                    resp_path = Path(resp_path_text)
                    if sha256_file(resp_path) != generation.get("response_sha256"):
                        raise ContractError(f"saved response hash is stale for {task_id}")
                    resp_text = resp_path.read_text(encoding="utf-8")
                grader_row = private_grader.get(task_id, {})
                expected_answer = grader_row.get("answer", "")
                passed, parsed_choice = score_mmlu(resp_text, expected_answer)
                status = "pass" if passed else "fail"
                reason = f"parsed_{parsed_choice}_expected_{expected_answer}"
                metrics = {
                    "elapsed_seconds": generation.get("elapsed_seconds"),
                    "request_count": generation.get("request_count"),
                    "tool_event_count": 0,
                    "reward": 1.0 if passed else 0.0,
                    "parsed_answer": parsed_choice,
                    "expected_answer": expected_answer,
                }
                artifacts = {key: generation[key] for key in (
                    "response_path", "response_sha256"
                ) if key in generation}
                builder = build_mmlu_trial_record

            elif suite_id == "terminal-bench":
                grader_row = private_grader.get(task_id, {})
                verifier_script = grader_row.get("verifier_path")
                task_dir = Path(generation.get("artifact_dir", run_dir / "tasks" / task_id.replace("/", "__") / "attempt-1"))
                if verifier_script and Path(verifier_script).is_file():
                    verifier_result = run_harbor_verifier(Path(verifier_script), task_dir=task_dir, timeout=300)
                    reward = normalize_terminal_reward(verifier_result)
                else:
                    verifier_result = {"reward": 0.0, "stdout": "", "stderr": "verifier not found", "exit_code": -1}
                    reward = 0.0
                passed = (reward == 1.0)
                status = "pass" if passed else "fail"
                reason = f"terminal_reward_{reward:.2f}"
                metrics = {
                    "elapsed_seconds": generation.get("elapsed_seconds"),
                    "request_count": generation.get("request_count"),
                    "tool_event_count": generation.get("tool_event_count", 0),
                    "reward": reward,
                    "harbor_verifier": verifier_result,
                }
                artifacts = {key: generation[key] for key in (
                    "patch_path", "patch_sha256", "trajectory_path", "trajectory_sha256", "stderr_path"
                ) if key in generation}
                builder = build_terminal_trial_record

            elif suite_id == "aider-refactor":
                code = ""
                sol_path_text = generation.get("solution_path")
                if isinstance(sol_path_text, str) and Path(sol_path_text).is_file():
                    sol_path = Path(sol_path_text)
                    if sha256_file(sol_path) != generation.get("solution_sha256"):
                        raise ContractError(f"saved solution hash is stale for {task_id}")
                    code = sol_path.read_text(encoding="utf-8")
                grader_row = private_grader.get(task_id, {})
                passed, reason = verify_refactor_ast(
                    code=code,
                    method=grader_row.get("method", ""),
                    method_children=grader_row.get("method_children", 0),
                    class_name=grader_row.get("class_name", ""),
                    class_children=grader_row.get("class_children", 0),
                )
                status = "pass" if passed else "fail"
                metrics = {
                    "elapsed_seconds": generation.get("elapsed_seconds"),
                    "request_count": generation.get("request_count"),
                    "tool_event_count": 0,
                    "reward": 1.0 if passed else 0.0,
                    "eval_reason": reason,
                }
                artifacts = {key: generation[key] for key in (
                    "solution_path", "solution_sha256", "response_path", "response_sha256"
                ) if key in generation}
                builder = build_aider_trial_record

            else:
                raise ContractError(f"unknown suite {suite_id}")

        if status in {"infra_error", "timeout", "cancelled"}:
            # Generation limits still consume time; retain measured latency.
            metrics.setdefault("elapsed_seconds", generation.get("elapsed_seconds"))
            builder = (
                build_swe_trial_record if suite_id == "swe-verified" else
                build_lcb_trial_record if suite_id == "livecodebench" else
                build_mmlu_trial_record if suite_id == "mmlu-pro" else
                build_aider_trial_record if suite_id == "aider-refactor" else
                build_terminal_trial_record
            )
            artifacts = {key: generation[key] for key in (
                "patch_path", "patch_sha256", "trajectory_path", "trajectory_sha256", "stderr_path",
                "solution_path", "solution_sha256", "response_path", "response_sha256"
            ) if key in generation}

        trial = builder(
            run_id=manifest["run_id"], task_id=task_id, attempt_id=generation["attempt_id"],
            status=status, reason=reason, started_utc=started, ended_utc=utc_now(),
            prompt_tokens=generation.get("prompt_tokens"), generated_tokens=generation.get("generated_tokens"),
            token_source=generation.get("token_source"), finish_reason=generation.get("termination_reason") or "stop",
            http_status=None,
            metrics=metrics,
            artifacts=artifacts,
            error_class=None if status in {"pass", "fail"} else reason,
        )
        append_trial(trials_path, trial)
        counts[status] += 1
        evaluated += 1

    atomic_write_json(run_dir / "evaluation-state.json", {
        "schema_version": SCHEMA_VERSION, "run_id": manifest["run_id"], "updated_utc": utc_now(),
        "evaluated_this_invocation": evaluated, "counts": counts,
    })
    return {"schema_version": SCHEMA_VERSION, "operation": "evaluate", "run_id": manifest["run_id"],
            "profile_id": profile["id"], "suite_id": suite["id"],
            "evaluated_this_invocation": evaluated,
            "remaining": len(manifest["scheduled_task_ids"]) - len(already) - evaluated,
            "counts": counts}
