"""Bounded execution of the pinned OpenCode CLI inside an existing container.

The caller supplies an official benchmark image by immutable image ID and a
pre-created internal container network whose only host service is the owned
model-endpoint relay. No host home directory, environment, repository, tests,
or credentials are mounted into the task container.
"""

from __future__ import annotations

import json
import os
import selectors
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from benchmarks.contracts import ContractError, canonical_json_bytes, sha256_file
from benchmarks.state import atomic_write_json


MAX_EVENT_LINE_BYTES = 1024 * 1024
MAX_TRAJECTORY_BYTES = 32 * 1024 * 1024


def _run(argv: list[str], *, timeout: int = 60) -> str:
    try:
        result = subprocess.run(
            argv,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        raise ContractError(f"container command failed: {argv!r}: {detail}") from exc
    return result.stdout.strip()


def build_benchmark_config(
    provider_config: Path,
    *,
    endpoint_base_url: str,
    model_alias: str,
    context_tokens: int,
    output_tokens: int,
) -> dict[str, Any]:
    """Derive the task-local config without mutating the public provider file."""
    try:
        config = json.loads(provider_config.read_text(encoding="utf-8"))
        provider = config["provider"]["pinnacles"]
        model = provider["models"][model_alias]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ContractError(f"cannot derive OpenCode config from {provider_config}: {exc}") from exc
    provider["options"]["baseURL"] = endpoint_base_url
    model["limit"] = {"context": context_tokens, "output": output_tokens}
    config["model"] = f"pinnacles/{model_alias}"
    config["small_model"] = f"pinnacles/{model_alias}"
    # Arbitrary shell commands are acceptable only because this file is copied
    # into a resource-limited, credential-free benchmark container.
    config["permission"] = {
        "*": "deny",
        "read": "allow",
        "glob": "allow",
        "grep": "allow",
        "edit": "allow",
        "write": "allow",
        "patch": "allow",
        "bash": "allow",
        "task": "deny",
    }
    return config


def _inspect_image_id(docker: str, image: str) -> str:
    return _run([docker, "image", "inspect", "--format", "{{.Id}}", image])


def validate_isolated_network_config(docker: str, network: str, endpoint_base_url: str) -> None:
    if network in {"host", "bridge", "default", "none"} or not network:
        raise ContractError("task containers require a dedicated internal Docker network")
    parsed = urlparse(endpoint_base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname in {None, "127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") != "/v1"
    ):
        raise ContractError("container endpoint must be an uncredentialed internal HTTP relay ending in /v1")
    raw = _run([docker, "network", "inspect", network])
    try:
        inspected = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ContractError(f"Docker returned malformed network metadata: {exc}") from exc
    if (
        not isinstance(inspected, list)
        or len(inspected) != 1
        or not isinstance(inspected[0], dict)
        or inspected[0].get("Name") != network
        or inspected[0].get("Internal") is not True
        or inspected[0].get("Scope") != "local"
    ):
        raise ContractError("task container network is not an exact local internal Docker network")


def run_opencode_in_container(
    *,
    image: str,
    expected_image_id: str,
    network: str,
    config: dict[str, Any],
    opencode_binary: Path,
    expected_opencode_sha256: str,
    prompt: str,
    artifact_dir: Path,
    model_alias: str,
    timeout_seconds: int,
    max_requests: int,
    max_tool_events: int,
    max_generated_tokens: int,
    cpus: int = 8,
    memory_gib: int = 64,
    client_pid_callback: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Run exactly one OpenCode trajectory and return measured metadata."""
    docker = shutil.which("docker")
    if not docker:
        raise ContractError("the official task-container path requires the Docker CLI")
    validate_isolated_network_config(docker, network, config.get("provider", {}).get("pinnacles", {}).get("options", {}).get("baseURL", ""))
    if not opencode_binary.is_file() or not os.access(opencode_binary, os.X_OK):
        raise ContractError(f"pinned OpenCode binary is missing or not executable: {opencode_binary}")
    binary_hash = sha256_file(opencode_binary)
    if binary_hash != expected_opencode_sha256:
        raise ContractError(
            f"OpenCode binary hash mismatch: {binary_hash} != {expected_opencode_sha256}"
        )
    version = _run([str(opencode_binary), "--version"], timeout=30)
    if version != "1.18.30":
        raise ContractError(f"OpenCode version mismatch: {version!r}")
    actual_image_id = _inspect_image_id(docker, image)
    if actual_image_id != expected_image_id:
        raise ContractError(
            f"task image identity changed for {image}: {actual_image_id!r} != {expected_image_id!r}"
        )
    if not isinstance(prompt, str) or not prompt:
        raise ContractError("OpenCode prompt must be a nonempty string")

    artifact_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(artifact_dir, 0o700)
    config_path = artifact_dir / "opencode.json"
    atomic_write_json(config_path, config)
    container_name = f"benchmark-opencode-{uuid.uuid4().hex}"
    trajectory_path = artifact_dir / "trajectory.jsonl"
    stderr_path = artifact_dir / "opencode.stderr"
    started = time.monotonic()
    events: list[dict[str, Any]] = []
    request_count = 0
    tool_count = 0
    generated_tokens = 0
    prompt_tokens = 0
    termination_reason: str | None = None
    exit_code: int | None = None
    first_token_seconds: float | None = None
    first_tool_seconds: float | None = None

    create_argv = [
        docker, "create", "--name", container_name,
        "--network", network,
        "--cpus", str(cpus),
        "--memory", f"{memory_gib}g",
        "--pids-limit", "4096",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--hostname", "benchmark-task",
        "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=4g",
        "--tmpfs", "/benchmark-home:rw,nosuid,nodev,size=2g",
        "--env", "HOME=/benchmark-home",
        "--env", "XDG_CONFIG_HOME=/benchmark-home/.config",
        "--env", "XDG_DATA_HOME=/benchmark-home/.local/share",
        "--env", "XDG_CACHE_HOME=/benchmark-home/.cache",
        "--workdir", "/testbed",
        image,
        "sleep", "infinity",
    ]
    try:
        _run(create_argv)
        _run([docker, "cp", str(opencode_binary), f"{container_name}:/usr/local/bin/opencode"])
        _run([docker, "cp", str(config_path), f"{container_name}:/testbed/opencode.json"])
        _run([docker, "start", container_name])
        _run([docker, "exec", container_name, "chmod", "0555", "/usr/local/bin/opencode"])

        command = [
            docker, "exec",
            "--env", "HOME=/benchmark-home",
            "--env", "XDG_CONFIG_HOME=/benchmark-home/.config",
            "--env", "XDG_DATA_HOME=/benchmark-home/.local/share",
            "--env", "XDG_CACHE_HOME=/benchmark-home/.cache",
            "--workdir", "/testbed",
            container_name,
            "/usr/local/bin/opencode", "run", "--pure", "--format", "json",
            "--model", f"pinnacles/{model_alias}", prompt,
        ]
        with trajectory_path.open("xb", buffering=0) as transcript, stderr_path.open("xb", buffering=0) as errors:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors, start_new_session=True)
            if client_pid_callback is not None:
                client_pid_callback(process.pid)
            assert process.stdout is not None
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            total_bytes = 0
            deadline = started + timeout_seconds
            try:
                while process.poll() is None:
                    if time.monotonic() >= deadline:
                        termination_reason = "wall_time_limit"
                        break
                    ready = selector.select(timeout=0.25)
                    if not ready:
                        continue
                    raw = process.stdout.readline(MAX_EVENT_LINE_BYTES + 1)
                    if not raw:
                        continue
                    if len(raw) > MAX_EVENT_LINE_BYTES:
                        termination_reason = "event_line_size_limit"
                        break
                    total_bytes += len(raw)
                    if total_bytes > MAX_TRAJECTORY_BYTES:
                        termination_reason = "trajectory_size_limit"
                        break
                    transcript.write(raw)
                    if not raw.endswith(b"\n"):
                        termination_reason = "truncated_event"
                        break
                    try:
                        event = json.loads(raw)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        termination_reason = "malformed_event"
                        break
                    if not isinstance(event, dict):
                        termination_reason = "malformed_event"
                        break
                    events.append(event)
                    event_type = event.get("type")
                    part = event.get("part", {})
                    event_text = part.get("text") if isinstance(part, dict) else None
                    if event_type in {"text", "reasoning"} and event_text and first_token_seconds is None:
                        first_token_seconds = time.monotonic() - started
                    if event_type == "step_finish":
                        request_count += 1
                        tokens = part.get("tokens", {}) if isinstance(part, dict) else {}
                        if isinstance(tokens, dict):
                            inp = tokens.get("input")
                            out = tokens.get("output")
                            reasoning = tokens.get("reasoning")
                            if isinstance(inp, int) and inp >= 0:
                                prompt_tokens += inp
                            if isinstance(out, int) and out >= 0:
                                generated_tokens += out
                            if isinstance(reasoning, int) and reasoning >= 0:
                                generated_tokens += reasoning
                    elif event_type == "tool_use":
                        tool_count += 1
                        if first_tool_seconds is None:
                            first_tool_seconds = time.monotonic() - started
                    if request_count >= max_requests:
                        termination_reason = "model_request_limit"
                        break
                    if tool_count >= max_tool_events:
                        termination_reason = "tool_event_limit"
                        break
                    if generated_tokens >= max_generated_tokens:
                        termination_reason = "generated_token_limit"
                        break
                if termination_reason:
                    _run([docker, "kill", container_name], timeout=30)
                try:
                    exit_code = process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    exit_code = process.wait(timeout=15)
                if exit_code not in {0, None} and termination_reason is None:
                    termination_reason = "client_nonzero_exit"
                if exit_code == 0 and request_count == 0 and termination_reason is None:
                    termination_reason = "missing_completion_event"
            finally:
                selector.close()
            transcript.flush()
            os.fsync(transcript.fileno())
            errors.flush()
            os.fsync(errors.fileno())

        if termination_reason:
            # docker kill preserves the writable layer. Restart only the fixed
            # sleep entrypoint so git can serialize the partial patch.
            _run([docker, "start", container_name], timeout=30)
        patch = _run([docker, "exec", "--workdir", "/testbed", container_name, "git", "diff", "--binary"], timeout=60)
        patch_path = artifact_dir / "model.patch"
        with patch_path.open("xb") as output:
            output.write((patch + ("\n" if patch else "")).encode("utf-8"))
            output.flush()
            os.fsync(output.fileno())
        directory_fd = os.open(artifact_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        subprocess.run(
            [docker, "rm", "--force", container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
            check=False,
        )

    ended = time.monotonic()
    return {
        "exit_code": exit_code,
        "termination_reason": termination_reason,
        "elapsed_seconds": ended - started,
        "first_token_seconds": first_token_seconds,
        "first_tool_seconds": first_tool_seconds,
        "request_count": request_count,
        "tool_event_count": tool_count,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "token_source": "opencode_step_finish",
        "visible_output": "".join(
            event.get("part", {}).get("text", "")
            for event in events
            if event.get("type") == "text" and isinstance(event.get("part"), dict)
        )[:65536],
        "visible_output_truncated": sum(
            len(event.get("part", {}).get("text", ""))
            for event in events
            if event.get("type") == "text" and isinstance(event.get("part"), dict)
        ) > 65536,
        "tool_events": [
            {
                "timestamp": event.get("timestamp"),
                "session_id": event.get("sessionID"),
                "tool": event.get("part", {}).get("tool"),
                "call_id": event.get("part", {}).get("callID"),
                "status": event.get("part", {}).get("state", {}).get("status"),
            }
            for event in events
            if event.get("type") == "tool_use" and isinstance(event.get("part"), dict)
        ],
        "patch_path": str((artifact_dir / "model.patch").resolve()),
        "patch_sha256": sha256_file(artifact_dir / "model.patch"),
        "trajectory_path": str(trajectory_path.resolve()),
        "trajectory_sha256": sha256_file(trajectory_path),
        "stderr_path": str(stderr_path.resolve()),
        "opencode_sha256": binary_hash,
        "container_create_argv": create_argv,
        "client_argv": command,
    }
