"""Allocation, listener, endpoint, GPU, and sandbox preflight checks."""

from __future__ import annotations

import json
import os
import pwd
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from benchmarks import SCHEMA_VERSION
from benchmarks.contracts import ContractError
from benchmarks.transport import ModelTransportClient, ModelTransportConfig


def owns_ipv4_loopback_listener(pid: int, port: int) -> bool:
    try:
        sockets: set[str] = set()
        for descriptor in Path(f"/proc/{pid}/fd").iterdir():
            try:
                sockets.add(descriptor.readlink().name)
            except FileNotFoundError:
                continue
        for line in Path("/proc/net/tcp").read_text(encoding="ascii").splitlines()[1:]:
            fields = line.split()
            if (
                fields[1] == f"0100007F:{port:04X}"
                and fields[3] == "0A"
                and f"socket:[{fields[9]}]" in sockets
            ):
                return True
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return False
    return False


def _endpoint_json(opener: urllib.request.OpenerDirector, url: str) -> tuple[int, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with opener.open(request, timeout=10) as response:
            body = response.read(1024 * 1024 + 1)
            if len(body) > 1024 * 1024:
                raise ContractError(f"endpoint response exceeds 1 MiB: {url}")
            if not body:
                return response.status, None
            return response.status, json.loads(body)
    except (OSError, UnicodeError, json.JSONDecodeError, urllib.error.URLError) as exc:
        raise ContractError(f"endpoint check failed for {url}: {exc}") from exc


def check_endpoint(profile: dict[str, Any], pid: int) -> dict[str, Any]:
    parsed = urlparse(profile["endpoint"]["base_url"])
    port = parsed.port
    if port is None or not owns_ipv4_loopback_listener(pid, port):
        raise ContractError(f"PID {pid} does not own the expected 127.0.0.1:{port} listener")
    root = f"{parsed.scheme}://{parsed.hostname}:{port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    health_status, _ = _endpoint_json(opener, f"{root}/health")
    if health_status != 200:
        raise ContractError(f"health endpoint returned HTTP {health_status}")
    models_status, models = _endpoint_json(opener, f"{root}/v1/models")
    if models_status != 200 or not isinstance(models, dict):
        raise ContractError("models endpoint did not return a JSON object with HTTP 200")
    model_ids = [item.get("id") for item in models.get("data", []) if isinstance(item, dict)]
    expected_alias = profile["endpoint"]["model_alias"]
    if model_ids != [expected_alias]:
        raise ContractError(f"expected only model alias {expected_alias!r}, received {model_ids!r}")
    return {"listener_pid": pid, "port": port, "model_alias": expected_alias}


def check_gpu(profile: dict[str, Any]) -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"(?:[0-9]+|GPU-[a-fA-F0-9-]+)", visible):
        raise ContractError("CUDA_VISIBLE_DEVICES must identify exactly one allocated GPU")
    try:
        result = subprocess.run(
            ["nvidia-smi", "-i", visible, "--query-gpu=name,uuid,memory.total", "--format=csv,noheader,nounits"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ContractError(f"cannot inspect the allocated GPU: {exc}") from exc
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise ContractError(f"preflight requires exactly one visible GPU, found {len(rows)}")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 3:
        raise ContractError("unexpected nvidia-smi output")
    name, uuid, memory_mib = fields
    expected = profile["deployment"]["gpu"].replace("nvidia_", "").replace("_nvl", "").lower()
    if expected not in name.lower().replace(" ", ""):
        raise ContractError(f"allocated GPU {name!r} does not match profile {expected!r}")
    try:
        memory = int(memory_mib)
    except ValueError as exc:
        raise ContractError("nvidia-smi returned invalid memory.total") from exc
    return {"name": name, "uuid": uuid, "memory_mib": memory}


def check_container_runtime(required: bool) -> dict[str, Any]:
    if not required:
        return {"required": False, "status": "not_required"}
    docker = shutil.which("docker")
    if docker:
        try:
            result = subprocess.run(
                [docker, "info", "--format", "{{json .ServerVersion}}"], check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise ContractError(f"suite requires a working unprivileged Docker runtime: {exc}") from exc
        version = result.stdout.strip()
        if not version or version == "null":
            raise ContractError("Docker did not report a server version")
        try:
            decoded_version = json.loads(version)
        except json.JSONDecodeError as exc:
            raise ContractError("Docker returned malformed server version JSON") from exc
        return {
            "required": True,
            "status": "available",
            "kind": "docker",
            "executable": docker,
            "server_version": decoded_version,
            "docker_host": os.environ.get("DOCKER_HOST"),
        }

    podman = shutil.which("podman")
    if podman:
        username = pwd.getpwuid(os.getuid()).pw_name
        uid_ranges: list[str] = []
        gid_ranges: list[str] = []
        for filename, target in (("/etc/subuid", uid_ranges), ("/etc/subgid", gid_ranges)):
            try:
                for line in Path(filename).read_text(encoding="ascii").splitlines():
                    if line.split(":", 1)[0] in {username, str(os.getuid())}:
                        target.append(line)
            except OSError:
                pass
        if not uid_ranges or not gid_ranges:
            raise ContractError(
                "Podman is installed but this account has no subordinate UID/GID ranges in "
                "/etc/subuid and /etc/subgid; official Docker-compatible benchmark containers "
                "cannot unpack safely. Ask the cluster administrator to provision subuid/subgid "
                "ranges or provide a supported Docker service."
            )
        raise ContractError(
            "Podman is installed, but no Docker-compatible API endpoint is configured. Start a "
            "rootless Podman service in the allocation, set DOCKER_HOST to its Unix socket, and "
            "install a Docker-compatible client before preparing this suite."
        )

    raise ContractError(
        "suite requires Docker or a validated rootless Docker-compatible runtime; none is installed"
    )


def check_system_resources(profile: dict[str, Any]) -> dict[str, Any]:
    cpus = os.cpu_count() or 1
    mem_total_gib = 0.0
    try:
        meminfo = Path("/proc/meminfo").read_text()
        for line in meminfo.splitlines():
            if line.startswith("MemTotal:"):
                kb = int(line.split()[1])
                mem_total_gib = round(kb / (1024 * 1024), 1)
                break
    except Exception:
        pass

    disk = shutil.disk_usage("/tmp")
    free_gib = round(disk.free / (1024 * 1024 * 1024), 1)
    if free_gib < 1.0:
        raise ContractError(f"insufficient disk space on /tmp: {free_gib} GiB available")

    requested_cpus = 8
    allocated_cpus_text = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated_cpus_text and allocated_cpus_text.isdecimal() and int(allocated_cpus_text) < requested_cpus:
        raise ContractError(
            f"allocation has {allocated_cpus_text} CPUs per task; benchmark requires at least {requested_cpus}"
        )
    requested_mem = profile.get("deployment", {}).get("system_ram_gib")
    slurm_mem_text = os.environ.get("SLURM_MEM_PER_NODE")
    if requested_mem and slurm_mem_text and slurm_mem_text.isdecimal():
        allocated_gib = int(slurm_mem_text) / 1024
        if allocated_gib + 0.01 < float(requested_mem):
            raise ContractError(
                f"allocation has {allocated_gib:.1f} GiB RAM; profile requires {requested_mem} GiB"
            )

    return {
        "cpus_available": cpus,
        "slurm_cpus_per_task": int(allocated_cpus_text) if allocated_cpus_text and allocated_cpus_text.isdecimal() else None,
        "mem_total_gib": mem_total_gib,
        "slurm_mem_per_node_mib": int(slurm_mem_text) if slurm_mem_text and slurm_mem_text.isdecimal() else None,
        "tmp_disk_free_gib": free_gib,
    }


def validate_endpoint_capabilities(profile: dict[str, Any]) -> dict[str, Any]:
    """Test model capabilities through common transport."""
    client = ModelTransportClient(
        ModelTransportConfig(
            base_url=profile["endpoint"]["base_url"],
            expected_model_alias=profile["endpoint"]["model_alias"],
            connect_timeout=5.0,
            read_timeout=30.0,
        )
    )
    client.check_model_alias()
    res_chat = client.chat_completion([{"role": "user", "content": "Respond with exactly READY"}], stream=False, max_tokens=16)
    res_stream = client.chat_completion([{"role": "user", "content": "Say hello in one short sentence."}], stream=True, max_tokens=32)
    if not (res_chat.content or res_chat.reasoning):
        raise ContractError("non-streaming capability probe returned no content or reasoning")
    if not (res_stream.content or res_stream.reasoning):
        raise ContractError("streaming capability probe returned no content or reasoning")
    reasoning_observed = bool(res_chat.reasoning or res_stream.reasoning)
    if profile["deployment"]["reasoning"] != "off" and not reasoning_observed:
        raise ContractError("profile requires reasoning content, but neither capability probe exposed it")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup_value",
                "description": "Look up one named deterministic value.",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": False,
                },
            },
        }
    ]
    tool_response = client.chat_completion(
        [{"role": "user", "content": "Call lookup_value twice in this response: once with name alpha and once with name beta."}],
        stream=True,
        tools=tools,
        tool_choice="required",
        max_tokens=1024,
    )
    if not (1 <= len(tool_response.tool_calls) <= 2):
        raise ContractError(
            f"tool probe expected 1 or 2 calls, received {len(tool_response.tool_calls)}"
        )
    names = []
    for call in tool_response.tool_calls:
        if call.name != "lookup_value":
            raise ContractError(f"tool probe returned unexpected tool {call.name!r}")
        arguments = json.loads(call.arguments)
        names.append(arguments.get("name"))
    if not (set(names) <= {"alpha", "beta"} and len(names) > 0):
        raise ContractError(f"tool probe returned wrong arguments: {names!r}")
    continuation_messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Call lookup_value twice in this response: once with name alpha and once with name beta."},
        {
            "role": "assistant",
            "content": tool_response.content or None,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }
                for call in tool_response.tool_calls
            ],
        },
    ]
    continuation_messages.extend(
        {"role": "tool", "tool_call_id": call.id, "content": f"value-for-{json.loads(call.arguments)['name']}"}
        for call in tool_response.tool_calls
    )
    continuation_messages.append({"role": "user", "content": "Acknowledge the tool results briefly."})
    continuation = client.chat_completion(continuation_messages, stream=True, max_tokens=128)
    if not (continuation.content or continuation.reasoning):
        raise ContractError("tool-result continuation returned no content")

    cancel_checks = 0
    def cancel_after_stream_starts() -> bool:
        nonlocal cancel_checks
        cancel_checks += 1
        return cancel_checks >= 3
    cancelled = False
    try:
        client.chat_completion(
            [{"role": "user", "content": "Count slowly from one to one hundred."}],
            stream=True,
            max_tokens=256,
            cancellation_check=cancel_after_stream_starts,
        )
    except Exception as exc:
        from benchmarks.transport import RequestCancelledError
        if isinstance(exc, RequestCancelledError):
            cancelled = True
        else:
            raise
    if not cancelled:
        raise ContractError("cancellation probe completed before the client could cancel it")
    recovery = client.chat_completion(
        [{"role": "user", "content": "Respond with exactly RECOVERED"}], stream=False, max_tokens=16
    )
    if not (recovery.content or recovery.reasoning):
        raise ContractError("post-cancellation recovery request returned no output")

    context_tokens = int(profile["deployment"]["context_tokens"])
    deliberately_overlong = " ".join(f"token{i}" for i in range(context_tokens + 1024))
    over_context_rejected = False
    try:
        client.chat_completion(
            [{"role": "user", "content": deliberately_overlong}],
            stream=False,
            max_tokens=profile["deployment"]["output_tokens"],
        )
    except Exception as exc:
        from benchmarks.transport import ModelRequestError
        if isinstance(exc, ModelRequestError) and exc.status in {400, 413, 422}:
            over_context_rejected = True
        else:
            raise
    if not over_context_rejected:
        raise ContractError("over-context request was not rejected explicitly")
    return {
        "models_verified": True,
        "non_stream_passed": True,
        "streaming_passed": True,
        "reasoning_observed": reasoning_observed,
        "reasoning_expected": profile["deployment"]["reasoning"] != "off",
        "multiple_tool_calls_passed": True,
        "tool_results_passed": True,
        "cancellation_recovery_passed": True,
        "over_context_rejected": True,
    }


def run_preflight(
    profile: dict[str, Any],
    suite: dict[str, Any],
    validate_capabilities: bool = True,
) -> dict[str, Any]:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or not job_id.isascii() or not job_id.isdecimal():
        raise ContractError("preflight must run inside a numeric Slurm allocation")
    pid_text = os.environ.get(profile["server_pid_env"])
    if not pid_text or not pid_text.isascii() or not pid_text.isdecimal() or int(pid_text) <= 1:
        raise ContractError(f"{profile['server_pid_env']} must identify the owned server process")

    endpoint_info = check_endpoint(profile, int(pid_text))
    capabilities = {}
    if validate_capabilities:
        capabilities = validate_endpoint_capabilities(profile)

    return {
        "schema_version": SCHEMA_VERSION,
        "profile_id": profile["id"],
        "suite_id": suite["id"],
        "slurm_job_id": job_id,
        "endpoint": endpoint_info,
        "capabilities": capabilities,
        "gpu": check_gpu(profile),
        "resources": check_system_resources(profile),
        "container": check_container_runtime(suite["execution"]["requires_container"]),
        "status": "pass",
    }
