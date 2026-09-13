#!/usr/bin/env python3
"""Own one profile server and one benchmark client in the same allocation."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from benchmarks.contracts import ContractError, load_profiles
from benchmarks.preflight import check_endpoint
from benchmarks.state import atomic_write_json


ROOT = Path(__file__).resolve().parent.parent


def server_environment(profile: dict) -> dict[str, str]:
    """Make the launched profile independent of a previous interactive session."""
    env = dict(os.environ)
    deployment = profile["deployment"]
    prefix = {"gemma": "GEMMA", "muse": "MUSE", "lightning": "NEMOTRON", "super": "NEMOTRON"}[
        profile["id"].split("-", 1)[0]
    ]
    for suffix, value in {
        "GPU": deployment["gpu"], "CONTEXT": deployment["context_tokens"],
        "OUTPUT": deployment["output_tokens"], "PARALLEL": deployment["sequences"],
        "PORT": 8000,
    }.items():
        env[f"{prefix}_{suffix}"] = str(value)
    return env


def stop_owned(
    process: subprocess.Popen[bytes] | None,
    graceful_signal: signal.Signals = signal.SIGTERM,
) -> None:
    if process is None:
        return
    try:
        os.killpg(process.pid, graceful_signal)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        pass
    # The leader can exit before workers that ignored SIGTERM. The whole
    # process group belongs to this runner and must stop before releasing it.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=20)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--log-dir", required=True, type=Path)
    parser.add_argument("client", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    client_argv = args.client[1:] if args.client[:1] == ["--"] else args.client
    if not client_argv:
        parser.error("a client command is required after --")
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or not job_id.isascii() or not job_id.isdecimal():
        raise ContractError("server runner must execute inside a numeric Slurm allocation")
    profiles = load_profiles(ROOT / "benchmarks/profiles.json", ROOT)
    if args.profile not in profiles:
        raise ContractError(f"unknown profile: {args.profile}")
    profile = profiles[args.profile]
    log_dir = args.log_dir.expanduser().resolve()
    if log_dir in {Path("/").resolve(), Path.home().resolve(), ROOT.resolve()}:
        raise ContractError(f"refusing broad server log directory: {log_dir}")
    log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(log_dir, 0o700)
    server_argv = ["bash", str(profile["_resolved_files"]["serve"])]
    server: subprocess.Popen[bytes] | None = None
    client: subprocess.Popen[bytes] | None = None
    interrupted: list[int] = []

    def handle_signal(signum: int, _frame: object) -> None:
        interrupted.append(signum)

    old_handlers = {sig: signal.signal(sig, handle_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with (log_dir / "server.log").open("xb", buffering=0) as server_log:
            server = subprocess.Popen(server_argv, env=server_environment(profile), stdout=server_log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 600
            while True:
                if interrupted:
                    raise ContractError(f"interrupted by signal {interrupted[0]} during server startup")
                if server.poll() is not None:
                    raise ContractError(f"server exited during startup with status {server.returncode}")
                try:
                    endpoint = check_endpoint(profile, server.pid)
                    break
                except ContractError:
                    if time.monotonic() >= deadline:
                        raise ContractError("owned server readiness exceeded 600 seconds")
                    time.sleep(1)
            child_env = dict(os.environ)
            child_env[profile["server_pid_env"]] = str(server.pid)
            child_env["BENCHMARK_SERVER_LOG"] = str((log_dir / "server.log").resolve())
            atomic_write_json(log_dir / "server-manifest.json", {
                "profile_id": profile["id"], "slurm_job_id": job_id,
                "server_pid": server.pid, "server_argv": server_argv,
                "client_argv": client_argv, "endpoint": endpoint,
            })
            client = subprocess.Popen(client_argv, env=child_env, start_new_session=True)
            while client.poll() is None:
                if interrupted:
                    # Give the benchmark client a chance to journal the active
                    # task as cancelled and fsync its telemetry before teardown.
                    stop_owned(client, signal.SIGINT)
                    return 128 + interrupted[0]
                if server.poll() is not None:
                    stop_owned(client)
                    raise ContractError("owned server exited while benchmark client was running")
                time.sleep(0.25)
            return client.returncode if client.returncode is not None and client.returncode >= 0 else 1
    finally:
        stop_owned(client)
        stop_owned(server)
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, OSError, subprocess.SubprocessError) as exc:
        print(f"benchmark server runner: {exc}", file=sys.stderr)
        raise SystemExit(2)
