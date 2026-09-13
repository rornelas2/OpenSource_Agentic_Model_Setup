"""GPU, process, and cgroup telemetry sampler with bounded sampling rate.

Collects 1 Hz samples for server/client processes, visible GPU metrics via nvidia-smi,
and system cgroups. Writes append-only telemetry records and computes summary statistics.
"""

from __future__ import annotations

import dataclasses
import datetime
import fcntl
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from benchmarks.contracts import canonical_json_bytes
from benchmarks.state import fsync_directory


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


@dataclasses.dataclass
class GpuMetrics:
    name: str | None = None
    uuid: str | None = None
    memory_used_mib: int | None = None
    memory_total_mib: int | None = None
    utilization_gpu_pct: int | None = None
    power_draw_w: float | None = None
    temperature_c: int | None = None


@dataclasses.dataclass
class ProcessMetrics:
    pid: int
    rss_kib: int | None = None
    num_fds: int | None = None


@dataclasses.dataclass
class TelemetrySample:
    timestamp_utc: str
    elapsed_seconds: float
    gpu: GpuMetrics | None = None
    server_process: ProcessMetrics | None = None
    client_process: ProcessMetrics | None = None
    cgroup_memory_bytes: int | None = None

    def to_dict(self) -> dict[str, Any]:
        res: dict[str, Any] = {
            "timestamp_utc": self.timestamp_utc,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }
        if self.gpu is not None:
            res["gpu"] = dataclasses.asdict(self.gpu)
        if self.server_process is not None:
            res["server_process"] = dataclasses.asdict(self.server_process)
        if self.client_process is not None:
            res["client_process"] = dataclasses.asdict(self.client_process)
        if self.cgroup_memory_bytes is not None:
            res["cgroup_memory_bytes"] = self.cgroup_memory_bytes
        return res


def sample_gpu(gpu_uuid: str | None = None) -> GpuMetrics | None:
    """Sample visible GPU metrics using nvidia-smi query."""
    query = "name,uuid,memory.used,memory.total,utilization.gpu,power.draw,temperature.gpu"
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                continue
            name, uuid, mem_used, mem_tot, util, power, temp = parts[:7]
            if gpu_uuid and uuid != gpu_uuid and len(lines) > 1:
                continue
            try:
                return GpuMetrics(
                    name=name,
                    uuid=uuid,
                    memory_used_mib=int(mem_used) if mem_used.isdigit() else None,
                    memory_total_mib=int(mem_tot) if mem_tot.isdigit() else None,
                    utilization_gpu_pct=int(util) if util.isdigit() else None,
                    power_draw_w=float(power) if power.replace(".", "", 1).isdigit() else None,
                    temperature_c=int(temp) if temp.isdigit() else None,
                )
            except ValueError:
                continue
    except (OSError, subprocess.SubprocessError):
        return None
    return None


def sample_process(pid: int) -> ProcessMetrics | None:
    """Sample process memory (RSS) and open file descriptors via /proc."""
    if pid <= 0:
        return None
    rss_kib: int | None = None
    num_fds: int | None = None
    try:
        status_path = Path(f"/proc/{pid}/status")
        if status_path.exists():
            for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        rss_kib = int(parts[1])
                        break
        fd_dir = Path(f"/proc/{pid}/fd")
        if fd_dir.is_dir():
            num_fds = sum(1 for _ in fd_dir.iterdir())
        return ProcessMetrics(pid=pid, rss_kib=rss_kib, num_fds=num_fds)
    except (OSError, PermissionError, ProcessLookupError):
        return None


def sample_cgroup_memory() -> int | None:
    """Sample cgroup v1 or v2 current memory usage if accessible."""
    candidates = [
        Path("/sys/fs/cgroup/memory.current"),
        Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ]
    for c in candidates:
        try:
            if c.is_file():
                val = c.read_text().strip()
                if val.isdigit():
                    return int(val)
        except (OSError, PermissionError):
            pass
    return None


class TelemetrySampler:
    """Background sampler running at bounded rate (default 1 Hz)."""

    def __init__(
        self,
        telemetry_file: Path,
        server_pid: int | None = None,
        gpu_uuid: str | None = None,
        sample_interval_seconds: float = 1.0,
    ) -> None:
        self.telemetry_file = telemetry_file
        self.server_pid = server_pid
        self.gpu_uuid = gpu_uuid
        self.sample_interval = max(0.2, sample_interval_seconds)
        self.client_pid = os.getpid()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time = 0.0
        self._lock = threading.Lock()
        self._samples_count = 0
        self._gpu_util_sum = 0
        self._gpu_util_count = 0
        self._peak_gpu_mem: int | None = None
        self._peak_gpu_util: int | None = None
        self._peak_server_rss_kib: int | None = None
        self._peak_client_rss_kib: int | None = None

    def start(self) -> None:
        self._start_time = time.monotonic()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="TelemetrySampler")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def set_client_pid(self, pid: int) -> None:
        """Switch process sampling to the active client subprocess."""
        if pid <= 0:
            raise ValueError("client PID must be positive")
        with self._lock:
            self.client_pid = pid

    def _run(self) -> None:
        while not self._stop_event.is_set():
            sample = self.sample_once()
            with self._lock:
                self._samples_count += 1
                if sample.gpu and sample.gpu.memory_used_mib is not None:
                    value = sample.gpu.memory_used_mib
                    self._peak_gpu_mem = value if self._peak_gpu_mem is None else max(self._peak_gpu_mem, value)
                if sample.gpu and sample.gpu.utilization_gpu_pct is not None:
                    value = sample.gpu.utilization_gpu_pct
                    self._gpu_util_sum += value
                    self._gpu_util_count += 1
                    self._peak_gpu_util = value if self._peak_gpu_util is None else max(self._peak_gpu_util, value)
                if sample.server_process and sample.server_process.rss_kib is not None:
                    value = sample.server_process.rss_kib
                    self._peak_server_rss_kib = value if self._peak_server_rss_kib is None else max(self._peak_server_rss_kib, value)
                if sample.client_process and sample.client_process.rss_kib is not None:
                    value = sample.client_process.rss_kib
                    self._peak_client_rss_kib = value if self._peak_client_rss_kib is None else max(self._peak_client_rss_kib, value)
            self._append_record(sample.to_dict())
            self._stop_event.wait(self.sample_interval)

    def sample_once(self) -> TelemetrySample:
        elapsed = time.monotonic() - self._start_time
        gpu = sample_gpu(self.gpu_uuid)
        srv = sample_process(self.server_pid) if self.server_pid else None
        with self._lock:
            client_pid = self.client_pid
        cli = sample_process(client_pid)
        cg = sample_cgroup_memory()
        return TelemetrySample(
            timestamp_utc=utc_now(),
            elapsed_seconds=elapsed,
            gpu=gpu,
            server_process=srv,
            client_process=cli,
            cgroup_memory_bytes=cg,
        )

    def _append_record(self, record: dict[str, Any]) -> None:
        payload = canonical_json_bytes(record)
        self.telemetry_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        file_was_missing = not self.telemetry_file.exists()
        fd = os.open(self.telemetry_file, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(fd, "ab", closefd=False) as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
                if file_was_missing:
                    fsync_directory(self.telemetry_file.parent)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def summary(self) -> dict[str, Any]:
        """Compute aggregated telemetry summary metrics."""
        with self._lock:
            return {
                "samples_count": self._samples_count,
                "peak_gpu_memory_mib": self._peak_gpu_mem,
                "peak_gpu_utilization_pct": self._peak_gpu_util,
                "mean_gpu_utilization_pct": (
                    round(self._gpu_util_sum / self._gpu_util_count, 1)
                    if self._gpu_util_count else None
                ),
                "peak_server_rss_kib": self._peak_server_rss_kib,
                "peak_client_rss_kib": self._peak_client_rss_kib,
            }
