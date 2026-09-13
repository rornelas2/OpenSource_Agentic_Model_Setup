"""Unit tests for telemetry sampler and metrics aggregation."""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from benchmarks.telemetry import (
    TelemetrySample,
    TelemetrySampler,
    sample_cgroup_memory,
    sample_process,
)


class TelemetryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.telemetry_path = Path(self.temp_dir.name) / "telemetry.jsonl"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_sample_current_process(self) -> None:
        metrics = sample_process(os.getpid())
        self.assertIsNotNone(metrics)
        self.assertEqual(metrics.pid, os.getpid())
        # Check RSS is sampled (on Linux /proc/<pid>/status)
        if Path(f"/proc/{os.getpid()}/status").exists():
            self.assertIsNotNone(metrics.rss_kib)
            self.assertGreater(metrics.rss_kib, 0)

    def test_sampler_lifecycle_and_output(self) -> None:
        sampler = TelemetrySampler(
            telemetry_file=self.telemetry_path,
            server_pid=os.getpid(),
            sample_interval_seconds=0.2,
        )
        sampler.start()
        time.sleep(0.5)
        sampler.stop()

        self.assertTrue(self.telemetry_path.exists())
        lines = [ln for ln in self.telemetry_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        self.assertGreaterEqual(len(lines), 1)

        summary = sampler.summary()
        self.assertGreaterEqual(summary["samples_count"], 1)
        self.assertIsNotNone(summary["peak_server_rss_kib"])
        self.assertIsNotNone(summary["peak_client_rss_kib"])


if __name__ == "__main__":
    unittest.main()
