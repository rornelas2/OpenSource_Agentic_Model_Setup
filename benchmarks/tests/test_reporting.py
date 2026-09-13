"""Unit tests for offline reporting recomputation and schema validation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.contracts import ContractError
from benchmarks.reporting import compute_metrics_for_run, generate_report
from benchmarks.state import append_trial


ROOT = Path(__file__).resolve().parents[2]


class ReportingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temp_dir.name) / "run_01"
        self.run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.output_dir = Path(self.temp_dir.name) / "reports"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def make_trial(self, task_id: str, status: str = "pass", attempt_id: str | None = None) -> dict:
        return {
            "schema_version": "1.0",
            "run_id": "test-run",
            "task_id": task_id,
            "attempt_id": attempt_id or f"{task_id}-att-1",
            "status": status,
            "reason": f"test reason {status}",
            "started_utc": "2026-09-11T12:00:00Z",
            "ended_utc": "2026-09-11T12:01:00Z",
            "http_status": 200,
            "finish_reason": "stop",
            "prompt_tokens": 100,
            "generated_tokens": 50,
            "token_source": "server_usage",
            "metrics": {"elapsed_seconds": 60.0, "reward": 1.0 if status == "pass" else 0.0},
            "artifacts": {},
            "error_class": None,
        }

    def test_report_recomputation_and_metrics(self) -> None:
        tasks = [{"task_id": f"task_{i}"} for i in range(1, 5)]
        trials_file = self.run_dir / "trials.jsonl"

        # Write 3 trials: 2 pass, 1 timeout; 1 not run
        append_trial(trials_file, self.make_trial("task_1", "pass"))
        append_trial(trials_file, self.make_trial("task_2", "pass"))
        append_trial(trials_file, self.make_trial("task_3", "timeout"))

        metrics = compute_metrics_for_run(
            run_dir=self.run_dir,
            suite_tasks=tasks,
            is_pilot=False,
            pilot_task_ids=["task_1", "task_2"],
        )

        self.assertEqual(metrics["scheduled_denominator"], 4)
        self.assertEqual(metrics["completed_count"], 3)
        self.assertEqual(metrics["coverage"], 0.75)
        self.assertEqual(metrics["pass_rate"], 0.5)
        self.assertEqual(metrics["mean_reward"], 0.5)
        self.assertEqual(metrics["status_counts"]["pass"], 2)
        self.assertEqual(metrics["status_counts"]["timeout"], 1)
        self.assertEqual(metrics["status_counts"]["not_run"], 1)
        self.assertEqual(metrics["latency_seconds"]["median"], 60.0)
        self.assertEqual(metrics["tokens"]["total_generated_tokens"], 150)

    def test_report_fails_on_duplicate_attempt_or_missing_trials(self) -> None:
        tasks = [{"task_id": "t1"}]
        # No trials file
        with self.assertRaises(ContractError):
            compute_metrics_for_run(self.run_dir, tasks, False, ["t1"])

    def test_zero_latency_is_measured_and_missing_latency_is_explicit(self) -> None:
        measured = self.make_trial("one")
        measured["metrics"] = {"elapsed_seconds": 0.0}
        missing = self.make_trial("two", "timeout")
        missing["metrics"] = {}
        append_trial(self.run_dir / "trials.jsonl", measured)
        append_trial(self.run_dir / "trials.jsonl", missing)
        metrics = compute_metrics_for_run(self.run_dir, [{"task_id": "one"}, {"task_id": "two"}], False, [])
        self.assertEqual(metrics["latency_seconds"]["median"], 0.0)
        self.assertEqual(metrics["latency_seconds"]["sample_count"], 1)
        self.assertEqual(metrics["latency_seconds"]["missing_count"], 1)

    def test_report_rejects_duplicate_attempt(self) -> None:
        tasks = [{"task_id": "t1"}]
        # With trials file containing duplicate attempt_id
        trials_file = self.run_dir / "trials.jsonl"
        t1 = self.make_trial("t1", "pass", attempt_id="dup_att")
        t2 = self.make_trial("t1", "pass", attempt_id="dup_att")
        # Direct write to bypass append_trial check to simulate corrupted state
        trials_file.write_text(json.dumps(t1) + "\n" + json.dumps(t2) + "\n")
        with self.assertRaises(ContractError):
            compute_metrics_for_run(self.run_dir, tasks, False, ["t1"])

    def test_generate_report_produces_artifacts(self) -> None:
        manifest = {
            "schema_version": "1.0",
            "run_id": "test-run",
            "profile_id": "muse-dynamic-a100",
            "suite_id": "terminal-bench",
            "pilot": True,
            "scheduled_task_ids": [
                "cancel-async-tasks", "dna-assembly", "fix-code-vulnerability",
                "qemu-alpine-ssh", "raman-fitting",
            ],
        }
        (self.run_dir / "manifest.json").write_text(json.dumps(manifest) + "\n")
        trials_file = self.run_dir / "trials.jsonl"

        # Append one pass trial for a terminal-bench pilot task
        append_trial(trials_file, self.make_trial("cancel-async-tasks", "pass"))
        for task_id in manifest["scheduled_task_ids"][1:]:
            append_trial(trials_file, self.make_trial(task_id, "not_run"))

        summary = generate_report(self.run_dir, self.output_dir, ROOT)
        self.assertEqual(summary["runs_count"], 1)
        self.assertTrue((self.output_dir / "summary.json").exists())
        self.assertTrue((self.output_dir / "summary.csv").exists())
        self.assertTrue((self.output_dir / "report.md").exists())

        md_content = (self.output_dir / "report.md").read_text()
        self.assertIn("Benchmark Results Report", md_content)
        self.assertIn("muse-dynamic-a100", md_content)
        self.assertIn("terminal-bench", md_content)

    def test_multiple_terminal_attempts_for_one_task_are_rejected(self) -> None:
        tasks = [{"task_id": "t1"}]
        trials_file = self.run_dir / "trials.jsonl"
        append_trial(trials_file, self.make_trial("t1", "pass", "attempt-1"))
        append_trial(trials_file, self.make_trial("t1", "fail", "attempt-2"))
        with self.assertRaisesRegex(ContractError, "multiple terminal trials"):
            compute_metrics_for_run(self.run_dir, tasks, False, [], scheduled_task_ids=["t1"])


if __name__ == "__main__":
    unittest.main()
