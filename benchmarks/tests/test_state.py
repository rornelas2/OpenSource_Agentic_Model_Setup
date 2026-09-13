from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from benchmarks.contracts import ContractError, canonical_json_bytes
from benchmarks.state import append_trial, pending_task_ids, read_trials


def trial(task_id: str, attempt_id: str, status: str = "pass") -> dict:
    return {
        "schema_version": "1.0",
        "run_id": "run-1",
        "task_id": task_id,
        "attempt_id": attempt_id,
        "status": status,
        "reason": "completed",
        "started_utc": "2026-09-11T00:00:00Z",
        "ended_utc": "2026-09-11T00:00:01Z",
        "http_status": 200,
        "finish_reason": "stop",
        "prompt_tokens": 10,
        "generated_tokens": 2,
        "token_source": "server_usage",
        "metrics": {"latency_seconds": 1.0},
        "artifacts": {},
        "error_class": None,
    }


class StateTests(unittest.TestCase):
    def test_append_and_resume_treat_failures_as_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trials.jsonl"
            append_trial(path, trial("a", "attempt-a"))
            append_trial(path, trial("b", "attempt-b", "fail"))
            records, recovered = read_trials(path)
            self.assertFalse(recovered)
            self.assertEqual(pending_task_ids(["a", "b", "c"], records), ["c"])
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_partial_trailing_record_is_recovered_but_blocks_append(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trials.jsonl"
            path.write_bytes(canonical_json_bytes(trial("a", "attempt-a")) + b'{"task_id":')
            records, recovered = read_trials(path)
            self.assertTrue(recovered)
            self.assertEqual([record["task_id"] for record in records], ["a"])
            with self.assertRaisesRegex(ContractError, "partial trailing record"):
                append_trial(path, trial("b", "attempt-b"))

    def test_valid_json_without_record_terminator_is_still_partial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trials.jsonl"
            path.write_bytes(canonical_json_bytes(trial("a", "attempt-a")).rstrip(b"\n"))
            records, recovered = read_trials(path)
            self.assertEqual(records, [])
            self.assertTrue(recovered)

    def test_invalid_complete_record_is_not_hidden(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trials.jsonl"
            path.write_bytes(b'{"broken":\n')
            with self.assertRaisesRegex(ContractError, "invalid JSON"):
                read_trials(path)

    def test_duplicate_attempt_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trials.jsonl"
            append_trial(path, trial("a", "attempt-a"))
            with self.assertRaisesRegex(ContractError, "attempt_id already exists"):
                append_trial(path, trial("b", "attempt-a"))


if __name__ == "__main__":
    unittest.main()
