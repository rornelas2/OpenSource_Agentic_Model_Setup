"""Unit tests for evaluator preparation validation and failure modes."""

from __future__ import annotations

import tempfile
import types
import unittest
from unittest.mock import patch
from pathlib import Path

from benchmarks.contracts import ContractError
from benchmarks.preparation import (
    canonical_hash,
    ensure_git_checkout,
    execute_prepare,
    validate_swe_dataset,
)


class PreparationFailureModeTests(unittest.TestCase):
    def test_mismatched_task_input_hash_fails(self) -> None:
        expected = [
            {
                "task_id": "test_task_1",
                "input_sha256": "0" * 64,
                "grader_sha256": "1" * 64,
                "stratum": "repo|easy",
            }
        ]
        row = {key: "fixture" for key in (
            "repo", "base_commit", "problem_statement", "patch", "test_patch",
            "FAIL_TO_PASS", "PASS_TO_PASS", "environment_setup_commit",
            "image", "eval_script", "log_parser", "eval_type",
        )}
        row["instance_id"] = "test_task_1"
        dataset = types.SimpleNamespace(load_dataset=lambda *args, **kwargs: [row])
        with patch.dict("sys.modules", {"datasets": dataset}):
            with self.assertRaisesRegex(ContractError, "input hash mismatch"):
                validate_swe_dataset(split="test", revision="fake", expected_tasks=expected)

    def test_missing_task_id_fails(self) -> None:
        expected = [
            {
                "task_id": "non_existent_instance_id_9999",
                "input_sha256": "a" * 64,
                "grader_sha256": "b" * 64,
                "stratum": "repo|easy",
            }
        ]
        dataset = types.SimpleNamespace(load_dataset=lambda *args, **kwargs: [])
        with patch.dict("sys.modules", {"datasets": dataset}):
            with self.assertRaisesRegex(ContractError, "missing expected task ID"):
                validate_swe_dataset(split="test", revision="c104f840cc67f8b6eec6f759ebc8b2693d585d4a", expected_tasks=expected)

    def test_immutable_manifest_conflict_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "manifest.json"
            manifest_path.write_text('{"schema_version": "1.0", "different": true}\n')

            suites = {
                "mock-suite": (
                    {
                        "id": "mock-suite",
                        "source": {
                            "evaluator_repository": "mock/eval",
                            "evaluator_revision": "0" * 40,
                            "dataset_repository": "mock/ds",
                            "dataset_revision": "1" * 40,
                            "split": "test",
                        },
                        "execution": {"requires_container": False},
                    },
                    [],
                )
            }
            with self.assertRaises(ContractError) as cm:
                execute_prepare("mock-suite", manifest_path, suites, Path(tmpdir))
            self.assertIn("already exists with different contents", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
