from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks/run.py"


class CliTests(unittest.TestCase):
    def test_dry_run_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            run_dir = Path(parent) / "absent-run"
            result = subprocess.run(
                [
                    "python3", str(RUNNER), "generate",
                    "--profile", "gemma-qat-a100", "--suite", "mmlu-pro",
                    "--run-dir", str(run_dir), "--pilot", "--dry-run",
                ],
                cwd=ROOT, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            plan = json.loads(result.stdout)
            self.assertFalse(run_dir.exists())
            self.assertEqual(plan["pending_count"], 28)
            self.assertEqual(plan["side_effects"], [])

    def test_non_dry_generation_is_closed(self) -> None:
        with tempfile.TemporaryDirectory() as run_dir:
            result = subprocess.run(
                [
                    "python3", str(RUNNER), "generate",
                    "--profile", "gemma-qat-a100", "--suite", "mmlu-pro",
                    "--run-dir", run_dir, "--pilot",
                ],
                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("requires a numeric Slurm allocation", result.stderr)


if __name__ == "__main__":
    unittest.main()
