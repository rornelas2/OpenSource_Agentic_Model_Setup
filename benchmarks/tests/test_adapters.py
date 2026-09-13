from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from benchmarks.adapters.livecodebench import extract_code
from benchmarks.adapters.mmlu_pro import parse_answer, score
from benchmarks.adapters.swe_verified import evaluate_patch, prediction
from benchmarks.adapters.terminal_bench import normalize_reward
from benchmarks.contracts import ContractError


class AdapterBoundaryTests(unittest.TestCase):
    def test_swe_keeps_empty_unsolved_patch(self) -> None:
        self.assertEqual(prediction("repo__repo-1", "", "profile")["model_patch"], "")

    def test_swe_prediction_serialization_rejects_invalid_types(self) -> None:
        with self.assertRaises(ContractError):
            prediction("", "diff --git ...", "profile")
        with self.assertRaises(ContractError):
            prediction("inst_1", 123, "profile")  # type: ignore
        with self.assertRaises(ContractError):
            prediction("inst_1", "diff", "")

    def test_swe_normalize_report_validates_resolved(self) -> None:
        from benchmarks.adapters.swe_verified import normalize_report
        self.assertTrue(normalize_report("inst_1", {"inst_1": {"resolved": True}}))
        self.assertFalse(normalize_report("inst_1", {"inst_1": {"resolved": False}}))
        with self.assertRaises(ContractError):
            normalize_report("inst_1", {"inst_2": {"resolved": True}})
        with self.assertRaises(ContractError):
            normalize_report("inst_1", {"inst_1": {"resolved": "yes"}})

    def test_swe_normalize_report_supports_official_schema_v2(self) -> None:
        from benchmarks.adapters.swe_verified import normalize_report

        report = {
            "schema_version": 2,
            "submitted_ids": ["inst_1"],
            "completed_ids": ["inst_1"],
            "resolved_ids": ["inst_1"],
            "unresolved_ids": [],
            "empty_patch_ids": [],
            "error_ids": [],
            "infra_failure_ids": [],
            "ambiguous_failure_ids": [],
        }
        self.assertTrue(normalize_report("inst_1", report))
        report["unresolved_ids"] = ["inst_1"]
        with self.assertRaisesRegex(ContractError, "ambiguous"):
            normalize_report("inst_1", report)

    def test_swe_evaluator_boundary_uses_exact_report_and_limits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset.jsonl"
            dataset.write_text("{}\n", encoding="utf-8")
            fake_python = root / "fake-python"
            fake_python.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                "args=sys.argv[1:]\n"
                "run_id=args[args.index('--run-id')+1]\n"
                "iid=args[args.index('--instance-id')+1]\n"
                "out=pathlib.Path('logs/evaluation')/run_id/'results.json'\n"
                "out.parent.mkdir(parents=True)\n"
                "fields={k:[] for k in ('unresolved_ids','empty_patch_ids','error_ids','infra_failure_ids','ambiguous_failure_ids')}\n"
                "fields.update({'schema_version':2,'submitted_ids':[iid],'completed_ids':[iid],'resolved_ids':[iid]})\n"
                "out.write_text(json.dumps(fields))\n",
                encoding="utf-8",
            )
            os.chmod(fake_python, 0o700)
            passed, reason, native = evaluate_patch(
                instance_id="inst_1",
                patch="diff --git a/a b/a\n",
                evaluator_dataset_path=dataset,
                evaluator_env_bin=fake_python,
                run_dir=root / "eval",
                timeout=17,
                cpus=3,
                memory_gib=5,
                pids_limit=19,
            )
            self.assertTrue(passed)
            self.assertEqual(reason, "official_resolved")
            self.assertEqual(native["resource_limits"]["cpus"], 3)
            self.assertIn("--predictions-path", native["command"])

    def test_terminal_reward_range(self) -> None:
        self.assertEqual(normalize_reward({"reward": 0}), 0.0)
        self.assertEqual(normalize_reward({"reward": 1.0}), 1.0)
        with self.assertRaises(ContractError):
            normalize_reward({"reward": 2})

    def test_terminal_reward_rejects_invalid_types_or_missing_keys(self) -> None:
        with self.assertRaises(ContractError):
            normalize_reward("invalid")  # type: ignore
        with self.assertRaises(ContractError):
            normalize_reward({})
        with self.assertRaises(ContractError):
            normalize_reward({"reward": -0.5})

    def test_livecodebench_extracts_last_code_fence(self) -> None:
        response = "draft\n```python\nprint('first')\n```\nfinal\n```python\nprint('second')\n```"
        self.assertEqual(extract_code(response), "print('second')")

    def test_livecodebench_extracts_code_without_fences(self) -> None:
        raw_code = "def solution(x):\n    return x + 1\n"
        self.assertEqual(extract_code(raw_code), raw_code.strip())

    def test_mmlu_parser_requires_declared_final_line(self) -> None:
        self.assertEqual(parse_answer("Reasoning.\nAnswer: C"), "C")
        self.assertIsNone(parse_answer("I think C is likely"))
        self.assertEqual(score("Reasoning.\nAnswer: C", "C"), (True, "C"))
        self.assertEqual(score("Answer: A or B", "A"), (False, None))


if __name__ == "__main__":
    unittest.main()
