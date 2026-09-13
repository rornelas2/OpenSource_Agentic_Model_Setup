from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.contracts import ContractError, load_all_suites, load_profiles, load_suite, load_task_records
from benchmarks.state import TERMINAL_STATUSES, TRIAL_FIELDS


ROOT = Path(__file__).resolve().parents[2]


class CatalogContractTests(unittest.TestCase):
    def test_repository_catalogs_validate(self) -> None:
        profiles = load_profiles(ROOT / "benchmarks/profiles.json", ROOT)
        suites = load_all_suites(ROOT / "benchmarks/suites", ROOT)
        self.assertEqual(
            set(profiles),
            {
                "gemma-bf16-h200", "muse-bf16-h200", "super-nvfp4-h200",
                "muse-dynamic-a100", "gemma-qat-a100", "lightning-nvfp4-a100",
                "gemma-bf16-a100", "muse-bf16-a100",
            },
        )
        self.assertEqual({name: len(value[1]) for name, value in suites.items()}, {
            "swe-verified": 500,
            "terminal-bench": 89,
            "livecodebench": 1055,
            "mmlu-pro": 12032,
            "aider-refactor": 89,
        })

    def test_unknown_profile_field_is_rejected(self) -> None:
        source = json.loads((ROOT / "benchmarks/profiles.json").read_text())
        source["profiles"][0]["surprise"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profiles.json"
            path.write_text(json.dumps(source))
            with self.assertRaisesRegex(ContractError, "unknown fields: surprise"):
                load_profiles(path, ROOT)

    def test_duplicate_task_id_is_rejected(self) -> None:
        line = json.dumps({
            "task_id": "duplicate",
            "input_sha256": "a" * 64,
            "grader_sha256": "b" * 64,
            "stratum": "fixture",
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.jsonl"
            path.write_text(f"{line}\n{line}\n")
            with self.assertRaisesRegex(ContractError, "duplicate task id"):
                load_task_records(path)

    def test_trial_json_schema_matches_runtime_contract(self) -> None:
        schema = json.loads((ROOT / "benchmarks/schemas/trial.schema.json").read_text())
        self.assertEqual(set(schema["required"]), TRIAL_FIELDS)
        self.assertEqual(set(schema["properties"]), TRIAL_FIELDS)
        self.assertEqual(set(schema["properties"]["status"]["enum"]), TERMINAL_STATUSES)

    def test_suite_sources_and_protocols_are_immutable_candidates(self) -> None:
        suites = load_all_suites(ROOT / "benchmarks/suites", ROOT)
        for suite, _ in suites.values():
            self.assertEqual(suite["protocol_status"], "candidate")
            for key, value in suite["source"].items():
                if key.endswith("revision"):
                    self.assertRegex(value, r"^[0-9a-f]{40}$")

    def test_stale_task_manifest_hash_is_rejected(self) -> None:
        swe_suite_path = ROOT / "benchmarks/suites/swe-verified/suite.json"
        suite_data = json.loads(swe_suite_path.read_text())
        suite_data["task_manifest"]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            tampered_path = Path(directory) / "swe-verified.json"
            tampered_path.write_text(json.dumps(suite_data))
            with self.assertRaisesRegex(ContractError, "SHA-256 does not match suite manifest"):
                load_suite(tampered_path, ROOT)


if __name__ == "__main__":
    unittest.main()
