import json
import os
import shutil
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.adapters.aider_refactor import (
    extract_code,
    format_prompt,
    verify_refactor_ast,
)
from benchmarks.contracts import ContractError, sha256_file
from benchmarks.execution import execute_evaluate, execute_generation
from benchmarks.state import read_json_records
from benchmarks.transport import ChatCompletionResult, TransportUsage


class AiderRefactorTests(unittest.TestCase):
    def test_extract_code(self) -> None:
        raw = "Here is the refactored code:\n```python\ndef foo():\n    return 42\n```\nDone."
        self.assertEqual(extract_code(raw), "def foo():\n    return 42")

        bare = "def foo():\n    return 42"
        self.assertEqual(extract_code(bare), "def foo():\n    return 42")

        with self.assertRaises(ContractError):
            extract_code(123)  # type: ignore

    def test_format_prompt(self) -> None:
        task = {
            "instructions": "Refactor bar into a top level function.",
            "source_code": "class Foo:\n    def bar(self):\n        pass",
            "src_file": "foo.py",
            "method": "bar",
            "class_name": "Foo",
        }
        messages = format_prompt(task)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Refactor bar", messages[1]["content"])
        self.assertIn("```python\nclass Foo:", messages[1]["content"])

    def test_verify_refactor_ast_success_and_failures(self) -> None:
        unrefactored = "class Calc:\n    def compute(self, x):\n        return x + 1\n"
        passed, reason = verify_refactor_ast(unrefactored, "compute", 9, "Calc", 11)
        self.assertFalse(passed)
        self.assertIn("not a top level function", reason)

        refactored = "def compute(x):\n    return x + 1\n\nclass Calc:\n    pass\n"
        passed, reason = verify_refactor_ast(refactored, "compute", 9, "Calc", 11)
        self.assertTrue(passed)
        self.assertEqual(reason, "all_passed")

        syntax_err = "def compute(x):\nreturn x +"
        passed, reason = verify_refactor_ast(syntax_err, "compute", 9, "Calc", 11)
        self.assertFalse(passed)
        self.assertTrue(reason.startswith("syntax_error"))

        passed, reason = verify_refactor_ast("", "compute", 9, "Calc", 11)
        self.assertFalse(passed)
        self.assertEqual(reason, "empty_code")

    @patch("benchmarks.execution.run_preflight")
    @patch("benchmarks.preparation.revalidate_prepared_suite")
    @patch("benchmarks.execution.input_hashes", return_value="mock-input-hash")
    @patch("benchmarks.execution._repository_state", return_value={"commit": "0" * 40, "dirty": False})
    @patch("benchmarks.execution._read_cmdline", return_value=["python3", "server.py"])
    def test_aider_execution_boundary(
        self, mock_cmd, mock_repo, mock_hashes, mock_reval, mock_preflight,
    ) -> None:
        mock_preflight.return_value = {
            "gpu": {"name": "NVIDIA A100", "uuid": "GPU-0000", "memory_mib": 40960},
            "profile_id": "mock-profile",
        }

        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            run_dir = base / "run"
            run_dir.mkdir(parents=True)

            checkout = base / "checkout"
            checkout.mkdir()
            py_bin = base / "python"
            py_bin.touch()
            lock = base / "uv.lock"
            lock.write_text("lock", encoding="utf-8")

            public_inputs = base / "public-inputs.jsonl"
            task_pub = {
                "task_id": "test_task",
                "src_file": "calc.py",
                "method": "add",
                "class_name": "Calc",
                "instructions": "Refactor add out of Calc.",
                "source_code": "class Calc:\n    def add(self, x):\n        return x + 1\n",
                "input_sha256": "in-hash-aider",
            }
            public_inputs.write_text(json.dumps(task_pub) + "\n", encoding="utf-8")

            private_grader = base / "private-grader.jsonl"
            task_priv = {
                "task_id": "test_task",
                "src_file": "calc.py",
                "method": "add",
                "method_children": 9,
                "class_name": "Calc",
                "class_children": 11,
                "grader_sha256": "gr-hash-aider",
            }
            private_grader.write_text(json.dumps(task_priv) + "\n", encoding="utf-8")

            prep_manifest_path = base / "prep.json"
            prep_data = {
                "schema_version": "1.0",
                "operation": "prepare",
                "suites": {
                    "aider-refactor": {
                        "status": "ready",
                        "evaluator": {
                            "revision": "c90dfb6",
                            "checkout_path": str(checkout),
                            "python_bin": str(py_bin),
                            "lock": {"path": str(lock), "sha256": sha256_file(lock)},
                        },
                        "dataset": {
                            "revision": "c90dfb6",
                            "validation": {
                                "public_inputs_path": str(public_inputs),
                                "public_inputs_sha256": sha256_file(public_inputs),
                                "private_grader_path": str(private_grader),
                                "private_grader_sha256": sha256_file(private_grader),
                            },
                        },
                    },
                },
            }
            prep_manifest_path.write_text(json.dumps(prep_data), encoding="utf-8")

            profile = {
                "id": "mock-profile",
                "model": "google/gemma",
                "runtime": "vllm",
                "endpoint": {"base_url": "http://127.0.0.1:8000/v1", "model_alias": "gemma"},
                "deployment": {"reasoning": "off", "gpu": "nvidia_a100"},
                "server_pid_env": "TEST_PID",
                "_opencode_sha256": "0" * 64,
                "_resolved_files": {"provider": base / "provider.json"},
            }
            suite = {
                "id": "aider-refactor",
                "protocol_status": "candidate",
                "release": "c90dfb6",
                "source": {
                    "evaluator_repository": "Aider-AI/refactor-benchmark",
                    "evaluator_revision": "c90dfb6",
                    "dataset_repository": "Aider-AI/refactor-benchmark",
                    "dataset_revision": "c90dfb6",
                    "split": "main",
                },
                "task_manifest": {"path": "tasks.jsonl", "sha256": "0" * 64, "expected_count": 1},
                "pilot_task_ids": ["test_task"],
                "protocol": {"path": "protocol.json", "sha256": "0" * 64},
                "execution": {
                    "adapter": "benchmarks/adapters/aider_refactor.py",
                    "requires_code_execution": True,
                    "requires_container": False,
                    "primary_concurrency": 1,
                },
                "budget": {
                    "attempts": 1,
                    "request_timeout_seconds": 60,
                    "max_model_requests": 1,
                    "max_tool_events": 1,
                    "max_generated_tokens": 4096,
                    "max_output_tokens": 4096,
                    "context_tokens": 16384,
                },
                "scoring": {"metric": "pass_at_1", "unit": "fraction", "denominator": "scheduled"},
            }
            tasks = [{"task_id": "test_task", "input_sha256": "in-hash-aider", "grader_sha256": "gr-hash-aider"}]
            args = types.SimpleNamespace(pilot=True, resume=False)

            mock_solution = "```python\ndef add(x):\n    return x + 1\n\nclass Calc:\n    pass\n```"
            mock_chat_result = ChatCompletionResult(
                content=mock_solution,
                reasoning="",
                tool_calls=[],
                finish_reason="stop",
                usage=TransportUsage(prompt_tokens=50, completion_tokens=30, total_tokens=80),
                http_status=200,
                request_sent_time=0.0,
                end_to_end_latency=0.2,
            )

            env = {
                "BENCHMARK_ALLOW_LOCAL": "1",
                "BENCHMARK_PREPARATION_MANIFEST": str(prep_manifest_path),
                "TEST_PID": "1234",
            }
            with patch.dict("os.environ", env, clear=False):
                with patch("benchmarks.execution.TelemetrySampler") as mock_sampler:
                    mock_sampler.return_value.summary.return_value = {"samples_count": 1}
                    with patch("benchmarks.transport.ModelTransportClient.chat_completion", return_value=mock_chat_result):
                        gen_summary = execute_generation(
                            args=args, profile=profile, suite=suite, tasks=tasks,
                            run_dir=run_dir, repo_root=base, profiles_path=base / "profiles.json",
                        )
                        self.assertEqual(gen_summary["generated_this_invocation"], 1)

                        eval_summary = execute_evaluate(
                            args=args, profile=profile, suite=suite, tasks=tasks, run_dir=run_dir,
                        )
                        self.assertEqual(eval_summary["evaluated_this_invocation"], 1)
                        self.assertEqual(eval_summary["counts"]["pass"], 1)
                        trials, _ = read_json_records(run_dir / "trials.jsonl", unique_field="attempt_id")
                        self.assertEqual(trials[0]["status"], "pass")
