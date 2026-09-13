"""Failure-boundary tests for preparation, sandbox config, and journals."""

from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.contracts import ContractError
from benchmarks.execution import _has_task_time, _slurm_end_epoch
from benchmarks.opencode_runner import build_benchmark_config, validate_isolated_network_config
from benchmarks.preparation import install_package_locked, validate_swe_dataset
from benchmarks.state import append_json_record, read_json_records


class ExecutionBoundaryTests(unittest.TestCase):
    def test_slurm_deadline_prevents_task_from_straddling_allocation(self) -> None:
        with patch.dict("os.environ", {"SLURM_JOB_END_TIME": "2000"}, clear=False):
            self.assertEqual(_slurm_end_epoch(), 2000.0)
        with patch("benchmarks.execution.time.time", return_value=1000.0):
            self.assertTrue(_has_task_time(2000.0, 600))
            self.assertFalse(_has_task_time(1800.0, 600))

    def test_benchmark_config_is_derived_and_denies_delegation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "provider.json"
            original = {
                "model": "pinnacles/model-a",
                "small_model": "pinnacles/model-a",
                "provider": {"pinnacles": {"options": {"baseURL": "http://127.0.0.1:8000/v1"},
                                              "models": {"model-a": {"limit": {"context": 1, "output": 1}}}}},
            }
            source.write_text(json.dumps(original), encoding="utf-8")
            derived = build_benchmark_config(
                source, endpoint_base_url="http://benchmark-relay:18000/v1",
                model_alias="model-a", context_tokens=16384, output_tokens=4096,
            )
            self.assertEqual(derived["permission"]["task"], "deny")
            self.assertEqual(derived["provider"]["pinnacles"]["options"]["baseURL"], "http://benchmark-relay:18000/v1")
            self.assertEqual(derived["provider"]["pinnacles"]["models"]["model-a"]["limit"]["output"], 4096)
            self.assertEqual(json.loads(source.read_text()), original)

    def test_container_network_must_be_dedicated_and_internal(self) -> None:
        with patch("benchmarks.opencode_runner._run", return_value=json.dumps([
            {"Name": "benchmark-net", "Internal": True, "Scope": "local"}
        ])):
            validate_isolated_network_config(
                "/usr/bin/docker", "benchmark-net", "http://model-relay:18000/v1"
            )
        with self.assertRaisesRegex(ContractError, "dedicated internal"):
            validate_isolated_network_config(
                "/usr/bin/docker", "host", "http://model-relay:18000/v1"
            )

    def test_unlocked_evaluator_is_rejected_before_install(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "venv/bin").mkdir(parents=True)
            (root / "venv/bin/python").touch()
            (root / "src").mkdir()
            with self.assertRaisesRegex(ContractError, "no uv.lock"):
                install_package_locked(root / "venv", root / "src")

    def test_pinned_swe_schema_incompatibility_is_explicit(self) -> None:
        fake_dataset = [{
            "instance_id": "owner__repo-1", "repo": "owner/repo", "base_commit": "a" * 40,
            "problem_statement": "fix it", "patch": "", "test_patch": "",
            "FAIL_TO_PASS": [], "PASS_TO_PASS": [], "environment_setup_commit": "b" * 40,
        }]
        fake_module = types.SimpleNamespace(load_dataset=lambda *args, **kwargs: fake_dataset)
        with patch.dict("sys.modules", {"datasets": fake_module}):
            with self.assertRaisesRegex(ContractError, "incompatible with the pinned evaluator"):
                validate_swe_dataset("test", "c" * 40, [])

    def test_generic_journal_duplicate_and_partial_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generations.jsonl"
            append_json_record(path, {"attempt_id": "a", "task_id": "one"}, unique_field="attempt_id")
            with self.assertRaisesRegex(ContractError, "already exists"):
                append_json_record(path, {"attempt_id": "a", "task_id": "two"}, unique_field="attempt_id")
            with path.open("ab") as output:
                output.write(b'{"attempt_id":"b"')
            records, partial = read_json_records(path, unique_field="attempt_id")
            self.assertTrue(partial)
            self.assertEqual([record["attempt_id"] for record in records], ["a"])

    @patch("benchmarks.execution.run_preflight")
    @patch("benchmarks.preparation.revalidate_prepared_suite")
    @patch("benchmarks.execution.input_hashes", return_value="mock-input-hash")
    @patch("benchmarks.execution._repository_state", return_value={"commit": "0" * 40, "dirty": False})
    @patch("benchmarks.execution._read_cmdline", return_value=["python3", "server.py"])
    def test_livecodebench_generation_and_evaluation(
        self, mock_cmd, mock_repo, mock_hashes, mock_reval, mock_preflight,
    ) -> None:
        from benchmarks.contracts import sha256_file
        from benchmarks.execution import execute_evaluate, execute_generation
        from benchmarks.transport import ChatCompletionResult, TransportUsage

        mock_preflight.return_value = {
            "gpu": {"name": "NVIDIA A100", "uuid": "GPU-0000", "memory_mib": 40960},
            "profile_id": "mock-profile",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            run_dir = base / "run"
            checkout = base / "checkout"
            checkout.mkdir()
            py_bin = base / "python"
            py_bin.write_text(
                '#!/usr/bin/env python3\nprint(\'{"passed":true,"results":[true],"metadata":{}}\')\n'
            )
            py_bin.chmod(0o700)
            lock = base / "uv.lock"
            lock.write_text("lock", encoding="utf-8")

            public_inputs = base / "public-inputs.jsonl"
            task_pub = {
                "task_id": "task-lcb-1",
                "question_id": 1,
                "question_title": "Echo Test",
                "question_content": "Read from stdin and print to stdout.",
                "starter_code": "",
                "public_test_cases": [],
                "input_sha256": "in-hash-lcb",
            }
            public_inputs.write_text(json.dumps(task_pub) + "\n", encoding="utf-8")

            private_grader = base / "private-grader.jsonl"
            task_priv = {
                "task_id": "task-lcb-1",
                "question_id": 1,
                "private_test_cases": [{"input": "hello\n", "output": "hello"}],
                "grader_sha256": "gr-hash-lcb",
            }
            private_grader.write_text(json.dumps(task_priv) + "\n", encoding="utf-8")

            suite_source = {
                "evaluator_repository": "LiveCodeBench/LiveCodeBench",
                "evaluator_revision": "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24",
                "dataset_repository": "livecodebench/code_generation_lite",
                "dataset_revision": "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
                "split": "test",
            }
            prep_manifest_path = base / "prep.json"
            prep_data = {
                "schema_version": "1.0",
                "operation": "prepare",
                "suites": {
                    "livecodebench": {
                        "status": "ready",
                        "evaluator": {
                            "revision": suite_source["evaluator_revision"],
                            "checkout_path": str(checkout),
                            "python_bin": str(py_bin),
                            "lock": {"path": str(lock), "sha256": sha256_file(lock)},
                        },
                        "dataset": {
                            "revision": suite_source["dataset_revision"],
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
                "id": "livecodebench",
                "protocol_status": "candidate",
                "release": "release_v6",
                "source": suite_source,
                "task_manifest": {"path": "tasks.jsonl", "sha256": "0" * 64, "expected_count": 1},
                "pilot_task_ids": ["task-lcb-1"],
                "protocol": {"path": "protocol.json", "sha256": "0" * 64},
                "execution": {
                    "adapter": "benchmarks/adapters/livecodebench.py",
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
            tasks = [{"task_id": "task-lcb-1", "input_sha256": "in-hash-lcb", "grader_sha256": "gr-hash-lcb"}]
            args = types.SimpleNamespace(pilot=True, resume=False)

            mock_solution = "```python\nimport sys\nprint(sys.stdin.read().strip())\n```"
            mock_chat_result = ChatCompletionResult(
                content=mock_solution,
                reasoning="",
                tool_calls=[],
                finish_reason="stop",
                usage=TransportUsage(prompt_tokens=40, completion_tokens=20, total_tokens=60),
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
                        generations_path = run_dir / "generations.jsonl"
                        records, partial = read_json_records(generations_path, unique_field="attempt_id")
                        self.assertFalse(partial)
                        self.assertEqual(len(records), 1)
                        self.assertEqual(records[0]["status"], "generated")
                        self.assertTrue(Path(records[0]["solution_path"]).is_file())

                        eval_summary = execute_evaluate(
                            args=args, profile=profile, suite=suite, tasks=tasks, run_dir=run_dir,
                        )
                        self.assertEqual(eval_summary["evaluated_this_invocation"], 1)
                        self.assertEqual(eval_summary["counts"]["pass"], 1)
                        trials, trial_partial = read_json_records(run_dir / "trials.jsonl", unique_field="attempt_id")
                        self.assertFalse(trial_partial)
                        self.assertEqual(trials[0]["status"], "pass")
                        self.assertEqual(trials[0]["metrics"]["reward"], 1.0)

    @patch("benchmarks.execution.run_preflight")
    @patch("benchmarks.preparation.revalidate_prepared_suite")
    @patch("benchmarks.execution.input_hashes", return_value="mock-input-hash")
    @patch("benchmarks.execution._repository_state", return_value={"commit": "0" * 40, "dirty": False})
    @patch("benchmarks.execution._read_cmdline", return_value=["python3", "server.py"])
    def test_mmlu_pro_generation_and_evaluation(
        self, mock_cmd, mock_repo, mock_hashes, mock_reval, mock_preflight,
    ) -> None:
        from benchmarks.contracts import sha256_file
        from benchmarks.execution import execute_evaluate, execute_generation
        from benchmarks.transport import ChatCompletionResult, TransportUsage

        mock_preflight.return_value = {
            "gpu": {"name": "NVIDIA A100", "uuid": "GPU-0000", "memory_mib": 40960},
            "profile_id": "mock-profile",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            run_dir = base / "run"
            checkout = base / "checkout"
            checkout.mkdir()
            py_bin = base / "python"
            py_bin.touch()
            lock = base / "uv.lock"
            lock.write_text("lock", encoding="utf-8")

            public_inputs = base / "public-inputs.jsonl"
            task_pub = {
                "task_id": "task-mmlu-1",
                "question_id": 100,
                "question": "What is 2 + 2?",
                "options": ["1", "2", "3", "4"],
                "category": "math",
                "src": "synthetic",
                "input_sha256": "in-hash-mmlu",
            }
            public_inputs.write_text(json.dumps(task_pub) + "\n", encoding="utf-8")

            private_grader = base / "private-grader.jsonl"
            task_priv = {
                "task_id": "task-mmlu-1",
                "question_id": 100,
                "answer": "D",
                "answer_index": 3,
                "grader_sha256": "gr-hash-mmlu",
            }
            private_grader.write_text(json.dumps(task_priv) + "\n", encoding="utf-8")

            validation_shots = base / "validation-shots.json"
            validation_shots.write_text(json.dumps({"math": []}), encoding="utf-8")

            suite_source = {
                "evaluator_repository": "TIGER-AI-Lab/MMLU-Pro",
                "evaluator_revision": "f418b116db00b065c2aea046518d8fcf74d39872",
                "dataset_repository": "TIGER-Lab/MMLU-Pro",
                "dataset_revision": "b189ec765aa7ed75c8acfea42df31fdae71f97be",
                "split": "test",
            }
            prep_manifest_path = base / "prep.json"
            prep_data = {
                "schema_version": "1.0",
                "operation": "prepare",
                "suites": {
                    "mmlu-pro": {
                        "status": "ready",
                        "evaluator": {
                            "revision": suite_source["evaluator_revision"],
                            "checkout_path": str(checkout),
                            "python_bin": str(py_bin),
                            "lock": {"path": str(lock), "sha256": sha256_file(lock)},
                        },
                        "dataset": {
                            "revision": suite_source["dataset_revision"],
                            "validation": {
                                "public_inputs_path": str(public_inputs),
                                "public_inputs_sha256": sha256_file(public_inputs),
                                "private_grader_path": str(private_grader),
                                "private_grader_sha256": sha256_file(private_grader),
                                "validation_shots_path": str(validation_shots),
                                "validation_shots_sha256": sha256_file(validation_shots),
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
                "id": "mmlu-pro",
                "protocol_status": "candidate",
                "release": "MMLU-Pro-pinned-test-five-shot",
                "source": suite_source,
                "task_manifest": {"path": "tasks.jsonl", "sha256": "0" * 64, "expected_count": 1},
                "pilot_task_ids": ["task-mmlu-1"],
                "protocol": {"path": "protocol.json", "sha256": "0" * 64},
                "execution": {
                    "adapter": "benchmarks/adapters/mmlu_pro.py",
                    "requires_code_execution": False,
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
                "scoring": {"metric": "micro_accuracy", "unit": "fraction", "denominator": "scheduled"},
            }
            tasks = [{"task_id": "task-mmlu-1", "input_sha256": "in-hash-mmlu", "grader_sha256": "gr-hash-mmlu"}]
            args = types.SimpleNamespace(pilot=True, resume=False)

            mock_chat_result = ChatCompletionResult(
                content="Let's compute: 2 + 2 = 4, which is option D.\nAnswer: D",
                reasoning="",
                tool_calls=[],
                finish_reason="stop",
                usage=TransportUsage(prompt_tokens=50, completion_tokens=15, total_tokens=65),
                http_status=200,
                request_sent_time=0.0,
                end_to_end_latency=0.15,
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
                        generations_path = run_dir / "generations.jsonl"
                        records, partial = read_json_records(generations_path, unique_field="attempt_id")
                        self.assertFalse(partial)
                        self.assertEqual(records[0]["status"], "generated")
                        self.assertEqual(records[0]["parsed_answer"], "D")

                        eval_summary = execute_evaluate(
                            args=args, profile=profile, suite=suite, tasks=tasks, run_dir=run_dir,
                        )
                        self.assertEqual(eval_summary["evaluated_this_invocation"], 1)
                        self.assertEqual(eval_summary["counts"]["pass"], 1)
                        trials, trial_partial = read_json_records(run_dir / "trials.jsonl", unique_field="attempt_id")
                        self.assertFalse(trial_partial)
                        self.assertEqual(trials[0]["status"], "pass")
                        self.assertEqual(trials[0]["metrics"]["reward"], 1.0)

    @patch("benchmarks.preparation.revalidate_prepared_suite")
    def test_terminal_bench_evaluation(self, mock_reval) -> None:
        from benchmarks.contracts import sha256_file
        from benchmarks.execution import execute_evaluate

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            run_dir = base / "run"
            run_dir.mkdir(parents=True, mode=0o700)
            task_artifact_dir = run_dir / "tasks" / "cancel-async-tasks" / "attempt-1"
            task_artifact_dir.mkdir(parents=True, mode=0o700)

            verifier_script = base / "test.sh"
            verifier_script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            verifier_script.chmod(0o755)

            private_grader = base / "private-grader.jsonl"
            task_priv = {
                "task_id": "cancel-async-tasks",
                "verifier_path": str(verifier_script),
                "grader_sha256": "gr-hash-tb",
            }
            private_grader.write_text(json.dumps(task_priv) + "\n", encoding="utf-8")

            suite_source = {
                "evaluator_repository": "harbor-framework/harbor",
                "evaluator_revision": "4797595c3858481285d0ea138580e35111cea0a1",
                "dataset_repository": "harbor-framework/terminal-bench-2",
                "dataset_revision": "2fd12b88aafdd04a52c298e3940bcb189f9766d6",
                "split": "all",
            }
            prep_manifest_path = base / "prep.json"
            (base / "checkout").mkdir()
            (base / "python").touch()
            lock_file = base / "lock"
            lock_file.write_text("lock", encoding="utf-8")

            prep_data = {
                "schema_version": "1.0",
                "operation": "prepare",
                "suites": {
                    "terminal-bench": {
                        "status": "ready",
                        "evaluator": {
                            "revision": suite_source["evaluator_revision"],
                            "checkout_path": str(base / "checkout"),
                            "python_bin": str(base / "python"),
                            "lock": {"path": str(lock_file), "sha256": sha256_file(lock_file)},
                        },
                        "dataset": {
                            "revision": suite_source["dataset_revision"],
                            "validation": {
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
            }
            suite = {
                "id": "terminal-bench",
                "protocol_status": "candidate",
                "release": "Terminal-Bench 2.0",
                "source": suite_source,
                "execution": {
                    "adapter": "benchmarks/adapters/terminal_bench.py",
                    "requires_code_execution": True,
                    "requires_container": False,
                    "primary_concurrency": 1,
                },
                "budget": {"attempts": 1, "request_timeout_seconds": 60},
            }
            tasks = [{"task_id": "cancel-async-tasks", "input_sha256": "in-hash-tb", "grader_sha256": "gr-hash-tb"}]

            run_manifest = {
                "schema_version": "1.0",
                "run_id": "tb-run-1",
                "profile_id": "mock-profile",
                "suite_id": "terminal-bench",
                "scheduled_task_ids": ["cancel-async-tasks"],
                "preparation_manifest_sha256": sha256_file(prep_manifest_path),
            }
            (run_dir / "manifest.json").write_text(json.dumps(run_manifest), encoding="utf-8")

            gen_record = {
                "schema_version": "1.0",
                "run_id": "tb-run-1",
                "task_id": "cancel-async-tasks",
                "attempt_id": "tb-run-1:cancel-async-tasks:1",
                "input_sha256": "in-hash-tb",
                "status": "generated",
                "artifact_dir": str(task_artifact_dir),
                "elapsed_seconds": 5.0,
                "request_count": 2,
                "tool_event_count": 3,
            }
            (run_dir / "generations.jsonl").write_text(json.dumps(gen_record) + "\n", encoding="utf-8")

            env = {
                "BENCHMARK_ALLOW_LOCAL": "1",
                "BENCHMARK_PREPARATION_MANIFEST": str(prep_manifest_path),
            }
            with patch.dict("os.environ", env, clear=False):
                eval_summary = execute_evaluate(
                    args=types.SimpleNamespace(pilot=True, resume=False),
                    profile=profile, suite=suite, tasks=tasks, run_dir=run_dir,
                )
                self.assertEqual(eval_summary["evaluated_this_invocation"], 1)
                self.assertEqual(eval_summary["counts"]["pass"], 1)
                trials, trial_partial = read_json_records(run_dir / "trials.jsonl", unique_field="attempt_id")
                self.assertFalse(trial_partial)
                self.assertEqual(trials[0]["status"], "pass")
                self.assertEqual(trials[0]["metrics"]["reward"], 1.0)


if __name__ == "__main__":
    unittest.main()
