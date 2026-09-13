"""LiveCodeBench deterministic response extraction and evaluation boundary.

Extracts code from model responses, runs official test cases in isolated
processes, and normalizes pass@1 results into the benchmark trial schema.
"""

from __future__ import annotations

import datetime
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from benchmarks import SCHEMA_VERSION
from benchmarks.contracts import ContractError
from benchmarks.transport import ChatCompletionResult, ModelTransportClient


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


_PYTHON_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(content: str) -> str:
    if not isinstance(content, str):
        raise ContractError("LiveCodeBench response content must be a string")
    matches = _PYTHON_FENCE.findall(content)
    extracted = matches[-1] if matches else content
    return extracted.strip()


def format_prompt(problem: dict[str, Any]) -> list[dict[str, Any]]:
    """Format LiveCodeBench problem prompt matching official code generation format."""
    title = problem.get("question_title", "")
    content = problem.get("question_content", "")
    starter = problem.get("starter_code", "")
    user_prompt = f"Problem Title: {title}\n\n{content}\n"
    if starter:
        user_prompt += f"\nStarter Code:\n```python\n{starter}\n```\n"
    user_prompt += "\nPlease write a complete, self-contained Python 3 solution inside a ```python ``` block."

    return [
        {
            "role": "system",
            "content": "You are an expert competitive programmer. Provide only the Python 3 solution in a markdown code block.",
        },
        {"role": "user", "content": user_prompt},
    ]


def unpack_test_cases(raw_tc: Any) -> list[dict[str, Any]]:
    if isinstance(raw_tc, list):
        return raw_tc
    if isinstance(raw_tc, str):
        try:
            return json.loads(raw_tc)
        except Exception:
            try:
                import base64, pickle, zlib
                return json.loads(pickle.loads(zlib.decompress(base64.b64decode(raw_tc.encode("utf-8")))))
            except Exception:
                return []
    return []


def evaluate_python_code(
    code: str,
    test_cases: list[dict[str, Any]] | str,
    metadata: dict[str, Any] | str | None = None,
    evaluator_python: Path | None = None,
    evaluator_src_dir: Path | None = None,
    timeout_per_case: int = 5,
) -> tuple[bool, str]:
    """Execute extracted python code against test cases in disposable subprocess or official evaluator."""
    if not code.strip():
        return False, "empty_code"

    cases = unpack_test_cases(test_cases)
    if not cases:
        return False, "no_test_cases"

    if isinstance(metadata, str):
        try:
            meta_dict = json.loads(metadata)
        except Exception:
            meta_dict = {}
    elif isinstance(metadata, dict):
        meta_dict = metadata
    else:
        meta_dict = {}

    fn_name = meta_dict.get("func_name")

    if evaluator_python is not None or evaluator_src_dir is not None:
        if not (evaluator_python and Path(evaluator_python).is_file()
                and evaluator_src_dir and Path(evaluator_src_dir).is_dir()):
            return False, "evaluator_infra_missing_runtime"
        eval_script = """
import sys, json
from lcb_runner.evaluation.compute_code_generation_metrics import check_correctness
payload = json.loads(sys.stdin.read())
sample = payload['sample']
code = payload['code']
timeout = payload.get('timeout', 5)
res, meta = check_correctness(sample, code, timeout=timeout, debug=False)
passed = bool(res and all(x is True for x in res))
print(json.dumps({'passed': passed, 'results': res, 'metadata': meta}))
"""
        sample = {
            "input_output": json.dumps({
                "inputs": [tc.get("input", "") for tc in cases],
                "outputs": [tc.get("output", "") for tc in cases],
                "fn_name": fn_name,
            })
        }
        payload = {"sample": sample, "code": code, "timeout": timeout_per_case}
        try:
            proc = subprocess.run(
                [str(evaluator_python), "-c", eval_script],
                cwd=str(evaluator_src_dir),
                input=json.dumps(payload),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=(timeout_per_case + 1) * len(cases) + 15,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                eval_out = json.loads(proc.stdout.strip())
                passed = eval_out.get("passed")
                if not isinstance(passed, bool):
                    return False, "evaluator_infra_invalid_report"
                res_list = eval_out.get("results", [])
                meta = eval_out.get("metadata", {})
                if passed:
                    return True, "all_passed"
                err_msg = meta.get("error_message") or (f"failed_cases_{res_list[:3]}" if res_list else "wrong_answer")
                return False, str(err_msg)
            return False, "evaluator_infra_process_failed"
        except subprocess.TimeoutExpired:
            return False, "evaluator_timeout"
        except (OSError, ValueError, TypeError, AttributeError):
            return False, "evaluator_infra_invalid_report"

    if fn_name:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "solution.py"
            escaped_code = code.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
            harness = f'''
import sys, json

{code}

cases = {json.dumps(cases)}
sol = Solution() if 'Solution' in globals() else None
fn = getattr(sol, '{fn_name}', globals().get('{fn_name}'))
if fn is None:
    sys.exit(101)

for idx, tc in enumerate(cases):
    tc_in = tc.get('input', '')
    expected = tc.get('output', '')
    try:
        args = json.loads(tc_in) if tc_in.startswith('[') or tc_in.startswith('{{') or tc_in.startswith('"') else [tc_in]
        if isinstance(args, list):
            out = fn(*args)
        else:
            out = fn(args)
    except Exception:
        sys.exit(102)
    exp_parsed = json.loads(expected) if (isinstance(expected, str) and (expected.startswith('[') or expected.startswith('{{') or expected.startswith('"'))) else expected
    if out != exp_parsed and str(out) != str(expected).strip():
        sys.exit(103)
sys.exit(0)
'''
            script_path.write_text(harness, encoding="utf-8")
            try:
                proc = subprocess.run(
                    [sys.executable, str(script_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout_per_case * len(cases) + 5,
                )
                if proc.returncode == 0:
                    return True, "all_passed"
                elif proc.returncode == 101:
                    return False, "missing_function"
                elif proc.returncode == 102:
                    return False, "runtime_error_functional"
                elif proc.returncode == 103:
                    return False, "wrong_answer_functional"
                else:
                    return False, f"error_functional_{proc.returncode}"
            except subprocess.TimeoutExpired:
                return False, "timeout_functional"
            except Exception as exc:
                return False, f"exec_error_functional: {exc}"

    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "solution.py"
        script_path.write_text(code, encoding="utf-8")

        for idx, tc in enumerate(cases):
            tc_input = tc.get("input", "")
            expected_output = tc.get("output", "").strip()

            try:
                proc = subprocess.run(
                    [sys.executable, str(script_path)],
                    input=tc_input,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout_per_case,
                )
            except subprocess.TimeoutExpired:
                return False, f"timeout_case_{idx}"
            except Exception as exc:
                return False, f"exec_error_case_{idx}: {exc}"

            if proc.returncode != 0:
                return False, f"runtime_error_case_{idx}: {proc.stderr[:200]}"

            actual_output = proc.stdout.strip()
            if actual_output != expected_output:
                return False, f"wrong_answer_case_{idx}"

    return True, "all_passed"


def generate_and_evaluate(
    *,
    client: ModelTransportClient,
    problem: dict[str, Any],
    test_cases: list[dict[str, Any]],
    max_output_tokens: int = 8192,
) -> tuple[str, bool, str, ChatCompletionResult]:
    """Generate solution via endpoint and evaluate against test cases."""
    messages = format_prompt(problem)
    result = client.chat_completion(
        messages=messages,
        stream=True,
        max_tokens=max_output_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    code = extract_code(result.content)
    passed, reason = evaluate_python_code(code, test_cases)
    return code, passed, reason, result


def build_trial_record(
    *,
    run_id: str,
    task_id: str,
    attempt_id: str,
    status: str,
    reason: str,
    started_utc: str,
    ended_utc: str,
    prompt_tokens: int | None = None,
    generated_tokens: int | None = None,
    token_source: str | None = "server_usage",
    finish_reason: str | None = "stop",
    http_status: int | None = 200,
    metrics: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    error_class: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "task_id": task_id,
        "attempt_id": attempt_id,
        "status": status,
        "reason": reason,
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "http_status": http_status,
        "finish_reason": finish_reason,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "token_source": token_source,
        "metrics": metrics or {},
        "artifacts": artifacts or {},
        "error_class": error_class,
    }
