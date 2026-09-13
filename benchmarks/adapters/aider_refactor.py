"""Aider refactoring benchmark response extraction and AST evaluation boundary.

Extracts refactored Python source code from model responses and verifies with
pure AST inspection that the requested method was refactored into a top-level
function while preserving logic without lazy elision.
"""

from __future__ import annotations

import ast
import datetime
import re
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
        raise ContractError("Aider refactor response content must be a string")
    matches = _PYTHON_FENCE.findall(content)
    extracted = matches[-1] if matches else content
    return extracted.strip()


def format_prompt(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Format Aider refactoring task prompt with instructions and source file."""
    instructions = task.get("instructions", "")
    source_code = task.get("source_code", "")
    src_file = task.get("src_file", "source.py")
    method = task.get("method", "")
    class_name = task.get("class_name", "")

    user_prompt = f"{instructions}\n\n"
    user_prompt += f"Here is the current content of `{src_file}`:\n\n```python\n{source_code}\n```\n\n"
    user_prompt += (
        f"Please provide the complete, updated `{src_file}` with the `{method}` method refactored "
        f"out of `{class_name}` into a top-level function. Provide the entire updated file in a single "
        "```python ... ``` code block. Do NOT truncate, abbreviate, or use placeholder comments like "
        "'# ... existing code ...'."
    )

    return [
        {
            "role": "system",
            "content": (
                "You are an expert Python software engineer specializing in clean, non-destructive refactoring. "
                "You always return full, functional, valid Python code without elision or placeholders."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]


class ParentNodeTransformer(ast.NodeTransformer):
    """Sets the 'parent' attribute on each AST node."""

    def generic_visit(self, node: ast.AST) -> ast.AST:
        for child in ast.iter_child_nodes(node):
            child.parent = node  # type: ignore[attr-defined]
        return super().generic_visit(node)


def verify_full_func_at_top_level(tree: ast.AST, func: str, func_children: int) -> None:
    func_nodes = [
        item for item in ast.walk(tree) if isinstance(item, ast.FunctionDef) and item.name == func
    ]
    if not func_nodes:
        raise AssertionError(f"Function {func} not found in refactored code")

    for func_node in func_nodes:
        parent = getattr(func_node, "parent", None)
        if not isinstance(parent, ast.Module):
            continue

        num_children = sum(1 for _ in ast.walk(func_node))
        if func_children > 0:
            pct_diff = abs(num_children - func_children) * 100 / func_children
            if pct_diff >= 10:
                raise AssertionError(
                    f"Old method had {func_children} AST nodes, new top-level function has {num_children} ({pct_diff:.1f}% difference, max allowed 10%)"
                )
        return

    raise AssertionError(f"{func} is not a top level function")


def verify_old_class_children(tree: ast.AST, old_class: str, expected_children: int) -> None:
    class_node = next(
        (
            item
            for item in ast.walk(tree)
            if isinstance(item, ast.ClassDef) and item.name == old_class
        ),
        None,
    )
    if class_node is None:
        raise AssertionError(f"Old class {old_class} not found in refactored code")

    num_children = sum(1 for _ in ast.walk(class_node))
    if expected_children > 0:
        pct_diff = abs(num_children - expected_children) * 100 / expected_children
        if pct_diff >= 10:
            raise AssertionError(
                f"Expected class {old_class} to have ~{expected_children} AST nodes after refactor, but found {num_children} ({pct_diff:.1f}% difference)"
            )


def verify_refactor_ast(
    code: str,
    method: str,
    method_children: int,
    class_name: str,
    class_children: int,
) -> tuple[bool, str]:
    """Pure AST validation verifying method extraction without elision."""
    if not code.strip():
        return False, "empty_code"

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"syntax_error: {exc.msg} at line {exc.lineno}"
    except Exception as exc:
        return False, f"ast_parse_error: {exc}"

    ParentNodeTransformer().visit(tree)

    try:
        verify_full_func_at_top_level(tree, method, method_children)
    except AssertionError as exc:
        return False, f"func_verification_failed: {exc}"

    try:
        target_class_children = class_children - method_children
        verify_old_class_children(tree, class_name, target_class_children)
    except AssertionError as exc:
        return False, f"class_verification_failed: {exc}"

    return True, "all_passed"


def generate_and_evaluate(
    *,
    client: ModelTransportClient,
    task: dict[str, Any],
    grader: dict[str, Any],
    max_output_tokens: int = 8192,
) -> tuple[str, bool, str, ChatCompletionResult]:
    """Generate solution via endpoint and evaluate AST refactor correctness."""
    messages = format_prompt(task)
    result = client.chat_completion(
        messages=messages,
        stream=True,
        max_tokens=max_output_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    code = extract_code(result.content)
    passed, reason = verify_refactor_ast(
        code=code,
        method=grader.get("method", ""),
        method_children=grader.get("method_children", 0),
        class_name=grader.get("class_name", ""),
        class_children=grader.get("class_children", 0),
    )
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
