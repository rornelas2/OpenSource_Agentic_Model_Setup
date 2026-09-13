"""MMLU-Pro strict final-answer parsing and scoring boundary.

Formats fixed five-shot chain-of-thought prompts using validation shots,
evaluates responses with strict answer parsing, and builds trial records.
"""

from __future__ import annotations

import datetime
import re
from typing import Any

from benchmarks import SCHEMA_VERSION
from benchmarks.contracts import ContractError
from benchmarks.transport import ChatCompletionResult, ModelTransportClient


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


_FINAL = re.compile(
    r"(?:^|\n|[\.\!\?]\s+)\s*(?:the\s+)?(?:final\s+)?answer\s*(?::|is)\s*[(\[]?([A-J])[)\]]?\s*[.!]?\s*$",
    re.IGNORECASE,
)


def parse_answer(content: str) -> str | None:
    if not isinstance(content, str):
        raise ContractError("MMLU-Pro response content must be a string")
    match = _FINAL.search(content)
    return match.group(1).upper() if match else None


def score(content: str, expected: str) -> tuple[bool, str | None]:
    if expected not in tuple("ABCDEFGHIJ"):
        raise ContractError("MMLU-Pro expected answer must be A..J")
    parsed = parse_answer(content)
    return parsed == expected, parsed


def format_five_shot_prompt(
    target_row: dict[str, Any],
    validation_shots: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Format prompt with 5 chain-of-thought shots from validation split."""
    category = target_row.get("category", "General")
    system_content = (
        f"The following are multiple choice questions (with answers) about {category}. "
        "Think step by step and then declare your final answer in the exact format:\n"
        "Answer: [A-J]"
    )
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_content}]

    for shot in validation_shots:
        q_text = shot["question"]
        options = shot["options"]
        opt_str = "\n".join(f"({chr(65 + i)}) {opt}" for i, opt in enumerate(options))
        shot_user = f"Question:\n{q_text}\n\nOptions:\n{opt_str}"
        shot_answer = shot.get("cot_content") or f"Answer: {shot['answer']}"
        messages.append({"role": "user", "content": shot_user})
        messages.append({"role": "assistant", "content": shot_answer})

    target_q = target_row["question"]
    target_opts = target_row["options"]
    target_opts_str = "\n".join(f"({chr(65 + i)}) {opt}" for i, opt in enumerate(target_opts))
    target_user = f"Question:\n{target_q}\n\nOptions:\n{target_opts_str}"
    messages.append({"role": "user", "content": target_user})

    return messages


def generate_and_score(
    *,
    client: ModelTransportClient,
    target_row: dict[str, Any],
    validation_shots: list[dict[str, Any]],
    max_output_tokens: int = 4096,
) -> tuple[str | None, bool, ChatCompletionResult]:
    """Send 5-shot MMLU-Pro question to model endpoint and score strictly."""
    messages = format_five_shot_prompt(target_row, validation_shots)
    result = client.chat_completion(
        messages=messages,
        stream=True,
        max_tokens=max_output_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    expected_answer = target_row["answer"]
    is_correct, parsed_choice = score(result.content, expected_answer)
    return parsed_choice, is_correct, result


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
