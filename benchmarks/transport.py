"""Bounded OpenAI-compatible HTTP and SSE transport client.

Supports /v1/models validation, streaming and non-streaming chat completions,
reasoning deltas, streaming tool call assembly, usage metadata, timeouts,
cancellation, response size limits, and bounded retries.
"""

from __future__ import annotations

import dataclasses
import http.client
import json
import math
import socket
import time
from typing import Any, Callable
from urllib.parse import urlparse


class TransportError(RuntimeError):
    """Base exception for transport errors."""


class AliasMismatchError(TransportError):
    """Raised when the endpoint does not expose the expected model alias."""


class ResponseTooLargeError(TransportError):
    """Raised when response exceeds maximum allowed bytes."""


class StreamTimeoutError(TransportError):
    """Raised when a connection, read, or total timeout expires."""


class StreamMalformedError(TransportError):
    """Raised when SSE or JSON format is corrupted or truncated."""


class ModelRequestError(TransportError):
    """Raised when the server returns an HTTP error status."""

    def __init__(self, status: int, message: str, body: str | None = None) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.body = body


class RequestCancelledError(TransportError):
    """Raised when a request is explicitly cancelled."""


class _TransientConnectionError(TransportError):
    """A failure before an HTTP request is sent; safe for bounded retry."""


@dataclasses.dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str
    type: str = "function"


@dataclasses.dataclass(frozen=True)
class TransportUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclasses.dataclass
class ChatCompletionChunk:
    content_delta: str | None = None
    reasoning_delta: str | None = None
    tool_calls_delta: list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    usage: TransportUsage | None = None


@dataclasses.dataclass
class ChatCompletionResult:
    content: str
    reasoning: str
    tool_calls: list[ToolCall]
    finish_reason: str | None
    usage: TransportUsage | None
    http_status: int
    request_sent_time: float
    first_byte_time: float | None = None
    first_token_time: float | None = None
    first_visible_time: float | None = None
    first_tool_time: float | None = None
    last_token_time: float | None = None
    end_to_end_latency: float = 0.0
    chunks_count: int = 0


@dataclasses.dataclass(frozen=True)
class ModelTransportConfig:
    base_url: str
    expected_model_alias: str
    connect_timeout: float = 15.0
    read_timeout: float = 120.0
    total_timeout: float = 600.0
    max_response_bytes: int = 32 * 1024 * 1024
    max_retries: int = 2
    retry_backoff_seconds: float = 0.2
    api_key: str = "EMPTY"


class ModelTransportClient:
    """Client for bounded communication with OpenAI-compatible inference endpoints."""

    def __init__(self, config: ModelTransportConfig) -> None:
        self.config = config
        if not config.expected_model_alias:
            raise ValueError("expected_model_alias must be nonempty")
        for name, value in {
            "connect_timeout": config.connect_timeout,
            "read_timeout": config.read_timeout,
            "total_timeout": config.total_timeout,
        }.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if config.max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive")
        if config.max_retries < 0 or config.max_retries > 8:
            raise ValueError("max_retries must be in 0..8")
        parsed = urlparse(config.base_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"unsupported scheme {parsed.scheme!r} in base_url")
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self.path_prefix = parsed.path.rstrip("/")
        self.is_https = parsed.scheme == "https"

    def _open_connection(self) -> http.client.HTTPConnection:
        if self.is_https:
            import ssl

            context = ssl.create_default_context()
            return http.client.HTTPSConnection(
                self.host,
                self.port,
                timeout=self.config.connect_timeout,
                context=context,
            )
        return http.client.HTTPConnection(
            self.host,
            self.port,
            timeout=self.config.connect_timeout,
        )

    def check_model_alias(self) -> list[str]:
        """Verify the /v1/models endpoint and confirm the expected alias is present."""
        path = f"{self.path_prefix}/models"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }
        conn = self._open_connection()
        try:
            conn.request("GET", path, headers=headers)
            response = conn.getresponse()
            raw_body = response.read(self.config.max_response_bytes + 1)
            if len(raw_body) > self.config.max_response_bytes:
                raise ResponseTooLargeError("models response exceeded limit")
            if response.status != 200:
                raise ModelRequestError(response.status, "failed to query /v1/models", raw_body.decode(errors="replace"))
            try:
                data = json.loads(raw_body)
            except json.JSONDecodeError as exc:
                raise StreamMalformedError(f"invalid JSON from /v1/models: {exc}") from exc
        except http.client.IncompleteRead as exc:
            raise StreamMalformedError(f"truncated non-streaming response: {exc}") from exc
        except (socket.timeout, TimeoutError) as exc:
            raise StreamTimeoutError(f"timeout querying models: {exc}") from exc
        except (OSError, http.client.HTTPException) as exc:
            raise TransportError(f"connection error querying models: {exc}") from exc
        finally:
            conn.close()

        if not isinstance(data, dict) or "data" not in data or not isinstance(data["data"], list):
            raise StreamMalformedError("expected {'data': [...]} structure from /v1/models")

        model_ids: list[str] = []
        for item in data["data"]:
            if isinstance(item, dict) and "id" in item and isinstance(item["id"], str):
                model_ids.append(item["id"])

        if model_ids != [self.config.expected_model_alias]:
            raise AliasMismatchError(
                f"expected exactly one alias {self.config.expected_model_alias!r}; received {model_ids!r}"
            )
        return model_ids

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        stream: bool = True,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int | None = None,
        seed: int | None = None,
        cancellation_check: Callable[[], bool] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatCompletionResult:
        """Send a chat completion request with bounded retries on transient errors."""
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a nonempty list")
        if max_tokens is not None and (isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0):
            raise ValueError("max_tokens must be a positive integer")
        if extra_body and any(key in extra_body for key in ("model", "messages", "stream")):
            raise ValueError("extra_body cannot override model, messages, or stream")
        payload: dict[str, Any] = {
            "model": self.config.expected_model_alias,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if seed is not None:
            payload["seed"] = seed
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice
        if stream:
            payload["stream_options"] = {"include_usage": True}
        if extra_body:
            payload.update(extra_body)

        body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")

        attempt = 0
        last_transient_error: Exception | None = None
        while attempt <= self.config.max_retries:
            attempt += 1
            if cancellation_check and cancellation_check():
                raise RequestCancelledError("request cancelled before send")
            try:
                if stream:
                    return self._chat_stream(body_bytes, cancellation_check=cancellation_check)
                return self._chat_non_stream(body_bytes, cancellation_check=cancellation_check)
            except _TransientConnectionError as exc:
                last_transient_error = exc
                if attempt <= self.config.max_retries:
                    time.sleep(self.config.retry_backoff_seconds * attempt)
                    continue
                raise TransportError(f"connection failed after {attempt} attempts: {exc}") from exc
            except ModelRequestError:
                raise
            except TransportError:
                raise

        raise TransportError(f"request failed: {last_transient_error}")

    def _chat_non_stream(
        self,
        body_bytes: bytes,
        *,
        cancellation_check: Callable[[], bool] | None = None,
    ) -> ChatCompletionResult:
        request_sent = time.monotonic()
        path = f"{self.path_prefix}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Length": str(len(body_bytes)),
        }
        conn = self._open_connection()
        try:
            try:
                conn.connect()
            except (OSError, http.client.HTTPException) as exc:
                raise _TransientConnectionError(str(exc)) from exc
            conn.sock.settimeout(min(self.config.connect_timeout, self.config.total_timeout)) if conn.sock else None
            conn.request("POST", path, body=body_bytes, headers=headers)
            response = conn.getresponse()
            if conn.sock:
                conn.sock.settimeout(self.config.read_timeout)
            first_byte_time = time.monotonic()
            if conn.sock:
                conn.sock.settimeout(min(self.config.read_timeout, self.config.total_timeout))
            raw_body = response.read(self.config.max_response_bytes + 1)
            if cancellation_check and cancellation_check():
                raise RequestCancelledError("request cancelled during non-streaming response")
            if len(raw_body) > self.config.max_response_bytes:
                raise ResponseTooLargeError(
                    f"response size exceeded maximum {self.config.max_response_bytes} bytes"
                )
            if response.status != 200:
                raise ModelRequestError(response.status, "chat completion error", raw_body.decode(errors="replace"))
            try:
                data = json.loads(raw_body)
            except json.JSONDecodeError as exc:
                raise StreamMalformedError(f"invalid JSON in response: {exc}") from exc
        except http.client.IncompleteRead as exc:
            raise StreamMalformedError(f"truncated non-streaming response: {exc}") from exc
        except (socket.timeout, TimeoutError) as exc:
            raise StreamTimeoutError(f"timeout during non-streaming chat: {exc}") from exc
        except (OSError, http.client.HTTPException) as exc:
            raise TransportError(f"connection error during non-streaming chat: {exc}") from exc
        finally:
            conn.close()

        if not isinstance(data, dict):
            raise StreamMalformedError("non-stream response must be a JSON object")
        response_model = data.get("model")
        if response_model is not None and response_model != self.config.expected_model_alias:
            raise AliasMismatchError(
                f"chat response model {response_model!r} does not match {self.config.expected_model_alias!r}"
            )
        choices = data.get("choices", [])
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise StreamMalformedError("non-stream response must contain exactly one choice object")
        content = ""
        reasoning = ""
        tool_calls: list[ToolCall] = []
        finish_reason = None
        choice = choices[0]
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise StreamMalformedError("finish_reason must be a string or null")
        message = choice.get("message", {})
        if not isinstance(message, dict):
            raise StreamMalformedError("choice.message must be an object")
        content_value = message.get("content")
        reasoning_value = message.get("reasoning_content")
        if reasoning_value is None:
            reasoning_value = message.get("reasoning")
        if content_value is not None and not isinstance(content_value, str):
            raise StreamMalformedError("content and reasoning must be strings or null")
        if reasoning_value is not None and not isinstance(reasoning_value, str):
            raise StreamMalformedError("content and reasoning must be strings or null")
        content = content_value or ""
        reasoning = reasoning_value or ""
        raw_tool_calls = message.get("tool_calls", [])
        if not isinstance(raw_tool_calls, list):
            raise StreamMalformedError("message.tool_calls must be a list")
        for raw_tc in raw_tool_calls:
            tool_calls.append(self._parse_complete_tool_call(raw_tc))

        usage: TransportUsage | None = None
        if "usage" in data:
            usage = self._parse_usage(data["usage"])

        now = time.monotonic()
        first_token = first_byte_time if (content or reasoning or tool_calls) else None
        first_visible = first_byte_time if content else None
        first_tool = first_byte_time if tool_calls else None

        return ChatCompletionResult(
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            http_status=response.status,
            request_sent_time=request_sent,
            first_byte_time=first_byte_time,
            first_token_time=first_token,
            first_visible_time=first_visible,
            first_tool_time=first_tool,
            last_token_time=now if (content or reasoning or tool_calls) else None,
            end_to_end_latency=now - request_sent,
            chunks_count=1,
        )

    def _chat_stream(
        self,
        body_bytes: bytes,
        *,
        cancellation_check: Callable[[], bool] | None = None,
    ) -> ChatCompletionResult:
        request_sent = time.monotonic()
        path = f"{self.path_prefix}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Length": str(len(body_bytes)),
        }
        conn = self._open_connection()
        try:
            try:
                conn.connect()
            except (OSError, http.client.HTTPException) as exc:
                raise _TransientConnectionError(str(exc)) from exc
            conn.sock.settimeout(min(self.config.connect_timeout, self.config.total_timeout)) if conn.sock else None
            conn.request("POST", path, body=body_bytes, headers=headers)
            response = conn.getresponse()
            if response.status != 200:
                err_body = response.read(16384).decode(errors="replace")
                raise ModelRequestError(response.status, "streaming chat completion error", err_body)

            sock = conn.sock
            if sock is None and hasattr(response, "fp") and hasattr(response.fp, "raw"):
                sock = getattr(response.fp.raw, "_sock", None)
            if sock:
                sock.settimeout(self.config.read_timeout)

            deadline = request_sent + self.config.total_timeout
            bytes_read = 0
            first_byte_time: float | None = None
            first_token_time: float | None = None
            first_visible_time: float | None = None
            first_tool_time: float | None = None
            last_token_time: float | None = None

            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            # Track tool calls by index: {index: {"id": str, "name": str, "arguments": [str, ...]}}
            tool_calls_builder: dict[int, dict[str, Any]] = {}
            finish_reason: str | None = None
            usage: TransportUsage | None = None
            chunks_count = 0
            saw_done = False

            event_data_lines: list[bytes] = []

            def process_event(data_bytes: bytes) -> bool:
                """Process one complete SSE event; return True for [DONE]."""
                nonlocal chunks_count, finish_reason, usage
                nonlocal first_token_time, first_visible_time, first_tool_time, last_token_time
                try:
                    data_str = data_bytes.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise StreamMalformedError(f"SSE data is not UTF-8: {exc}") from exc
                if data_str.strip() == "[DONE]":
                    return True
                try:
                    parsed_chunk = json.loads(data_str)
                except json.JSONDecodeError as exc:
                    raise StreamMalformedError(f"malformed SSE JSON chunk: {exc} in {data_str!r}") from exc
                if not isinstance(parsed_chunk, dict):
                    raise StreamMalformedError("SSE JSON chunk must be an object")
                response_model = parsed_chunk.get("model")
                if response_model is not None and response_model != self.config.expected_model_alias:
                    raise AliasMismatchError(
                        f"stream response model {response_model!r} does not match {self.config.expected_model_alias!r}"
                    )
                chunks_count += 1
                event_time = time.monotonic()
                if "usage" in parsed_chunk:
                    usage = self._parse_usage(parsed_chunk["usage"])
                choices = parsed_chunk.get("choices", [])
                if not isinstance(choices, list):
                    raise StreamMalformedError("SSE choices must be a list")
                if not choices:
                    return False
                if len(choices) != 1 or not isinstance(choices[0], dict):
                    raise StreamMalformedError("SSE chunk must contain at most one choice object")
                choice = choices[0]
                if choice.get("finish_reason") is not None:
                    if not isinstance(choice["finish_reason"], str):
                        raise StreamMalformedError("finish_reason must be a string or null")
                    finish_reason = choice["finish_reason"]
                delta = choice.get("delta", {})
                if not isinstance(delta, dict):
                    raise StreamMalformedError("choice.delta must be an object")
                has_token = False
                r_text = delta.get("reasoning_content")
                if r_text is None:
                    r_text = delta.get("reasoning")
                if r_text is not None:
                    if not isinstance(r_text, str):
                        raise StreamMalformedError("reasoning delta must be a string")
                    if r_text:
                        reasoning_parts.append(r_text)
                        has_token = True
                c_text = delta.get("content")
                if c_text is not None:
                    if not isinstance(c_text, str):
                        raise StreamMalformedError("content delta must be a string")
                    if c_text:
                        content_parts.append(c_text)
                        has_token = True
                        if first_visible_time is None:
                            first_visible_time = event_time
                t_calls = delta.get("tool_calls")
                if t_calls is not None:
                    if not isinstance(t_calls, list):
                        raise StreamMalformedError("tool_calls delta must be a list")
                    if t_calls:
                        has_token = True
                        if first_tool_time is None:
                            first_tool_time = event_time
                    for tc_delta in t_calls:
                        if not isinstance(tc_delta, dict):
                            raise StreamMalformedError("tool call delta must be an object")
                        idx = tc_delta.get("index")
                        if isinstance(idx, bool) or not isinstance(idx, int) or idx < 0 or idx > 1024:
                            raise StreamMalformedError("tool call delta index must be an integer in 0..1024")
                        entry = tool_calls_builder.setdefault(idx, {
                            "id": "", "name": "", "arguments": [], "type": "function",
                        })
                        for key in ("id", "type"):
                            value = tc_delta.get(key)
                            if value is not None:
                                if not isinstance(value, str):
                                    raise StreamMalformedError(f"tool call {key} must be a string")
                                if entry[key] and entry[key] != value:
                                    raise StreamMalformedError(f"conflicting tool call {key} fragments")
                                entry[key] = value
                        fn = tc_delta.get("function")
                        if fn is not None:
                            if not isinstance(fn, dict):
                                raise StreamMalformedError("tool call function delta must be an object")
                            name = fn.get("name")
                            if name is not None:
                                if not isinstance(name, str):
                                    raise StreamMalformedError("tool call name must be a string")
                                if entry["name"] and entry["name"] != name:
                                    raise StreamMalformedError("conflicting tool call name fragments")
                                entry["name"] = name
                            arguments = fn.get("arguments")
                            if arguments is not None:
                                if not isinstance(arguments, str):
                                    raise StreamMalformedError("tool call arguments fragment must be a string")
                                entry["arguments"].append(arguments)
                if has_token:
                    if first_token_time is None:
                        first_token_time = event_time
                    last_token_time = event_time
                return False

            while True:
                if cancellation_check and cancellation_check():
                    conn.close()
                    raise RequestCancelledError("request cancelled during streaming")

                now = time.monotonic()
                if now > deadline:
                    conn.close()
                    raise StreamTimeoutError("total request timeout exceeded during streaming")

                # Dynamic read timeout to not overshoot deadline
                remaining = min(self.config.read_timeout, max(0.1, deadline - now))
                if sock:
                    sock.settimeout(remaining)

                try:
                    # HTTPResponse.read(4096) may wait for all 4096 bytes. readline
                    # returns as soon as an SSE line arrives, preserving TTFT and
                    # applying backpressure one bounded line at a time.
                    chunk = response.readline(min(1024 * 1024, self.config.max_response_bytes) + 1)
                except (socket.timeout, TimeoutError) as exc:
                    conn.close()
                    raise StreamTimeoutError(f"read timeout waiting for stream data: {exc}") from exc
                except OSError as exc:
                    conn.close()
                    raise TransportError(f"socket error during stream read: {exc}") from exc

                if not chunk:
                    # Connection closed by server
                    break

                if first_byte_time is None:
                    first_byte_time = time.monotonic()

                bytes_read += len(chunk)
                if bytes_read > self.config.max_response_bytes:
                    conn.close()
                    raise ResponseTooLargeError(
                        f"stream exceeded maximum allowed size of {self.config.max_response_bytes} bytes"
                    )

                if len(chunk) > min(1024 * 1024, self.config.max_response_bytes):
                    raise ResponseTooLargeError("individual SSE line exceeded the bounded line size")
                line_bytes = chunk.rstrip(b"\r\n")
                if not line_bytes:
                    if event_data_lines:
                        saw_done = process_event(b"\n".join(event_data_lines))
                        event_data_lines.clear()
                    if saw_done:
                        break
                    continue
                if line_bytes.startswith(b":"):
                    continue
                if line_bytes.startswith(b"data:"):
                    value = line_bytes[5:]
                    if value.startswith(b" "):
                        value = value[1:]
                    event_data_lines.append(value)
                    continue
                if not line_bytes.startswith((b"event:", b"id:", b"retry:")):
                    raise StreamMalformedError(f"malformed SSE field: {line_bytes[:80]!r}")

            if event_data_lines:
                # EOF in the middle of an event is always truncation, even if
                # the bytes happen to form valid JSON.
                raise StreamMalformedError("stream terminated with an incomplete SSE event")
            if not saw_done:
                raise StreamMalformedError("stream terminated before the [DONE] marker")

        finally:
            conn.close()

        # Assemble tool calls
        assembled_tool_calls: list[ToolCall] = []
        for idx in sorted(tool_calls_builder.keys()):
            entry = tool_calls_builder[idx]
            assembled_tool_calls.append(self._parse_complete_tool_call({
                "id": entry["id"],
                "type": entry.get("type", "function"),
                "function": {"name": entry["name"], "arguments": "".join(entry["arguments"])},
            }))

        end_time = time.monotonic()
        return ChatCompletionResult(
            content="".join(content_parts),
            reasoning="".join(reasoning_parts),
            tool_calls=assembled_tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            http_status=response.status,
            request_sent_time=request_sent,
            first_byte_time=first_byte_time,
            first_token_time=first_token_time,
            first_visible_time=first_visible_time,
            first_tool_time=first_tool_time,
            last_token_time=last_token_time,
            end_to_end_latency=end_time - request_sent,
            chunks_count=chunks_count,
        )

    @staticmethod
    def _parse_usage(raw: Any) -> TransportUsage:
        if not isinstance(raw, dict):
            raise StreamMalformedError("usage must be an object")
        values: dict[str, int | None] = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = raw.get(key)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise StreamMalformedError(f"usage.{key} must be a nonnegative integer or null")
            values[key] = value
        return TransportUsage(**values)

    @staticmethod
    def _parse_complete_tool_call(raw: Any) -> ToolCall:
        if not isinstance(raw, dict):
            raise StreamMalformedError("tool call must be an object")
        call_id = raw.get("id")
        call_type = raw.get("type", "function")
        function = raw.get("function")
        if not isinstance(call_id, str) or not call_id:
            raise StreamMalformedError("complete tool call is missing a nonempty id")
        if call_type != "function" or not isinstance(function, dict):
            raise StreamMalformedError("only function tool calls are supported")
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(name, str) or not name or not isinstance(arguments, str):
            raise StreamMalformedError("complete tool call requires a name and string arguments")
        try:
            decoded = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise StreamMalformedError(f"complete tool arguments are invalid JSON: {exc}") from exc
        if not isinstance(decoded, dict):
            raise StreamMalformedError("complete tool arguments must decode to an object")
        return ToolCall(id=call_id, name=name, arguments=arguments, type=call_type)
