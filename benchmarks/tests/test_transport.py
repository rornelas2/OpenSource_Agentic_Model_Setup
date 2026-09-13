"""Unit tests for bounded model transport with fake HTTP/SSE server."""

from __future__ import annotations

import http.server
import json
import socket
import threading
import time
import unittest
from unittest.mock import patch

from benchmarks.transport import (
    AliasMismatchError,
    ChatCompletionResult,
    ModelRequestError,
    ModelTransportClient,
    ModelTransportConfig,
    RequestCancelledError,
    ResponseTooLargeError,
    StreamMalformedError,
    StreamTimeoutError,
    ToolCall,
    TransportUsage,
    _TransientConnectionError,
)


class FakeServerHandler(http.server.BaseHTTPRequestHandler):
    """Customizable handler for fake OpenAI-compatible server."""

    protocol_version = "HTTP/1.1"
    scenario: str = "normal"
    model_alias: str = "test-model"

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        self.close_connection = True
        if self.path.endswith("/models"):
            if self.scenario == "wrong_alias":
                body = json.dumps({"data": [{"id": "wrong-model"}]}).encode()
            elif self.scenario == "malformed_models":
                body = b"{not-json"
            elif self.scenario == "server_error":
                self.send_response(500)
                self.send_header("Connection", "close")
                self.send_header("Content-Length", "21")
                self.end_headers()
                self.wfile.write(b"internal server error")
                return
            else:
                body = json.dumps({"data": [{"id": self.model_alias}]}).encode()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Connection", "close")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.send_header("Connection", "close")
        self.end_headers()

    def do_POST(self) -> None:
        self.close_connection = True
        if not self.path.endswith("/chat/completions"):
            self.send_response(404)
            self.send_header("Connection", "close")
            self.end_headers()
            return

        content_len = int(self.headers.get("Content-Length", 0))
        req_body = json.loads(self.rfile.read(content_len)) if content_len > 0 else {}
        stream = req_body.get("stream", False)

        if self.scenario == "http_400":
            body = b'{"error": "context length exceeded"}'
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Connection", "close")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if not stream:
            resp = {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 12345,
                "model": self.model_alias,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello non-streaming!",
                            "reasoning_content": "Thinking step by step.",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "test_fn", "arguments": '{"x": 1}'},
                                }
                            ],
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
            body = json.dumps(resp).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Connection", "close")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        def safe_write(data: bytes) -> bool:
            try:
                self.wfile.write(data)
                self.wfile.flush()
                return True
            except (BrokenPipeError, ConnectionResetError, OSError):
                return False

        if self.scenario == "split_chunks":
            events = [
                'data: {"choices":[{"index":0,"delta":{"content":"Hel"}}]}\n\n',
                'data: {"choices":[{"index":0,"delta":{"content":"lo"}}]}\n\n',
                'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
                "data: [DONE]\n\n",
            ]
            full_stream = "".join(events).encode("utf-8")
            for i in range(0, len(full_stream), 5):
                if not safe_write(full_stream[i : i + 5]):
                    break
            return

        if self.scenario == "coalesced_chunks":
            events = (
                'data: {"choices":[{"index":0,"delta":{"content":"A"}}]}\n\n'
                'data: {"choices":[{"index":0,"delta":{"content":"B"}}]}\n\n'
                'data: {"choices":[{"index":0,"delta":{"content":"C"},"finish_reason":"stop"}]}\n\n'
                'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":3,"total_tokens":6}}\n\n'
                "data: [DONE]\n\n"
            )
            safe_write(events.encode("utf-8"))
            return

        if self.scenario == "reasoning_only":
            events = [
                'data: {"choices":[{"index":0,"delta":{"reasoning_content":"Let me calculate..."}}]}\n\n',
                'data: {"choices":[{"index":0,"delta":{"reasoning_content":" 2 + 2 = 4."}}]}\n\n',
                'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
                "data: [DONE]\n\n",
            ]
            for ev in events:
                if not safe_write(ev.encode("utf-8")):
                    break
            return

        if self.scenario == "multiple_tool_calls":
            events = [
                'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"read_file","arguments":""}}]}}]}\n\n',
                'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"path\\": \\"foo.py\\"}"}}]}}]}\n\n',
                'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_b","type":"function","function":{"name":"run_bash","arguments":"{\\"cmd\\": "}}]}}]}\n\n',
                'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"\\"pytest\\"}"}}]}}]}\n\n',
                'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n',
                "data: [DONE]\n\n",
            ]
            for ev in events:
                if not safe_write(ev.encode("utf-8")):
                    break
            return

        if self.scenario == "usage_only_final":
            events = [
                'data: {"choices":[{"index":0,"delta":{"content":"Answer: 42"}}]}\n\n',
                'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
                'data: {"choices":[],"usage":{"prompt_tokens":20,"completion_tokens":4,"total_tokens":24}}\n\n',
                "data: [DONE]\n\n",
            ]
            for ev in events:
                if not safe_write(ev.encode("utf-8")):
                    break
            return

        if self.scenario == "malformed_stream":
            safe_write(b"data: {invalid-json}\n\n")
            return

        if self.scenario == "truncated_stream":
            safe_write(b'data: {"choices":[{"index":0,"delta":{"content":"Incomplete..."}}]}\n\n')
            return

        if self.scenario == "timeout_simulation":
            safe_write(b'data: {"choices":[{"index":0,"delta":{"content":"Wait"}}]}\n\n')
            time.sleep(1.0)
            safe_write(b"data: [DONE]\n\n")
            return

        if self.scenario == "cancellation_simulation":
            for i in range(20):
                line = f'data: {{"choices":[{{"index":0,"delta":{{"content":"word{i} "}}}}]}}\n\n'.encode()
                if not safe_write(line):
                    break
                time.sleep(0.05)
            safe_write(b"data: [DONE]\n\n")
            return

        events = [
            'data: {"choices":[{"index":0,"delta":{"content":"Normal "}}]}\n\n',
            'data: {"choices":[{"index":0,"delta":{"content":"stream."}}]}\n\n',
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
            "data: [DONE]\n\n",
        ]
        for ev in events:
            if not safe_write(ev.encode("utf-8")):
                break


class TransportTests(unittest.TestCase):
    server: http.server.HTTPServer
    server_thread: threading.Thread
    port: int

    @classmethod
    def setUpClass(cls) -> None:
        cls.server = http.server.HTTPServer(("127.0.0.1", 0), FakeServerHandler)
        cls.port = cls.server.server_port
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()

    def make_client(
        self,
        alias: str = "test-model",
        read_timeout: float = 2.0,
        total_timeout: float = 5.0,
        max_response_bytes: int = 1024 * 1024,
    ) -> ModelTransportClient:
        config = ModelTransportConfig(
            base_url=f"http://127.0.0.1:{self.port}/v1",
            expected_model_alias=alias,
            connect_timeout=2.0,
            read_timeout=read_timeout,
            total_timeout=total_timeout,
            max_response_bytes=max_response_bytes,
        )
        return ModelTransportClient(config)

    def test_check_model_alias_success(self) -> None:
        FakeServerHandler.scenario = "normal"
        FakeServerHandler.model_alias = "test-model"
        client = self.make_client(alias="test-model")
        models = client.check_model_alias()
        self.assertIn("test-model", models)

    def test_check_model_alias_mismatch(self) -> None:
        FakeServerHandler.scenario = "wrong_alias"
        client = self.make_client(alias="expected-alias")
        with self.assertRaises(AliasMismatchError):
            client.check_model_alias()

    def test_non_streaming_chat(self) -> None:
        FakeServerHandler.scenario = "normal"
        client = self.make_client()
        result = client.chat_completion([{"role": "user", "content": "Hi"}], stream=False)
        self.assertEqual(result.content, "Hello non-streaming!")
        self.assertEqual(result.reasoning, "Thinking step by step.")
        self.assertEqual(len(result.tool_calls), 1)
        self.assertEqual(result.tool_calls[0].name, "test_fn")
        self.assertEqual(result.tool_calls[0].arguments, '{"x": 1}')
        self.assertEqual(result.finish_reason, "stop")
        self.assertIsNotNone(result.usage)
        self.assertEqual(result.usage.total_tokens, 15)

    def test_streaming_split_sse_chunks(self) -> None:
        FakeServerHandler.scenario = "split_chunks"
        client = self.make_client()
        result = client.chat_completion([{"role": "user", "content": "Hi"}], stream=True)
        self.assertEqual(result.content, "Hello")
        self.assertEqual(result.finish_reason, "stop")
        self.assertIsNotNone(result.first_token_time)
        self.assertIsNotNone(result.first_visible_time)

    def test_streaming_coalesced_chunks(self) -> None:
        FakeServerHandler.scenario = "coalesced_chunks"
        client = self.make_client()
        result = client.chat_completion([{"role": "user", "content": "Hi"}], stream=True)
        self.assertEqual(result.content, "ABC")
        self.assertEqual(result.finish_reason, "stop")
        self.assertIsNotNone(result.usage)
        self.assertEqual(result.usage.total_tokens, 6)

    def test_streaming_reasoning_only_output(self) -> None:
        FakeServerHandler.scenario = "reasoning_only"
        client = self.make_client()
        result = client.chat_completion([{"role": "user", "content": "Hi"}], stream=True)
        self.assertEqual(result.content, "")
        self.assertEqual(result.reasoning, "Let me calculate... 2 + 2 = 4.")
        self.assertEqual(result.finish_reason, "stop")
        self.assertIsNotNone(result.first_token_time)
        self.assertIsNone(result.first_visible_time)

    def test_streaming_multiple_tool_calls(self) -> None:
        FakeServerHandler.scenario = "multiple_tool_calls"
        client = self.make_client()
        result = client.chat_completion([{"role": "user", "content": "Hi"}], stream=True)
        self.assertEqual(len(result.tool_calls), 2)
        self.assertEqual(result.tool_calls[0].id, "call_a")
        self.assertEqual(result.tool_calls[0].name, "read_file")
        self.assertEqual(result.tool_calls[0].arguments, '{"path": "foo.py"}')
        self.assertEqual(result.tool_calls[1].id, "call_b")
        self.assertEqual(result.tool_calls[1].name, "run_bash")
        self.assertEqual(result.tool_calls[1].arguments, '{"cmd": "pytest"}')
        self.assertEqual(result.finish_reason, "tool_calls")
        self.assertIsNotNone(result.first_tool_time)

    def test_streaming_usage_only_final_chunk(self) -> None:
        FakeServerHandler.scenario = "usage_only_final"
        client = self.make_client()
        result = client.chat_completion([{"role": "user", "content": "Hi"}], stream=True)
        self.assertEqual(result.content, "Answer: 42")
        self.assertEqual(result.finish_reason, "stop")
        self.assertIsNotNone(result.usage)
        self.assertEqual(result.usage.prompt_tokens, 20)
        self.assertEqual(result.usage.completion_tokens, 4)
        self.assertEqual(result.usage.total_tokens, 24)

    def test_malformed_stream(self) -> None:
        FakeServerHandler.scenario = "malformed_stream"
        client = self.make_client()
        with self.assertRaises(StreamMalformedError):
            client.chat_completion([{"role": "user", "content": "Hi"}], stream=True)

    def test_truncated_stream_is_rejected_even_after_valid_content(self) -> None:
        FakeServerHandler.scenario = "truncated_stream"
        client = self.make_client()
        with self.assertRaisesRegex(StreamMalformedError, "DONE"):
            client.chat_completion([{"role": "user", "content": "Hi"}], stream=True)

    def test_stream_read_timeout(self) -> None:
        FakeServerHandler.scenario = "timeout_simulation"
        client = self.make_client(read_timeout=0.2)
        with self.assertRaises(StreamTimeoutError):
            client.chat_completion([{"role": "user", "content": "Hi"}], stream=True)

    def test_stream_cancellation(self) -> None:
        FakeServerHandler.scenario = "cancellation_simulation"
        client = self.make_client()
        calls = 0

        def cancel_after_few_calls() -> bool:
            nonlocal calls
            calls += 1
            return calls >= 2

        with self.assertRaises(RequestCancelledError):
            client.chat_completion(
                [{"role": "user", "content": "Hi"}],
                stream=True,
                cancellation_check=cancel_after_few_calls,
            )

    def test_recovery_after_error_or_cancellation(self) -> None:
        FakeServerHandler.scenario = "cancellation_simulation"
        client = self.make_client()
        with self.assertRaises(RequestCancelledError):
            client.chat_completion(
                [{"role": "user", "content": "Hi"}],
                stream=True,
                cancellation_check=lambda: True,
            )

        FakeServerHandler.scenario = "normal"
        recovery = client.chat_completion([{"role": "user", "content": "Hi again"}], stream=True)
        self.assertEqual(recovery.content, "Normal stream.")

    def test_response_too_large(self) -> None:
        FakeServerHandler.scenario = "normal"
        client = self.make_client(max_response_bytes=10)
        with self.assertRaises(ResponseTooLargeError):
            client.chat_completion([{"role": "user", "content": "Hi"}], stream=True)

    def test_http_400_not_retried(self) -> None:
        FakeServerHandler.scenario = "http_400"
        client = self.make_client()
        with self.assertRaises(ModelRequestError) as cm:
            client.chat_completion([{"role": "user", "content": "Too long"}], stream=True)
        self.assertEqual(cm.exception.status, 400)

    def test_pre_send_connection_failure_is_retried_with_bound(self) -> None:
        client = self.make_client()
        expected = ChatCompletionResult(
            content="recovered", reasoning="", tool_calls=[], finish_reason="stop",
            usage=None, http_status=200, request_sent_time=0.0,
        )
        with patch.object(
            client,
            "_chat_non_stream",
            side_effect=[_TransientConnectionError("refused"), expected],
        ) as request, patch("benchmarks.transport.time.sleep"):
            result = client.chat_completion([{"role": "user", "content": "Hi"}], stream=False)
        self.assertIs(result, expected)
        self.assertEqual(request.call_count, 2)


if __name__ == "__main__":
    unittest.main()
