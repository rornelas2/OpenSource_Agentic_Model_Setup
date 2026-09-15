"""Model-native thinking controls and a session-owned Responses adapter.

Only localhost is used (validated by the launcher). Requests are capped at 8 MiB;
JSON processing is O(request bytes), responses stream in bounded 64 KiB chunks.
The adapter never retries requests or logs prompts and never owns the GPU server.
"""
from contextlib import contextmanager
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading
from urllib.parse import urlsplit

MUSE = {"muse-glimmer-30b", "muse-glimmer-30b-dynamic"}
BINARY = {"gemma4-31b", "gemma-4-31b-qat", "nemotron-3.5-lightning"}
MAX_BODY = 8 * 1024 * 1024


def levels(model):
    if model in MUSE:
        return ("low", "medium", "high", "xhigh")
    if model == "nemotron-3-super":
        return ("none", "low", "high")
    if model in BINARY:
        return ("none", "high")
    raise ValueError("No thinking controls registered for model: " + model)


def default_effort(model):
    levels(model)
    return "none" if model.startswith("gemma") else "high"


def template_kwargs(model, effort):
    if effort not in levels(model):
        raise ValueError(f"Unsupported effort {effort!r} for {model}; choose {', '.join(levels(model))}")
    if model == "nemotron-3-super":
        return {"enable_thinking": effort != "none", "low_effort": effort == "low"}
    return ({"reasoning_strength": effort} if model in MUSE
            else {"enable_thinking": effort != "none"})


def translate(body, model):
    if not isinstance(body, dict) or body.get("model") != model:
        raise ValueError("Request must select the launcher's model.")
    reasoning = body.get("reasoning", {})
    if reasoning is None:
        reasoning = {}
    if not isinstance(reasoning, dict):
        raise ValueError("reasoning must be an object.")
    effort = reasoning.get("effort")
    if effort is None:
        effort = default_effort(model)
    kwargs = body.get("chat_template_kwargs", {})
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, dict):
        raise ValueError("chat_template_kwargs must be an object.")
    body = dict(body)
    body["chat_template_kwargs"] = {**kwargs, **template_kwargs(model, effort)}
    # The native template setting is authoritative. Avoid conflicting backend
    # interpretations of standard effort values; retain any unrelated fields.
    body["reasoning"] = {k: v for k, v in reasoning.items() if k != "effort"}
    if not body["reasoning"]:
        body.pop("reasoning")
    return body


@contextmanager
def responses_adapter(url, model):
    """Bind an ephemeral loopback port; close upstream sockets on session exit."""
    target = urlsplit(url)
    connections = set()
    lock = threading.Lock()
    slots = threading.BoundedSemaphore(8)
    stopping = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(180)

        def log_message(self, *args):
            pass

        def do_POST(self):
            if self.path not in {"/v1/responses", "/v1/responses/input_tokens"}:
                self.send_error(404)
                return
            if not slots.acquire(blocking=False):
                self.send_error(503, "Too many active requests")
                return
            upstream = None
            response = None
            upstream_socket = None
            started = False
            try:
                if self.headers.get("Transfer-Encoding") or self.headers.get("Content-Encoding"):
                    raise ValueError("Only uncompressed Content-Length requests are supported.")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY:
                    raise ValueError("Request body must be between 1 byte and 8 MiB.")
                raw = self.rfile.read(length)
                if len(raw) != length:
                    raise ValueError("Truncated request body.")
                payload = json.dumps(translate(json.loads(raw), model)).encode()
                upstream = http.client.HTTPConnection(target.hostname, target.port, timeout=180)
                upstream.connect()
                upstream_socket = upstream.sock
                with lock:
                    if stopping.is_set():
                        raise OSError("Session is closing")
                    connections.add(upstream_socket)
                upstream.request("POST", target.path.rstrip("/") + self.path[3:], payload,
                                 {"Content-Type": "application/json", "Accept": "text/event-stream"})
                response = upstream.getresponse()
                self.send_response(response.status)
                self.send_header("Content-Type", response.getheader("Content-Type", "application/json"))
                self.send_header("Connection", "close")
                self.end_headers()
                started = True
                self.close_connection = True
                while chunk := response.read1(65536):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (ValueError, UnicodeError) as exc:
                if not started:
                    self.send_error(400, str(exc))
            except (OSError, http.client.HTTPException):
                if not started:
                    self.send_error(502, "Local model server connection failed")
            finally:
                if response is not None:
                    response.close()
                if upstream is not None:
                    upstream.close()
                    with lock:
                        connections.discard(upstream_socket)
                slots.release()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        stopping.set()
        server.shutdown()
        server.server_close()
        with lock:
            for connection in connections:
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                connection.close()
        thread.join()
