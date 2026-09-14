"""Checks for profile routing and the actual HTTP preflight boundary."""
import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import unittest


spec = importlib.util.spec_from_file_location(
    "launch_codex", Path(__file__).resolve().parents[1] / "launch-codex.py")
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/redirect/models":
                    self.send_response(302)
                    self.send_header("Location", "/v1/models")
                    self.end_headers()
                    return
                self.send_response(200)
                self.end_headers()
                self.wfile.write(cls.payload)

            def log_message(self, *args):
                pass

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        type(self).payload = json.dumps({"data": [{"id": "test", "meta": {"n_ctx": 32768}}]}).encode()

    def test_every_profile_routes_to_same_model_and_limit(self):
        for path in launcher.CONFIG_DIR.glob("opencode-*.json"):
            with self.subTest(path=path):
                model, context, url = launcher.load_profile(path.stem[9:])
                config = json.loads(path.read_text())
                self.assertEqual("pinnacles/" + model, config["model"])
                self.assertEqual(context, config["provider"]["pinnacles"]["models"][model]["limit"]["context"])
                self.assertEqual(launcher.validate_url(url), url)

    def test_unknown_and_traversal_profile_rejected(self):
        for name in ("missing", "../gemma", "/tmp/file"):
            with self.assertRaises(ValueError):
                launcher.load_profile(name)

    def test_local_url_boundary(self):
        self.assertEqual(launcher.validate_url(self.base + "/v1/"), self.base + "/v1")
        for url in ("https://example.com/v1", "http://user:pass@localhost/v1",
                    "http://localhost/v1?key=x", "http://localhost:0/v1", "http://localhost/no"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                launcher.validate_url(url)

    def test_context_clamped_in_both_directions(self):
        self.assertEqual(launcher.inspect_endpoint(self.base + "/v1", "test", 131072), 32768)
        self.assertEqual(launcher.inspect_endpoint(self.base + "/v1", "test", 16384), 16384)

    def test_wrong_model_rejected(self):
        with self.assertRaises(ValueError):
            launcher.inspect_endpoint(self.base + "/v1", "other", 32768)

    def test_malformed_and_oversize_catalogs(self):
        for payload in (b"[]", b"bad json", b'{"data":null}',
                        b"x" * (launcher.MAX_CATALOG_BYTES + 1)):
            type(self).payload = payload
            with self.subTest(size=len(payload)), self.assertRaises(ValueError):
                launcher.inspect_endpoint(self.base + "/v1", "test", 32768)

    def test_redirect_rejected(self):
        with self.assertRaises(ValueError):
            launcher.inspect_endpoint(self.base + "/redirect", "test", 32768)

    def test_arguments_preserved_without_shell_interpretation(self):
        extra = ["exec", "literal $(no) `no` and spaces"]
        command = launcher.codex_command("/a path/codex", "test", 32768, self.base + "/v1", extra)
        self.assertEqual(command[-2:], extra)
        self.assertEqual(command[0], "/a path/codex")
        self.assertIn('model_providers.pinnacles.wire_api="responses"', command)


if __name__ == "__main__":
    unittest.main()
