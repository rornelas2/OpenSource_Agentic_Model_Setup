from __future__ import annotations

import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from benchmarks.contracts import ContractError
from benchmarks.preflight import (
    check_container_runtime,
    check_endpoint,
    check_gpu,
    check_system_resources,
    owns_ipv4_loopback_listener,
)


class Handler(BaseHTTPRequestHandler):
    alias = "fixture-model"

    def do_GET(self) -> None:
        if self.path == "/health":
            payload = b""
        elif self.path == "/v1/models":
            payload = json.dumps({"data": [{"id": self.alias}]}).encode()
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        pass


class PreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def profile(self, alias: str = "fixture-model", gpu: str = "a100") -> dict:
        return {
            "endpoint": {
                "base_url": f"http://127.0.0.1:{self.server.server_port}/v1",
                "model_alias": alias,
            },
            "deployment": {"gpu": gpu},
        }

    def test_listener_and_endpoint_identity(self) -> None:
        self.assertTrue(owns_ipv4_loopback_listener(os.getpid(), self.server.server_port))
        result = check_endpoint(self.profile(), os.getpid())
        self.assertEqual(result["model_alias"], "fixture-model")

    def test_wrong_alias_fails(self) -> None:
        with self.assertRaisesRegex(ContractError, "expected only model alias"):
            check_endpoint(self.profile("wrong"), os.getpid())

    def test_container_runtime_not_required(self) -> None:
        res = check_container_runtime(False)
        self.assertFalse(res["required"])
        self.assertEqual(res["status"], "not_required")

    def test_container_runtime_detection(self) -> None:
        with patch("benchmarks.preflight.shutil.which", side_effect=lambda name: "/usr/bin/docker" if name == "docker" else None), patch("subprocess.run") as mock_run:
            mock_run.returncode = 0
            mock_run.return_value.stdout = '"24.0.5"'
            mock_run.return_value.stderr = ""
            res = check_container_runtime(True)
            self.assertTrue(res["required"])
            self.assertEqual(res["server_version"], "24.0.5")

    def test_rootless_podman_without_subids_fails_closed(self) -> None:
        def which(name: str) -> str | None:
            return "/usr/bin/podman" if name == "podman" else None

        with patch("benchmarks.preflight.shutil.which", side_effect=which), patch(
            "benchmarks.preflight.Path.read_text", return_value=""
        ):
            with self.assertRaisesRegex(ContractError, "no subordinate UID/GID ranges"):
                check_container_runtime(True)

    def test_wrong_gpu_profile_fails(self) -> None:
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value.returncode = 0
                mock_run.return_value.stdout = "NVIDIA A100-PCIE-40GB, GPU-12345, 40960\n"
                mock_run.return_value.stderr = ""
                # Expecting H200 but GPU is A100
                with self.assertRaisesRegex(ContractError, "does not match profile"):
                    check_gpu(self.profile(gpu="nvidia_h200_nvl"))

    def test_system_resources_check(self) -> None:
        res = check_system_resources(self.profile())
        self.assertIn("cpus_available", res)
        self.assertIn("tmp_disk_free_gib", res)
        self.assertGreaterEqual(res["cpus_available"], 1)


if __name__ == "__main__":
    unittest.main()
