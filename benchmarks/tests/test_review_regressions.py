"""CPU-only regressions for GPU selection, profile launch, and score provenance."""

import os
from pathlib import Path
import subprocess
import signal
import select
import sys
import time
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.adapters.livecodebench import evaluate_python_code
from benchmarks.contracts import load_profiles
from benchmarks.server_runner import server_environment, stop_owned


ROOT = Path(__file__).resolve().parents[2]


class ReviewRegressions(unittest.TestCase):
    def test_shutdown_cleans_workers_after_leader_exits(self):
        worker = "import os,signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print(os.getpid(),flush=True); time.sleep(60)"
        leader = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{worker!r}]); time.sleep(60)"
        process = subprocess.Popen([sys.executable, "-c", leader], start_new_session=True,
                                   stdout=subprocess.PIPE, text=True)
        try:
            self.assertTrue(select.select([process.stdout], [], [], 5)[0])
            worker_pid = int(process.stdout.readline())
            stop_owned(process)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    state = Path(f"/proc/{worker_pid}/stat").read_text().split(") ", 1)[1][0]
                except FileNotFoundError:
                    break
                if state == "Z":
                    break
                time.sleep(0.01)
            else:
                self.fail("owned worker survived server shutdown")
        finally:
            stop_owned(process)
            process.stdout.close()

    def gpu_guard(self, family, visible, rows, gpu="a100"):
        common = "gemma-qat" if family == "gemma" else "muse-gguf"
        env = dict(os.environ, SLURM_GPUS_ON_NODE="2", CUDA_VISIBLE_DEVICES=visible,
                   GPU_ROWS=rows, **{f"{family.upper()}_GPU": gpu})
        return subprocess.run(
            ["bash", "-c", '''set -euo pipefail
source "$1"
nvidia-smi() {
  [[ $1 == -i && $2 == "$CUDA_VISIBLE_DEVICES" ]] || return 2
  printf '%s\\n' "$GPU_ROWS"
}
"$2"
''', "guard", str(ROOT / f"scripts/{common}-common.sh"), f"{family}_multigpu_gpu"],
            env=env, capture_output=True, text=True, timeout=5,
        )

    def test_two_gpu_guard_validates_each_selected_device(self):
        a100 = "NVIDIA A100-PCIE-40GB, 40960"
        l40s = "NVIDIA L40S, 46068"
        for family in ("gemma", "muse"):
            for visible in ("0,1", "GPU-abcd,GPU-1234"):
                with self.subTest(family=family, visible=visible):
                    self.assertEqual(self.gpu_guard(family, visible, f"{a100}\n{a100}").returncode, 0)
            for visible, rows in (("", f"{a100}\n{a100}"), ("0", f"{a100}\n{a100}"),
                                  ("0,0", f"{a100}\n{a100}"), ("0,1", f"{a100}\n{l40s}"),
                                  ("0,1", a100)):
                with self.subTest(family=family, visible=visible, rows=rows):
                    self.assertNotEqual(self.gpu_guard(family, visible, rows).returncode, 0)
            self.assertEqual(self.gpu_guard(family, "0,1", f"{l40s}\n{l40s}", "l40s").returncode, 0)

    def test_profile_launch_overrides_stale_session_limits(self):
        profiles = load_profiles(ROOT / "benchmarks/profiles.json", ROOT)
        with patch.dict(os.environ, {"GEMMA_CONTEXT": "131072", "GEMMA_GPU": "l40s"}):
            env = server_environment(profiles["gemma-qat-a100"])
        self.assertEqual(env["GEMMA_CONTEXT"], "32768")
        self.assertEqual(env["GEMMA_GPU"], "a100")
        self.assertEqual(env["GEMMA_OUTPUT"], "8192")
        self.assertEqual(env["GEMMA_PARALLEL"], "1")

    def test_official_evaluator_failure_never_falls_back_to_another_grader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = "print('correct')"
            cases = [{"input": "", "output": "correct"}]
            passed, reason = evaluate_python_code(code, cases, evaluator_python=root / "missing",
                                                  evaluator_src_dir=root)
            self.assertFalse(passed)
            self.assertTrue(reason.startswith("evaluator_infra"))
            with patch("benchmarks.adapters.livecodebench.subprocess.run",
                       return_value=subprocess.CompletedProcess([], 1, "", "import failed")) as run:
                passed, reason = evaluate_python_code(code, cases, evaluator_python=Path(__file__),
                                                      evaluator_src_dir=root)
            self.assertFalse(passed)
            self.assertTrue(reason.startswith("evaluator_infra"))
            self.assertEqual(run.call_count, 1)


if __name__ == "__main__":
    unittest.main()
