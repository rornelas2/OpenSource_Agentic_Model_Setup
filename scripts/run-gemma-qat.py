#!/usr/bin/env python3
"""Own one server and optional client command inside a bounded Slurm job for Gemma QAT.

Usage: python scripts/run-gemma-qat.py [-- command argument ...]
Without a command, serve until the allocation or an interrupt ends the process.
Logs remain local. Readiness requires an owned listener, /health and /v1/models.
"""
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


def owns_listener(pid, port):
    """Only accept a loopback listener held by the server process we started."""
    try:
        sockets = set()
        for entry in Path(f'/proc/{pid}/fd').iterdir():
            try:
                sockets.add(entry.readlink().name)
            except FileNotFoundError:
                continue
        for line in Path('/proc/net/tcp').read_text().splitlines()[1:]:
            fields = line.split()
            if (fields[1] == f'0100007F:{port:04X}' and fields[3] == '0A'
                    and f'socket:[{fields[9]}]' in sockets):
                return True
    except (FileNotFoundError, ProcessLookupError):
        pass
    return False


def stop(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def interrupted(signum, frame):
    raise SystemExit(128 + signum)


def main():
    command = sys.argv[1:]
    if command[:1] == ['--']:
        command = command[1:]
    port_text = os.environ.get('GEMMA_PORT', '8000')
    if not port_text.isascii() or not port_text.isdecimal() or not 1024 <= int(port_text) <= 65535:
        raise ValueError('GEMMA_PORT must be in 1024..65535')
    port = int(port_text)
    repo = Path(__file__).resolve().parent.parent
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory = repo / '.local' / f'gemma-qat-{stamp}-{os.getpid()}'
    directory.mkdir(parents=True, mode=0o700)
    print(f'Server log: {directory / "server.log"}', flush=True)
    server = client = None
    start = time.monotonic()
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with (directory / 'server.log').open('w') as log:
            server = subprocess.Popen(['bash', str(repo / 'scripts/serve-gemma-qat.sh')],
                                      stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'Server exited during startup ({server.returncode}); see {log.name}')
                if time.monotonic() - start > 600:
                    raise TimeoutError('Server readiness exceeded 600 seconds')
                if not owns_listener(server.pid, port):
                    time.sleep(1)
                    continue
                try:
                    req_health = urllib.request.Request(f'http://127.0.0.1:{port}/health')
                    with opener.open(req_health, timeout=2) as response:
                        health_ok = response.status == 200
                    if health_ok:
                        req_models = urllib.request.Request(f'http://127.0.0.1:{port}/v1/models')
                        with opener.open(req_models, timeout=2) as response:
                            models_data = json.load(response)
                            model_ids = [m.get('id') for m in models_data.get('data', [])]
                            if 'gemma-4-31b-qat' in model_ids and server.poll() is None:
                                break
                except (OSError, ValueError, urllib.error.URLError):
                    pass
                time.sleep(1)
            print(f'Server ready in {time.monotonic() - start:.2f}s; pid={server.pid}', flush=True)
            os.environ['GEMMA_SERVER_LOG'] = log.name
            os.environ['GEMMA_SERVER_PID'] = str(server.pid)
            os.environ['GEMMA_STARTUP_SECONDS'] = str(time.monotonic() - start)
            if command:
                client = subprocess.Popen(command, start_new_session=True)
                deadline = time.monotonic() + 3600
                while client.poll() is None:
                    if server.poll() is not None:
                        raise RuntimeError('Server exited while client was running')
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Client command exceeded 3600 seconds')
                    time.sleep(0.5)
                return client.returncode if client.returncode >= 0 else 128 - client.returncode
            code = server.wait()
            return code if code else 1
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        stop(client)
        stop(server)
        print('Owned server/client processes stopped.', flush=True)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        sys.exit(f'Gemma QAT: {exc}')
