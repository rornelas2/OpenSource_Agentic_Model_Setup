#!/usr/bin/env python3
"""Own one server and optional client command inside a bounded Slurm job.

Usage: python scripts/run-muse-gguf.py [-- command argument ...]
Without a command, serve until the allocation or an interrupt ends the process.
Logs remain local. Readiness requires a listening socket owned by our child.
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
    """Linux /proc check prevents health probes from accepting another server."""
    try:
        sockets = set()
        for entry in Path(f'/proc/{pid}/fd').iterdir():
            try:
                sockets.add(entry.readlink().name)
            except FileNotFoundError:
                continue  # The child may close a descriptor during inspection.
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
    # Kill only process groups we created. Include descendants even if their
    # parent exited, so tool/client failures do not orphan a subprocess.
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
    port_text = os.environ.get('MUSE_PORT', '8000')
    if not port_text.isascii() or not port_text.isdecimal() or not 1024 <= int(port_text) <= 65535:
        raise ValueError('MUSE_PORT must be in 1024..65535')
    port = int(port_text)
    repo = Path(__file__).resolve().parent.parent
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    directory = repo / '.local' / f'muse-gguf-{stamp}-{os.getpid()}'
    directory.mkdir(parents=True, mode=0o700)
    print(f'Server log: {directory / "server.log"}', flush=True)
    server = client = None
    start = time.monotonic()
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with (directory / 'server.log').open('w') as log:
            server = subprocess.Popen(['bash', str(repo / 'scripts/serve-muse-gguf.sh')],
                                      stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True)
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f'Server exited during startup ({server.returncode}); see {log.name}')
                if time.monotonic() - start > 600:
                    raise TimeoutError('Server readiness exceeded 600 seconds')
                if owns_listener(server.pid, port):
                    try:
                        with opener.open(f'http://127.0.0.1:{port}/health', timeout=2) as response:
                            ready = response.status == 200 and json.load(response).get('status') == 'ok'
                        if ready and server.poll() is None:
                            break
                    except (OSError, ValueError, urllib.error.URLError):
                        pass
                time.sleep(1)
            print(f'Server ready in {time.monotonic() - start:.2f}s; pid={server.pid}', flush=True)
            os.environ['MUSE_SERVER_LOG'] = log.name
            os.environ['MUSE_SERVER_PID'] = str(server.pid)
            os.environ['MUSE_STARTUP_SECONDS'] = str(time.monotonic() - start)
            if command:
                client = subprocess.Popen(command, start_new_session=True)
                deadline = time.monotonic() + 2400
                while client.poll() is None:
                    if server.poll() is not None:
                        raise RuntimeError('Server exited while client was running')
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Client command exceeded 2400 seconds')
                    time.sleep(0.5)
                return client.returncode if client.returncode >= 0 else 128 - client.returncode
            code = server.wait()
            return code if code else 1  # Unexpected server exit is a failed job.
    finally:
        # Ignore repeated signals during bounded cleanup.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        stop(client)
        stop(server)
        print('Owned server/client processes stopped.', flush=True)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        sys.exit(f'Muse GGUF: {exc}')
