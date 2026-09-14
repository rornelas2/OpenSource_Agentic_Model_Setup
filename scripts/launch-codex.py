#!/usr/bin/env python3
"""Launch pinned Codex against an existing tutorial endpoint, from the project cwd.

Profile settings come from the same JSON files OpenCode uses. No server is
started or stopped. Endpoint inspection reads at most 1 MiB with a 10s socket timeout;
profile/model processing is linear in the returned catalog size.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request


CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
VERSION = "codex-cli 0.154.0"
MAX_CATALOG_BYTES = 1024 * 1024


def load_profile(name):
    available = {p.stem.removeprefix("opencode-"): p
                 for p in CONFIG_DIR.glob("opencode-*.json")}
    if name not in available:
        raise ValueError("Unknown profile. Choose: " + ", ".join(sorted(available)))
    config = json.loads(available[name].read_text())
    model = config["model"].removeprefix("pinnacles/")
    provider = config["provider"]["pinnacles"]
    context = provider["models"][model]["limit"]["context"]
    if type(context) is not int or context < 8192:
        raise ValueError("Profile context must be an integer of at least 8192 tokens.")
    return model, context, provider["options"]["baseURL"]


def validate_url(url):
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path.rstrip("/") != "/v1"):
        raise ValueError("Use a local http://127.0.0.1:PORT/v1 endpoint on the model's compute node.")
    # Accessing port also rejects malformed/out-of-range port strings.
    if parsed.port is not None and parsed.port < 1:
        raise ValueError("Endpoint port must be positive.")
    return url.rstrip("/")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("The local model endpoint must not redirect requests.")


def inspect_endpoint(url, model, context):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(url + "/models", timeout=10) as response:
        raw = response.read(MAX_CATALOG_BYTES + 1)
    if len(raw) > MAX_CATALOG_BYTES:
        raise ValueError("Model catalog exceeds 1 MiB.")
    catalog = json.loads(raw)
    if not isinstance(catalog, dict) or not isinstance(catalog.get("data"), list):
        raise ValueError("Endpoint returned an invalid model catalog.")
    matches = [item for item in catalog["data"]
               if isinstance(item, dict) and item.get("id") == model]
    if len(matches) != 1:
        raise ValueError("The selected model is not uniquely advertised by this server: " + model)
    item = matches[0]
    meta = item.get("meta") or {}
    if not isinstance(meta, dict):
        raise ValueError("Invalid server model metadata.")
    # llama.cpp reports n_ctx; vLLM can report max_model_len. Never raise a
    # profile's limit just because the server advertises a larger capacity.
    for limit in (item.get("max_model_len"), meta.get("n_ctx")):
        if limit is not None:
            if type(limit) is not int or limit < 8192:
                raise ValueError("Server context must be an integer of at least 8192 tokens.")
            context = min(context, limit)
    return context


def codex_command(binary, model, context, url, extra):
    settings = {
        "model": model,
        "model_provider": "pinnacles",
        "model_context_window": context,
        "model_auto_compact_token_limit": context * 3 // 4,
        "approval_policy": "on-request",
        "sandbox_mode": "workspace-write",
        "web_search": "disabled",
        "model_providers.pinnacles.name": "Pinnacles local model",
        "model_providers.pinnacles.base_url": url,
        "model_providers.pinnacles.wire_api": "responses",
        "model_providers.pinnacles.requires_openai_auth": False,
        "model_providers.pinnacles.request_max_retries": 0,
        "model_providers.pinnacles.stream_max_retries": 0,
        "model_providers.pinnacles.stream_idle_timeout_ms": 180000,
    }
    command = [str(binary)]
    for key, value in settings.items():
        command.extend(["-c", key + "=" + json.dumps(value)])
    return command + extra


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List shared model profiles and exit")
    parser.add_argument("--base-url", help="Local endpoint override, e.g. http://127.0.0.1:8001/v1")
    parser.add_argument("profile", nargs="?", help="Profile suffix, e.g. muse-gguf-128k")
    parser.add_argument("codex_args", nargs=argparse.REMAINDER, help="Codex arguments after --")
    args = parser.parse_args()
    if args.list:
        for path in sorted(CONFIG_DIR.glob("opencode-*.json")):
            name = path.stem.removeprefix("opencode-")
            model, context, _ = load_profile(name)
            print(f"{name:24} {model:30} {context:7} tokens")
        return
    if not args.profile:
        parser.error("Choose a profile or use --list.")
    if not os.environ.get("SLURM_JOB_ID") or os.uname().nodename.split(".")[0].startswith("rclogin"):
        raise ValueError("Run inside the model's Slurm compute shell, after sourcing its activation script.")
    model, context, url = load_profile(args.profile)
    url = validate_url(args.base_url or url)
    root = Path("/data") / os.environ["USER"] / "pinnacles-agents"
    binary = root / "tools/codex-v0.154.0/bin/codex"
    if not binary.is_file():
        raise ValueError("Codex is not installed. Run scripts/setup-codex.sh in a CPU allocation.")
    # Each profile has separate persistent sessions and trust settings. Nothing
    # is written to the user's ordinary ~/.codex or existing OpenCode configs.
    state = root / "codex" / args.profile
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    env = dict(os.environ, CODEX_HOME=str(state))
    version = subprocess.run([str(binary), "--version"], env=env, check=True,
                             capture_output=True, text=True, timeout=15).stdout.strip()
    if version != VERSION:
        raise ValueError("Expected " + VERSION + "; reinstall with scripts/setup-codex.sh.")
    context = inspect_endpoint(url, model, context)
    extra = args.codex_args
    if extra[:1] == ["--"]:
        extra = extra[1:]
    print(f"Codex → {model} at {url}; context {context}; project {Path.cwd()}",
          file=sys.stderr, flush=True)
    os.execve(binary, codex_command(binary, model, context, url, extra), env)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, urllib.error.URLError, subprocess.SubprocessError) as exc:
        print(f"Cannot launch Codex: {exc}", file=sys.stderr)
        sys.exit(1)
