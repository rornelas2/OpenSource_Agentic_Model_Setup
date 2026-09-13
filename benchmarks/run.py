#!/usr/bin/env python3
"""Single entry point for the Pinnacles benchmark campaign.

Coordinates catalog inspection, evaluator preparation, cluster preflight,
model generation, evaluator grading, and offline reporting.
"""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Any

from benchmarks.contracts import ContractError, canonical_json_bytes, load_all_suites, load_profiles
from benchmarks.planning import build_plan, ensure_private_directory
from benchmarks.preflight import run_preflight
from benchmarks.state import atomic_write_json


REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILES_PATH = REPO_ROOT / "benchmarks" / "profiles.json"
SUITES_DIR = REPO_ROOT / "benchmarks" / "suites"


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def print_json(value: Any) -> None:
    sys.stdout.buffer.write(canonical_json_bytes(value))


def safe_run_dir(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    forbidden = {Path("/").resolve(), Path.home().resolve(), REPO_ROOT.resolve()}
    if path in forbidden:
        raise ContractError(f"refusing broad run directory: {path}")
    return path


def load_catalogs() -> tuple[dict[str, dict[str, Any]], dict[str, tuple[dict[str, Any], list[dict[str, Any]]]]]:
    return load_profiles(PROFILES_PATH, REPO_ROOT), load_all_suites(SUITES_DIR, REPO_ROOT)


def select(
    args: argparse.Namespace,
    profiles: dict[str, dict[str, Any]],
    suites: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if args.profile not in profiles:
        raise ContractError(f"unknown profile {args.profile!r}; choose from {', '.join(sorted(profiles))}")
    if args.suite not in suites:
        raise ContractError(f"unknown suite {args.suite!r}; choose from {', '.join(sorted(suites))}")
    suite, tasks = suites[args.suite]
    return profiles[args.profile], suite, tasks


def immutable_write(path: Path, value: Any) -> None:
    payload = canonical_json_bytes(value)
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise ContractError(f"immutable artifact already exists with different content: {path}")
    atomic_write_json(path, value)


def command_generate(args: argparse.Namespace) -> int:
    profiles, suites = load_catalogs()
    profile, suite, tasks = select(args, profiles, suites)
    run_dir = safe_run_dir(args.run_dir)
    if args.dry_run:
        plan = build_plan(
            repo_root=REPO_ROOT,
            profiles_path=PROFILES_PATH,
            profile=profile,
            suite=suite,
            tasks=tasks,
            run_dir=run_dir,
            pilot=args.pilot,
            resume=args.resume,
        )
        print_json(plan)
        return 0

    if not args.pilot and suite["protocol_status"] != "frozen":
        raise ContractError("full generation is blocked until the suite protocol passes its pilot and is frozen")

    from benchmarks.execution import execute_generation

    summary = execute_generation(
        args=args,
        profile=profile,
        suite=suite,
        tasks=tasks,
        run_dir=run_dir,
        repo_root=REPO_ROOT,
        profiles_path=PROFILES_PATH,
    )
    print_json(summary)
    return 0


def command_prepare(args: argparse.Namespace) -> int:
    profiles, suites = load_catalogs()
    manifest_path = Path(args.manifest).expanduser().resolve()
    from benchmarks.preparation import execute_prepare

    manifest = execute_prepare(args.suite, manifest_path, suites, REPO_ROOT)
    print_json(manifest)
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    profiles, suites = load_catalogs()
    profile, suite, tasks = select(args, profiles, suites)
    run_dir = safe_run_dir(args.run_dir)
    from benchmarks.execution import execute_evaluate

    result = execute_evaluate(args, profile, suite, tasks, run_dir)
    print_json(result)
    return 0


def command_report(args: argparse.Namespace) -> int:
    campaign_path = Path(args.campaign).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    from benchmarks.reporting import generate_report

    summary = generate_report(campaign_path, output_dir, REPO_ROOT)
    print_json(summary)
    return 0


def command_preflight(args: argparse.Namespace) -> int:
    profiles, suites = load_catalogs()
    profile, suite, tasks = select(args, profiles, suites)
    run_dir = safe_run_dir(args.run_dir)
    ensure_private_directory(run_dir)
    report = run_preflight(profile, suite)
    report["checked_utc"] = utc_now()
    report["scheduled_task_count"] = len(tasks)
    report["protocol_status"] = suite["protocol_status"]
    immutable_write(run_dir / "preflight.json", report)
    print_json(report)
    return 0


def command_catalog(_: argparse.Namespace) -> int:
    profiles, suites = load_catalogs()
    print_json({
        "profiles": sorted(profiles),
        "suites": {
            suite_id: {
                "task_count": len(tasks),
                "pilot_count": len(suite["pilot_task_ids"]),
                "protocol_status": suite["protocol_status"],
            }
            for suite_id, (suite, tasks) in sorted(suites.items())
        },
    })
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("catalog", help="validate and list frozen profiles and suite candidates")

    prepare = subparsers.add_parser("prepare", help="stage pinned evaluator and dataset artifacts")
    prepare.add_argument("--suite", required=True)
    prepare.add_argument("--manifest", required=True)

    for name in ("preflight", "generate", "evaluate"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--profile", required=True)
        subparser.add_argument("--suite", required=True)
        subparser.add_argument("--run-dir", required=True)
        if name == "generate":
            subparser.add_argument("--dry-run", action="store_true")
            subparser.add_argument("--pilot", action="store_true")
            subparser.add_argument("--resume", action="store_true")

    report = subparsers.add_parser("report", help="aggregate saved observations without inference")
    report.add_argument("--campaign", required=True)
    report.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "catalog":
            return command_catalog(args)
        if args.command == "prepare":
            return command_prepare(args)
        if args.command == "preflight":
            return command_preflight(args)
        if args.command == "generate":
            return command_generate(args)
        if args.command == "evaluate":
            return command_evaluate(args)
        if args.command == "report":
            return command_report(args)
        raise ContractError(f"unknown command: {args.command}")
    except (ContractError, OSError) as exc:
        print(f"benchmark: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
