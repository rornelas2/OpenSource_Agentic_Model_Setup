#!/usr/bin/env python3
"""Resource-bounded entry point for the pinned official SWE-bench evaluator.

This wrapper does not grade anything itself. It injects Docker resource and
network limits into container creation, then calls the installed upstream
``swebench.harness.run_evaluation.main`` implementation.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--predictions-path", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeout", required=True, type=int)
    parser.add_argument("--cpus", required=True, type=int)
    parser.add_argument("--memory-gib", required=True, type=int)
    parser.add_argument("--pids-limit", required=True, type=int)
    args = parser.parse_args()
    if min(args.timeout, args.cpus, args.memory_gib, args.pids_limit) <= 0:
        parser.error("all resource limits must be positive")

    from docker.models.containers import ContainerCollection
    from swebench.harness.run_evaluation import main as official_main

    original_create = ContainerCollection.create

    def bounded_create(self, image, command=None, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["nano_cpus"] = args.cpus * 1_000_000_000
        kwargs["mem_limit"] = f"{args.memory_gib}g"
        kwargs["pids_limit"] = args.pids_limit
        kwargs["network_disabled"] = True
        return original_create(self, image, command, **kwargs)

    ContainerCollection.create = bounded_create
    official_main(
        dataset_name=args.dataset_name,
        split=args.split,
        instance_ids=[args.instance_id],
        predictions_path=args.predictions_path,
        max_workers=1,
        open_file_limit=4096,
        run_id=args.run_id,
        timeout=args.timeout,
        rewrite_reports=False,
        modal=False,
        task_repo=None,
    )


if __name__ == "__main__":
    main()
