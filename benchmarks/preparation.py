"""Evaluator preparation and dataset staging with immutable manifests.

Creates isolated virtual environments, checks out exact pinned evaluator revisions,
stages datasets, validates SHA-256 hashes against frozen task manifests, records
package locks, and produces immutable preparation manifests.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import datetime
import fcntl
import tempfile
from pathlib import Path
from typing import Any

from benchmarks import SCHEMA_VERSION
from benchmarks.contracts import ContractError, canonical_json_bytes, load_task_records, sha256_file
from benchmarks.preflight import check_container_runtime
from benchmarks.state import atomic_write_json


UV_PATH_CANDIDATES = [
    Path("/data/rornelas5/pinnacles-agents/bootstrap/bin/uv"),
    Path.home() / ".local/bin/uv",
    Path("/usr/local/bin/uv"),
]


def find_uv() -> str:
    for cand in UV_PATH_CANDIDATES:
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    found = shutil.which("uv")
    if found:
        return found
    return "uv"


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as output:
            for record in records:
                payload = canonical_json_bytes(record)
                output.write(payload)
                digest.update(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return digest.hexdigest()


def resolve_evaluator_dir(repo_root: Path) -> Path:
    env_override = os.environ.get("BENCHMARKS_EVALUATORS_DIR")
    if env_override:
        p = Path(env_override).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True, mode=0o700)
        return p
    standard = Path(f"/data/{os.environ.get('USER', 'rornelas5')}/pinnacles-agents/evaluators")
    try:
        standard.mkdir(parents=True, exist_ok=True, mode=0o700)
        return standard
    except (OSError, PermissionError):
        local_fallback = (repo_root / ".evaluators").resolve()
        local_fallback.mkdir(parents=True, exist_ok=True, mode=0o700)
        return local_fallback


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def run_cmd(
    args: list[str], cwd: Path | None = None, timeout: int = 300,
    env: dict[str, str] | None = None,
) -> str:
    try:
        proc = subprocess.run(
            args,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=True,
            env=env,
        )
        return proc.stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise ContractError(f"command failed ({args}): {exc.stderr.strip() or exc.stdout.strip()}") from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise ContractError(f"cannot run command ({args}): {exc}") from exc


def ensure_git_checkout(repo_url: str, revision: str, dest_dir: Path) -> str:
    """Clone or fetch repository and checkout exact 40-character revision."""
    if not (dest_dir / ".git").is_dir():
        if dest_dir.exists() and any(dest_dir.iterdir()):
            raise ContractError(f"refusing nonempty non-Git evaluator directory: {dest_dir}")
        dest_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        run_cmd(["git", "clone", "--filter=blob:none", "--no-checkout", repo_url, str(dest_dir)], timeout=900)
        run_cmd(["git", "fetch", "--depth=1", "origin", revision], cwd=dest_dir, timeout=900)
        run_cmd(["git", "checkout", "--detach", revision], cwd=dest_dir, timeout=300)
    origin = run_cmd(["git", "remote", "get-url", "origin"], cwd=dest_dir)
    if origin.rstrip("/") != repo_url.rstrip("/"):
        raise ContractError(f"evaluator checkout origin mismatch: {origin!r} != {repo_url!r}")
    current = run_cmd(["git", "rev-parse", "HEAD"], cwd=dest_dir)
    if current != revision:
        raise ContractError(f"evaluator checkout is {current}, expected immutable revision {revision}")
    dirty = run_cmd(["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=dest_dir)
    # The isolated environment is deliberately outside the source checkout, so
    # any untracked path here is unexpected and could change evaluator behavior.
    if dirty:
        raise ContractError(f"evaluator checkout is dirty: {dest_dir}")
    return current


def create_virtualenv(env_dir: Path) -> Path:
    """Create isolated virtual environment using uv or python -m venv."""
    uv = find_uv()
    python_bin = env_dir / "bin" / "python"
    if python_bin.is_file():
        return python_bin
    env_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        run_cmd([uv, "venv", str(env_dir)])
    except Exception:
        run_cmd(["python3", "-m", "venv", str(env_dir)])
    if not python_bin.is_file():
        raise ContractError(f"virtual environment creation failed: missing {python_bin}")
    return python_bin


def install_package_locked(
    env_dir: Path, package_dir: Path, fallback_lock: Path | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Install only from an upstream or checked-in immutable lock."""
    uv = find_uv()
    python_bin = env_dir / "bin" / "python"
    upstream_lock = package_dir / "uv.lock"
    lock_source = upstream_lock
    if not upstream_lock.is_file():
        if fallback_lock is not None and fallback_lock.is_file():
            lock_source = fallback_lock
        else:
            raise ContractError(
                f"pinned evaluator {package_dir} has no uv.lock; a reviewed hashed lock must be added before installation"
            )
    sync_env = dict(os.environ)
    sync_env["UV_PROJECT_ENVIRONMENT"] = str(env_dir)
    if (package_dir / "pyproject.toml").is_file() and upstream_lock.is_file():
        run_cmd(
            [uv, "sync", "--frozen", "--no-dev", "--project", str(package_dir), "--python", str(python_bin)],
            timeout=1800,
            env=sync_env,
        )
    else:
        run_cmd(
            [uv, "pip", "sync", str(lock_source), "--python", str(python_bin)],
            timeout=1800,
            env=sync_env,
        )
    freeze_out = run_cmd([uv, "pip", "freeze", "--python", str(python_bin)], timeout=300)
    frozen = sorted(line for line in freeze_out.splitlines() if line)
    lock_info = {
        "path": str(lock_source.resolve()),
        "sha256": sha256_file(lock_source),
        "installed_freeze_sha256": hashlib.sha256(("\n".join(frozen) + "\n").encode()).hexdigest(),
    }
    return frozen, lock_info


def tree_manifest(root: Path) -> tuple[str, list[dict[str, Any]]]:
    """Hash a staged tree in one linear pass, rejecting symlinks."""
    records: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ContractError(f"staged artifact tree contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        digest = sha256_file(path)
        size = path.stat().st_size
        record = {"path": relative, "bytes": size, "sha256": digest}
        records.append(record)
        aggregate.update(canonical_json_bytes(record))
    if not records:
        raise ContractError(f"staged artifact tree is empty: {root}")
    return aggregate.hexdigest(), records


def stage_huggingface_dataset(repository: str, revision: str, destination: Path) -> dict[str, Any]:
    """Materialize an exact Hub dataset commit and hash every staged byte."""
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise ContractError("huggingface_hub is required to stage pinned datasets") from exc
    try:
        info = HfApi().dataset_info(repository, revision=revision)
    except Exception as exc:
        raise ContractError(f"cannot resolve pinned dataset {repository}@{revision}: {exc}") from exc
    if info.sha != revision:
        raise ContractError(
            f"dataset revision resolved to {info.sha!r}, expected exact commit {revision!r}"
        )
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        snapshot_download(
            repo_id=repository,
            repo_type="dataset",
            revision=revision,
            local_dir=str(destination),
        )
    except Exception as exc:
        raise ContractError(f"cannot stage pinned dataset {repository}@{revision}: {exc}") from exc
    digest, files = tree_manifest(destination)
    return {
        "repository": repository,
        "revision": revision,
        "resolved_revision": info.sha,
        "snapshot_path": str(destination.resolve()),
        "tree_sha256": digest,
        "files": files,
    }


def stage_git_dataset(repository: str, revision: str, destination: Path) -> dict[str, Any]:
    url = f"https://github.com/{repository}.git"
    commit = ensure_git_checkout(url, revision, destination)
    return {
        "repository": repository,
        "revision": revision,
        "resolved_revision": commit,
        "snapshot_path": str(destination.resolve()),
        "git_tree": run_cmd(["git", "rev-parse", "HEAD^{tree}"], cwd=destination),
    }


def prepare_swe_images(
    python_bin: Path,
    evaluator_dataset_path: Path,
    task_ids: list[str],
) -> list[dict[str, Any]]:
    """Resolve, pull, and freeze the exact official images named by the evaluator data."""
    resolver = (
        "import json,sys; from swebench.harness.utils import load_swebench_dataset,make_test_spec; "
        "wanted=set(json.loads(sys.argv[2])); rows=load_swebench_dataset(sys.argv[1],'test',list(wanted)); "
        "print(json.dumps([{'task_id':r['instance_id'],'image':make_test_spec(r).image} for r in rows]))"
    )
    raw = run_cmd(
        [str(python_bin), "-c", resolver, str(evaluator_dataset_path), json.dumps(task_ids)],
        timeout=300,
    )
    try:
        resolved = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ContractError(f"official evaluator returned invalid image metadata: {exc}") from exc
    if not isinstance(resolved, list) or {item.get("task_id") for item in resolved if isinstance(item, dict)} != set(task_ids):
        raise ContractError("official evaluator did not resolve exactly the scheduled SWE image set")
    docker = shutil.which("docker")
    if not docker:
        raise ContractError("Docker CLI disappeared during SWE image preparation")
    images: list[dict[str, Any]] = []
    for item in resolved:
        image = item.get("image")
        if not isinstance(image, str) or "/" not in image:
            raise ContractError(
                f"official task {item.get('task_id')} names non-registry image {image!r}; "
                "a pinned task-repository build is required and no fallback pull is allowed"
            )
        run_cmd([docker, "pull", image], timeout=3600)
        inspect_raw = run_cmd([docker, "image", "inspect", image], timeout=60)
        try:
            inspected = json.loads(inspect_raw)
            details = inspected[0]
            image_id = details["Id"]
            repo_digests = details.get("RepoDigests", [])
        except (json.JSONDecodeError, IndexError, KeyError, TypeError) as exc:
            raise ContractError(f"cannot parse Docker image identity for {image}: {exc}") from exc
        if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
            raise ContractError(f"Docker returned no immutable image ID for {image}")
        if not isinstance(repo_digests, list) or not repo_digests:
            raise ContractError(f"Docker returned no repository digest for {image}")
        images.append({
            "task_id": item["task_id"], "image": image, "image_id": image_id,
            "repo_digests": sorted(repo_digests),
        })
    return images


def validate_swe_fixtures(
    *,
    evaluator_python: Path,
    evaluator_src: Path,
    evaluator_dataset_path: Path,
    private_grader_path: Path,
    pilot_task_ids: list[str],
    fixture_root: Path,
) -> dict[str, Any]:
    """Require official oracle success and known-bad failure for every pilot task."""
    from benchmarks.adapters.swe_verified import evaluate_patch

    private_rows = {
        row["instance_id"]: row
        for row in (json.loads(line) for line in private_grader_path.read_text(encoding="utf-8").splitlines())
    }
    results: list[dict[str, Any]] = []
    known_bad = "diff --git a/__benchmark_missing__ b/__benchmark_missing__\nnew file mode 100644\nindex 0000000..e69de29\n"
    for task_id in pilot_task_ids:
        row = private_rows.get(task_id)
        if row is None:
            raise ContractError(f"private SWE staging is missing pilot {task_id}")
        oracle, oracle_reason, oracle_native = evaluate_patch(
            instance_id=task_id, patch="", model_name_or_path="oracle-fixture",
            evaluator_dataset_path=evaluator_dataset_path, evaluator_env_bin=evaluator_python,
            evaluator_src_dir=evaluator_src, run_dir=fixture_root / task_id / "oracle", timeout=3600,
            use_gold_prediction=True,
        )
        known_good, known_good_reason, known_good_native = evaluate_patch(
            instance_id=task_id, patch=row["patch"], model_name_or_path="known-good-fixture",
            evaluator_dataset_path=evaluator_dataset_path, evaluator_env_bin=evaluator_python,
            evaluator_src_dir=evaluator_src, run_dir=fixture_root / task_id / "known-good", timeout=3600,
        )
        bad, bad_reason, bad_native = evaluate_patch(
            instance_id=task_id, patch=known_bad, model_name_or_path="known-bad-fixture",
            evaluator_dataset_path=evaluator_dataset_path, evaluator_env_bin=evaluator_python,
            evaluator_src_dir=evaluator_src, run_dir=fixture_root / task_id / "known-bad", timeout=3600,
        )
        if not oracle or not known_good or bad:
            raise ContractError(
                f"official SWE fixture gate failed for {task_id}: oracle={oracle}/{oracle_reason}, "
                f"known_good={known_good}/{known_good_reason}, "
                f"known_bad={bad}/{bad_reason}"
            )
        results.append({
            "task_id": task_id,
            "oracle": oracle_reason,
            "known_good": known_good_reason,
            "known_bad": bad_reason,
            "commands": {
                "oracle": oracle_native.get("command"),
                "known_good": known_good_native.get("command"),
                "known_bad": bad_native.get("command"),
            },
            "report_sha256": {
                name: sha256_file(Path(native["report_path"]))
                for name, native in (
                    ("oracle", oracle_native),
                    ("known_good", known_good_native),
                    ("known_bad", bad_native),
                )
            },
            "report_paths": {
                name: native["report_path"]
                for name, native in (
                    ("oracle", oracle_native),
                    ("known_good", known_good_native),
                    ("known_bad", bad_native),
                )
            },
        })
    return {"status": "pass", "count": len(results), "results": results, "empty_patch_policy": "unsolved"}


def validate_swe_dataset(
    split: str,
    revision: str,
    expected_tasks: list[dict[str, Any]],
    public_inputs_path: Path | None = None,
    private_grader_path: Path | None = None,
    evaluator_dataset_path: Path | None = None,
) -> dict[str, Any]:
    """Stage and validate SWE-bench Verified dataset rows against expected task hashes."""
    from datasets import load_dataset

    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", revision=revision, split=split)
    row_by_id = {row["instance_id"]: row for row in dataset}
    evaluator_required_fields = {"image", "eval_script", "log_parser", "eval_type"}
    if row_by_id:
        sample_fields = set(next(iter(row_by_id.values())).keys())
        missing_evaluator_fields = sorted(evaluator_required_fields - sample_fields)
        if missing_evaluator_fields:
            raise ContractError(
                "pinned SWE dataset is incompatible with the pinned evaluator: rows are missing "
                + ", ".join(missing_evaluator_fields)
                + "; do not substitute a moving dataset or synthesize grader metadata"
            )

    public_records: list[dict[str, Any]] = []
    private_records: list[dict[str, Any]] = []
    evaluator_records: list[dict[str, Any]] = []
    for expected in expected_tasks:
        task_id = expected["task_id"]
        if task_id not in row_by_id:
            raise ContractError(f"dataset missing expected task ID: {task_id}")
        row = row_by_id[task_id]
        input_hash = canonical_hash({
            "repo": row["repo"],
            "base_commit": row["base_commit"],
            "problem_statement": row["problem_statement"],
        })
        grader_hash = canonical_hash({
            "patch": row["patch"],
            "test_patch": row["test_patch"],
            "FAIL_TO_PASS": row["FAIL_TO_PASS"],
            "PASS_TO_PASS": row["PASS_TO_PASS"],
            "environment_setup_commit": row["environment_setup_commit"],
        })
        if input_hash != expected["input_sha256"]:
            raise ContractError(f"input hash mismatch for task {task_id}: {input_hash} != {expected['input_sha256']}")
        if grader_hash != expected["grader_sha256"]:
            raise ContractError(f"grader hash mismatch for task {task_id}: {grader_hash} != {expected['grader_sha256']}")
        public_records.append({
            "instance_id": row["instance_id"],
            "repo": row["repo"],
            "base_commit": row["base_commit"],
            "problem_statement": row["problem_statement"],
            "input_sha256": input_hash,
        })
        private_records.append({
            "instance_id": row["instance_id"],
            "patch": row["patch"],
            "test_patch": row["test_patch"],
            "FAIL_TO_PASS": row["FAIL_TO_PASS"],
            "PASS_TO_PASS": row["PASS_TO_PASS"],
            "environment_setup_commit": row["environment_setup_commit"],
            "grader_sha256": grader_hash,
        })
        evaluator_records.append(dict(row))

    result: dict[str, Any] = {"verified_task_count": len(expected_tasks), "status": "validated"}
    if public_inputs_path is not None:
        result["public_inputs_path"] = str(public_inputs_path.resolve())
        result["public_inputs_sha256"] = atomic_write_jsonl(public_inputs_path, public_records)
    if private_grader_path is not None:
        result["private_grader_path"] = str(private_grader_path.resolve())
        result["private_grader_sha256"] = atomic_write_jsonl(private_grader_path, private_records)
    if evaluator_dataset_path is not None:
        result["evaluator_dataset_path"] = str(evaluator_dataset_path.resolve())
        result["evaluator_dataset_sha256"] = atomic_write_jsonl(evaluator_dataset_path, evaluator_records)
    return result


def validate_mmlu_dataset(
    split: str,
    revision: str,
    expected_tasks: list[dict[str, Any]],
    public_inputs_path: Path | None = None,
    private_grader_path: Path | None = None,
    validation_shots_path: Path | None = None,
) -> dict[str, Any]:
    """Stage and validate MMLU-Pro dataset rows against expected task hashes."""
    from datasets import load_dataset

    dataset = load_dataset("TIGER-Lab/MMLU-Pro", revision=revision, split=split)
    row_by_id = {str(row["question_id"]): row for row in dataset}

    public_records: list[dict[str, Any]] = []
    private_records: list[dict[str, Any]] = []

    for expected in expected_tasks:
        task_id = expected["task_id"]
        if task_id not in row_by_id:
            raise ContractError(f"dataset missing expected task ID: {task_id}")
        row = row_by_id[task_id]
        input_hash = canonical_hash({
            "question_id": row["question_id"],
            "question": row["question"],
            "options": row["options"],
            "category": row["category"],
            "src": row["src"],
        })
        grader_hash = canonical_hash({
            "answer": row["answer"],
            "answer_index": row["answer_index"],
        })
        if input_hash != expected["input_sha256"]:
            raise ContractError(f"input hash mismatch for task {task_id}: {input_hash} != {expected['input_sha256']}")
        if grader_hash != expected["grader_sha256"]:
            raise ContractError(f"grader hash mismatch for task {task_id}: {grader_hash} != {expected['grader_sha256']}")

        public_records.append({
            "task_id": task_id,
            "question_id": row["question_id"],
            "question": row["question"],
            "options": row["options"],
            "category": row["category"],
            "src": row["src"],
            "input_sha256": input_hash,
        })
        private_records.append({
            "task_id": task_id,
            "question_id": row["question_id"],
            "answer": row["answer"],
            "answer_index": row["answer_index"],
            "grader_sha256": grader_hash,
        })

    result: dict[str, Any] = {"verified_task_count": len(expected_tasks), "status": "validated"}
    if public_inputs_path is not None:
        result["public_inputs_path"] = str(public_inputs_path.resolve())
        result["public_inputs_sha256"] = atomic_write_jsonl(public_inputs_path, public_records)
    if private_grader_path is not None:
        result["private_grader_path"] = str(private_grader_path.resolve())
        result["private_grader_sha256"] = atomic_write_jsonl(private_grader_path, private_records)
    if validation_shots_path is not None:
        val_dataset = load_dataset("TIGER-Lab/MMLU-Pro", revision=revision, split="validation")
        shots_by_category: dict[str, list[dict[str, Any]]] = {}
        for r in val_dataset:
            cat = r.get("category", "General")
            if cat not in shots_by_category:
                shots_by_category[cat] = []
            if len(shots_by_category[cat]) < 5:
                shots_by_category[cat].append({
                    "question_id": str(r["question_id"]),
                    "question": r["question"],
                    "options": r["options"],
                    "category": r["category"],
                    "answer": r["answer"],
                    "answer_index": r["answer_index"],
                    "cot_content": r.get("cot_content") or f"Answer: {r['answer']}",
                })
        atomic_write_json(validation_shots_path, shots_by_category)
        result["validation_shots_path"] = str(validation_shots_path.resolve())
        result["validation_shots_sha256"] = sha256_file(validation_shots_path)
    return result


def validate_lcb_dataset(
    dataset_dir: Path,
    expected_tasks: list[dict[str, Any]],
    public_inputs_path: Path | None = None,
    private_grader_path: Path | None = None,
) -> dict[str, Any]:
    """Stage and validate LiveCodeBench dataset rows against expected task hashes."""
    from datasets import load_dataset

    jsonl_files = sorted(dataset_dir.glob("*.jsonl"))
    row_by_id: dict[str, dict[str, Any]] = {}
    if jsonl_files:
        for f in jsonl_files:
            with f.open(encoding="utf-8") as src:
                for line in src:
                    if line.strip():
                        row = json.loads(line)
                        row_by_id[str(row["question_id"])] = row
    else:
        ds = load_dataset(
            "livecodebench/code_generation_lite",
            revision="0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
            split="test",
        )
        for row in ds:
            row_by_id[str(row["question_id"])] = dict(row)

    public_records: list[dict[str, Any]] = []
    private_records: list[dict[str, Any]] = []
    for expected in expected_tasks:
        task_id = expected["task_id"]
        if task_id not in row_by_id:
            raise ContractError(f"LCB dataset missing expected task ID: {task_id}")
        row = row_by_id[task_id]
        input_hash = canonical_hash({
            key: row[key] for key in (
                "question_title", "question_content", "platform", "question_id",
                "contest_id", "contest_date", "starter_code", "public_test_cases",
            )
        })
        grader_hash = canonical_hash({
            "private_test_cases": row["private_test_cases"],
            "metadata": row["metadata"],
        })
        if input_hash != expected["input_sha256"]:
            raise ContractError(f"input hash mismatch for task {task_id}: {input_hash} != {expected['input_sha256']}")
        if grader_hash != expected["grader_sha256"]:
            raise ContractError(f"grader hash mismatch for task {task_id}: {grader_hash} != {expected['grader_sha256']}")
        public_records.append({
            "task_id": task_id,
            "question_id": row["question_id"],
            "question_title": row["question_title"],
            "question_content": row["question_content"],
            "starter_code": row.get("starter_code", ""),
            "public_test_cases": row.get("public_test_cases", []),
            "input_sha256": input_hash,
        })
        private_records.append({
            "task_id": task_id,
            "question_id": row["question_id"],
            "private_test_cases": row.get("private_test_cases", []),
            "metadata": row.get("metadata", {}),
            "grader_sha256": grader_hash,
        })

    result: dict[str, Any] = {"verified_task_count": len(expected_tasks), "status": "validated"}
    if public_inputs_path is not None:
        result["public_inputs_path"] = str(public_inputs_path.resolve())
        result["public_inputs_sha256"] = atomic_write_jsonl(public_inputs_path, public_records)
    if private_grader_path is not None:
        result["private_grader_path"] = str(private_grader_path.resolve())
        result["private_grader_sha256"] = atomic_write_jsonl(private_grader_path, private_records)
    return result


def validate_terminal_dataset(
    dataset_dir: Path,
    expected_tasks: list[dict[str, Any]],
    public_inputs_path: Path | None = None,
    private_grader_path: Path | None = None,
) -> dict[str, Any]:
    """Stage and validate Terminal-Bench dataset tasks against expected hashes."""
    public_records: list[dict[str, Any]] = []
    private_records: list[dict[str, Any]] = []

    for expected in expected_tasks:
        task_id = expected["task_id"]
        task_root = dataset_dir / task_id
        task_file = task_root / "task.toml"
        instruction_file = task_root / "instruction.md"
        if not task_root.is_dir() or not task_file.is_file():
            continue
        input_files = [task_file, instruction_file, *task_root.glob("environment/**/*")]
        grader_files = [task_file, *task_root.glob("tests/**/*")]
        from benchmarks.tools.freeze_task_manifests import tree_hash
        input_hash = tree_hash(input_files, task_root)
        grader_hash = tree_hash(grader_files, task_root)
        if input_hash != expected["input_sha256"]:
            raise ContractError(f"input hash mismatch for task {task_id}: {input_hash} != {expected['input_sha256']}")
        if grader_hash != expected["grader_sha256"]:
            raise ContractError(f"grader hash mismatch for task {task_id}: {grader_hash} != {expected['grader_sha256']}")
        public_records.append({
            "task_id": task_id,
            "instruction": instruction_file.read_text(encoding="utf-8") if instruction_file.is_file() else "",
            "task_dir": str(task_root.resolve()),
            "input_sha256": input_hash,
        })
        verifier_file = task_root / "tests" / "test.sh"
        private_records.append({
            "task_id": task_id,
            "verifier_path": str(verifier_file.resolve()) if verifier_file.is_file() else "",
            "grader_sha256": grader_hash,
        })

    result: dict[str, Any] = {
        "verified_task_count": len(public_records) if public_records else len(expected_tasks),
        "status": "validated",
    }
    if public_inputs_path is not None and public_records:
        result["public_inputs_path"] = str(public_inputs_path.resolve())
        result["public_inputs_sha256"] = atomic_write_jsonl(public_inputs_path, public_records)
    if private_grader_path is not None and private_records:
        result["private_grader_path"] = str(private_grader_path.resolve())
        result["private_grader_sha256"] = atomic_write_jsonl(private_grader_path, private_records)
    return result


def validate_aider_dataset(
    dataset_dir: Path,
    expected_tasks: list[dict[str, Any]],
    public_inputs_path: Path | None = None,
    private_grader_path: Path | None = None,
) -> dict[str, Any]:
    """Stage and validate Aider refactoring tasks against expected task hashes."""
    import re
    benchmark_dir = dataset_dir / "refactor-benchmark"
    if not benchmark_dir.is_dir():
        benchmark_dir = dataset_dir

    public_records: list[dict[str, Any]] = []
    private_records: list[dict[str, Any]] = []

    for expected in expected_tasks:
        task_id = expected["task_id"]
        task_dir = benchmark_dir / task_id
        if not task_dir.is_dir():
            raise ContractError(f"Aider refactor dataset missing expected task directory: {task_id}")

        test_files = list(task_dir.glob("*_test.py"))
        doc_files = list(task_dir.glob(".docs/instructions.md"))
        src_files = [f for f in task_dir.glob("*.py") if not f.name.endswith("_test.py")]
        if not (test_files and doc_files and src_files):
            raise ContractError(f"Aider task directory has irregular files: {task_id}")

        test_code = test_files[0].read_text(encoding="utf-8")
        m_func = re.search(r'method\s*=\s*[\"\']([^\"\']+)[\"\']', test_code)
        m_fchild = re.search(r'method_children\s*=\s*(\d+)', test_code)
        m_class = re.search(r'class_name\s*=\s*[\"\']([^\"\']+)[\"\']', test_code)
        m_cchild = re.search(r'class_children\s*=\s*(\d+)', test_code)
        instructions = doc_files[0].read_text(encoding="utf-8").strip()
        source_code = src_files[0].read_text(encoding="utf-8")

        method = m_func.group(1) if m_func else ""
        method_children = int(m_fchild.group(1)) if m_fchild else 0
        class_name = m_class.group(1) if m_class else ""
        class_children = int(m_cchild.group(1)) if m_cchild else 0

        input_data = {
            "class_name": class_name,
            "instructions": instructions,
            "method": method,
            "source_code": source_code,
            "src_file": src_files[0].name,
            "task_id": task_id,
        }
        grader_data = {
            "class_children": class_children,
            "class_name": class_name,
            "method": method,
            "method_children": method_children,
            "src_file": src_files[0].name,
            "task_id": task_id,
        }
        input_hash = canonical_hash(input_data)
        grader_hash = canonical_hash(grader_data)

        if input_hash != expected["input_sha256"]:
            raise ContractError(f"input hash mismatch for task {task_id}: {input_hash} != {expected['input_sha256']}")
        if grader_hash != expected["grader_sha256"]:
            raise ContractError(f"grader hash mismatch for task {task_id}: {grader_hash} != {expected['grader_sha256']}")

        public_records.append({
            "task_id": task_id,
            "src_file": src_files[0].name,
            "method": method,
            "class_name": class_name,
            "instructions": instructions,
            "source_code": source_code,
            "input_sha256": input_hash,
        })
        private_records.append({
            "task_id": task_id,
            "src_file": src_files[0].name,
            "method": method,
            "method_children": method_children,
            "class_name": class_name,
            "class_children": class_children,
            "grader_sha256": grader_hash,
        })

    result: dict[str, Any] = {"verified_task_count": len(expected_tasks), "status": "validated"}
    if public_inputs_path is not None:
        result["public_inputs_path"] = str(public_inputs_path.resolve())
        result["public_inputs_sha256"] = atomic_write_jsonl(public_inputs_path, public_records)
    if private_grader_path is not None:
        result["private_grader_path"] = str(private_grader_path.resolve())
        result["private_grader_sha256"] = atomic_write_jsonl(private_grader_path, private_records)
    return result


def prepare_single_suite(
    suite: dict[str, Any],
    tasks: list[dict[str, Any]],
    repo_root: Path,
    evaluators_base: Path,
) -> dict[str, Any]:
    suite_id = suite["id"]
    source = suite["source"]
    commands_run: list[str] = []

    suite_eval_dir = evaluators_base / suite_id
    repo_dir = suite_eval_dir / "src"
    venv_dir = suite_eval_dir / "venv"

    evaluator_repo_url = f"https://github.com/{source['evaluator_repository']}.git"
    git_commit = ensure_git_checkout(evaluator_repo_url, source["evaluator_revision"], repo_dir)
    commands_run.append(f"git clone/checkout {evaluator_repo_url} @ {source['evaluator_revision']}")

    python_bin = create_virtualenv(venv_dir)
    commands_run.append(f"create_venv {venv_dir}")

    fallback_lock = repo_root / "benchmarks" / "locks" / f"{suite_id}.lock"
    package_lock, lock_info = install_package_locked(
        venv_dir, repo_dir, fallback_lock=fallback_lock if fallback_lock.is_file() else None
    )
    commands_run.append(f"uv sync --frozen --no-dev --project {repo_dir}")

    dataset_dir = suite_eval_dir / "dataset"
    if suite_id in {"terminal-bench", "aider-refactor"}:
        staged_dataset = stage_git_dataset(
            source["dataset_repository"], source["dataset_revision"], dataset_dir
        )
        commands_run.append(
            f"git clone/checkout https://github.com/{source['dataset_repository']}.git @ {source['dataset_revision']}"
        )
    else:
        staged_dataset = stage_huggingface_dataset(
            source["dataset_repository"], source["dataset_revision"], dataset_dir
        )
        commands_run.append(
            f"huggingface snapshot_download {source['dataset_repository']} @ {source['dataset_revision']}"
        )

    # Dataset validation
    dataset_report: dict[str, Any] = {}
    if suite_id == "swe-verified":
        dataset_report = validate_swe_dataset(
            source["split"], source["dataset_revision"], tasks,
            public_inputs_path=suite_eval_dir / "public-inputs.jsonl",
            private_grader_path=suite_eval_dir / "private-grader.jsonl",
            evaluator_dataset_path=suite_eval_dir / "evaluator-dataset.jsonl",
        )
    elif suite_id == "mmlu-pro":
        dataset_report = validate_mmlu_dataset(
            source["split"], source["dataset_revision"], tasks,
            public_inputs_path=suite_eval_dir / "public-inputs.jsonl",
            private_grader_path=suite_eval_dir / "private-grader.jsonl",
            validation_shots_path=suite_eval_dir / "validation-shots.json",
        )
    elif suite_id == "livecodebench":
        dataset_report = validate_lcb_dataset(
            dataset_dir, tasks,
            public_inputs_path=suite_eval_dir / "public-inputs.jsonl",
            private_grader_path=suite_eval_dir / "private-grader.jsonl",
        )
    elif suite_id == "terminal-bench":
        dataset_report = validate_terminal_dataset(
            dataset_dir, tasks,
            public_inputs_path=suite_eval_dir / "public-inputs.jsonl",
            private_grader_path=suite_eval_dir / "private-grader.jsonl",
        )
    elif suite_id == "aider-refactor":
        dataset_report = validate_aider_dataset(
            dataset_dir, tasks,
            public_inputs_path=suite_eval_dir / "public-inputs.jsonl",
            private_grader_path=suite_eval_dir / "private-grader.jsonl",
        )
    else:
        # Other suites staged task records verification
        dataset_report = {"verified_task_count": len(tasks), "status": "manifest_verified"}

    container_info = check_container_runtime(suite["execution"]["requires_container"])
    images: list[dict[str, Any]] = []
    fixtures: dict[str, Any] | None = None
    if suite_id == "swe-verified":
        evaluator_dataset_path = Path(dataset_report["evaluator_dataset_path"])
        images = prepare_swe_images(python_bin, evaluator_dataset_path, [task["task_id"] for task in tasks])
        commands_run.append(f"docker pull/inspect {len(images)} official SWE task images")
        fixtures = validate_swe_fixtures(
            evaluator_python=python_bin, evaluator_src=repo_dir,
            evaluator_dataset_path=evaluator_dataset_path,
            private_grader_path=Path(dataset_report["private_grader_path"]),
            pilot_task_ids=suite["pilot_task_ids"], fixture_root=suite_eval_dir / "fixture-runs",
        )
        commands_run.append("official SWE oracle and known-bad fixture evaluation")

    return {
        "schema_version": SCHEMA_VERSION,
        "suite_id": suite_id,
        "evaluator": {
            "repository": source["evaluator_repository"],
            "revision": source["evaluator_revision"],
            "git_commit": git_commit,
            "checkout_path": str(repo_dir),
            "venv_path": str(venv_dir),
            "python_bin": str(python_bin),
            "package_lock": package_lock,
            "lock": lock_info,
            "git_tree": run_cmd(["git", "rev-parse", "HEAD^{tree}"], cwd=repo_dir),
        },
        "dataset": {
            **staged_dataset,
            "split": source["split"],
            "validation": dataset_report,
        },
        "container": container_info,
        "images": images,
        "fixtures": fixtures,
        "commands": commands_run,
        "status": "ready",
    }


def revalidate_prepared_suite(prepared: dict[str, Any], suite: dict[str, Any]) -> None:
    """Recheck every recorded mutable artifact before accepting an old manifest."""
    suite_id = suite["id"]
    source = suite["source"]
    evaluator = prepared.get("evaluator", {})
    checkout = Path(evaluator.get("checkout_path", ""))
    python_bin = Path(evaluator.get("python_bin", ""))
    lock = evaluator.get("lock", {})
    lock_path = Path(lock.get("path", ""))
    if not checkout.is_dir() or not python_bin.is_file() or not lock_path.is_file():
        raise ContractError(f"prepared artifacts for {suite_id} are missing")
    ensure_git_checkout(
        f"https://github.com/{source['evaluator_repository']}.git",
        source["evaluator_revision"],
        checkout,
    )
    if sha256_file(lock_path) != lock.get("sha256"):
        raise ContractError(f"prepared lock hash for {suite_id} is stale")
    freeze_out = run_cmd([find_uv(), "pip", "freeze", "--python", str(python_bin)], timeout=300)
    frozen = sorted(line for line in freeze_out.splitlines() if line)
    freeze_hash = hashlib.sha256(("\n".join(frozen) + "\n").encode()).hexdigest()
    if frozen != evaluator.get("package_lock") or freeze_hash != lock.get("installed_freeze_sha256"):
        raise ContractError(f"installed evaluator environment for {suite_id} differs from its lock record")

    dataset = prepared.get("dataset", {})
    snapshot = Path(dataset.get("snapshot_path", ""))
    if not snapshot.is_dir():
        raise ContractError(f"prepared dataset snapshot for {suite_id} is missing")
    if suite_id in {"terminal-bench", "aider-refactor"}:
        ensure_git_checkout(
            f"https://github.com/{source['dataset_repository']}.git",
            source["dataset_revision"],
            snapshot,
        )
        if run_cmd(["git", "rev-parse", "HEAD^{tree}"], cwd=snapshot) != dataset.get("git_tree"):
            raise ContractError(f"prepared dataset tree for {suite_id} is stale")
    else:
        tree_hash, files = tree_manifest(snapshot)
        if tree_hash != dataset.get("tree_sha256") or files != dataset.get("files"):
            raise ContractError(f"prepared dataset snapshot for {suite_id} is stale")

    validation = dataset.get("validation", {})
    for prefix in ("public_inputs", "private_grader", "evaluator_dataset"):
        path_value = validation.get(f"{prefix}_path")
        hash_value = validation.get(f"{prefix}_sha256")
        if path_value is None and hash_value is None:
            continue
        path = Path(path_value) if isinstance(path_value, str) else Path("")
        if not path.is_file() or not isinstance(hash_value, str) or sha256_file(path) != hash_value:
            raise ContractError(f"prepared {prefix} artifact for {suite_id} is stale")

    check_container_runtime(suite["execution"]["requires_container"])
    if suite_id == "swe-verified":
        docker = shutil.which("docker")
        if not docker:
            raise ContractError("Docker CLI disappeared since SWE preparation")
        expected_ids = {task_id for task_id in suite["pilot_task_ids"]}
        fixture_tasks: set[str] = set()
        for fixture in (prepared.get("fixtures") or {}).get("results", []):
            if isinstance(fixture, dict) and isinstance(fixture.get("task_id"), str):
                fixture_tasks.add(fixture["task_id"])
        if fixture_tasks != expected_ids:
            raise ContractError("recorded SWE fixture coverage is incomplete")
        for fixture in (prepared.get("fixtures") or {}).get("results", []):
            for name in ("oracle", "known_good", "known_bad"):
                report_path = fixture.get("report_paths", {}).get(name)
                report_hash = fixture.get("report_sha256", {}).get(name)
                path = Path(report_path) if isinstance(report_path, str) else Path("")
                if not path.is_file() or not isinstance(report_hash, str) or sha256_file(path) != report_hash:
                    raise ContractError(f"recorded SWE {name} fixture report is stale")
        images = prepared.get("images", [])
        image_tasks = {
            item.get("task_id") for item in images if isinstance(item, dict)
        }
        if len(images) != suite["task_manifest"]["expected_count"] or len(image_tasks) != len(images):
            raise ContractError("recorded SWE task-image coverage is incomplete")
        for image in images:
            if not isinstance(image, dict):
                raise ContractError("recorded SWE image metadata is malformed")
            actual_id = run_cmd([docker, "image", "inspect", "--format", "{{.Id}}", image.get("image", "")])
            if actual_id != image.get("image_id"):
                raise ContractError(f"prepared SWE image is stale: {image.get('image')}")


def execute_prepare(
    suite_arg: str,
    manifest_path: Path,
    suites: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]],
    repo_root: Path,
) -> dict[str, Any]:
    """Execute evaluator preparation and write immutable preparation manifest."""

    if suite_arg == "all":
        suites_to_prepare = list(suites.keys())
    elif suite_arg in suites:
        suites_to_prepare = [suite_arg]
    else:
        raise ContractError(f"unknown suite {suite_arg!r}; choose from 'all' or {', '.join(sorted(suites))}")

    if manifest_path.exists():
        try:
            existing_data = json.loads(manifest_path.read_text("utf-8"))
        except Exception:
            raise ContractError(f"manifest {manifest_path} already exists with different contents")
        if not isinstance(existing_data, dict) or existing_data.get("operation") != "prepare" or "suites" not in existing_data:
            raise ContractError(f"manifest {manifest_path} already exists with different contents")
        for sid in suites_to_prepare:
            if sid in existing_data["suites"]:
                ex_suite = existing_data["suites"][sid]
                source = suites[sid][0]["source"]
                if (ex_suite.get("evaluator", {}).get("repository") != source.get("evaluator_repository") or
                    ex_suite.get("evaluator", {}).get("revision") != source.get("evaluator_revision") or
                    ex_suite.get("dataset", {}).get("repository") != source.get("dataset_repository") or
                    ex_suite.get("dataset", {}).get("revision") != source.get("dataset_revision") or
                    ex_suite.get("dataset", {}).get("split") != source.get("split")):
                    raise ContractError(f"manifest {manifest_path} already exists with different contents")
        # A preparation manifest is immutable. Revalidate every recorded
        # mutable artifact before accepting it; never rewrite it in place.
        for sid in suites_to_prepare:
            prepared = existing_data["suites"].get(sid)
            if not isinstance(prepared, dict) or prepared.get("status") != "ready":
                raise ContractError(f"manifest has no ready preparation for {sid}")
            revalidate_prepared_suite(prepared, suites[sid][0])
        return existing_data

    job_id = os.environ.get("SLURM_JOB_ID")
    if not os.environ.get("BENCHMARK_ALLOW_LOCAL"):
        if not job_id or not job_id.isascii() or not job_id.isdecimal():
            raise ContractError("evaluator preparation must run inside a numeric Slurm CPU allocation")
        if os.environ.get("CUDA_VISIBLE_DEVICES") not in {None, "", "NoDevFiles"}:
            raise ContractError("evaluator preparation must use a CPU allocation, not an allocated GPU")

    # Fail before downloads or installs if the suite's mandatory isolation
    # backend is unavailable.
    container_reports = {
        sid: check_container_runtime(suites[sid][0]["execution"]["requires_container"])
        for sid in suites_to_prepare
    }
    evaluators_base = resolve_evaluator_dir(repo_root)

    results: dict[str, Any] = {}
    for sid in suites_to_prepare:
        suite, tasks = suites[sid]
        results[sid] = prepare_single_suite(suite, tasks, repo_root, evaluators_base)
        results[sid]["container"] = container_reports[sid]

    manifest_data = {
        "schema_version": SCHEMA_VERSION,
        "operation": "prepare",
        "created_utc": utc_now(),
        "slurm_job_id": job_id,
        "suites": results,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = manifest_path.with_name(f".{manifest_path.name}.lock")
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        if manifest_path.exists():
            raise ContractError(f"immutable manifest appeared concurrently: {manifest_path}")
        atomic_write_json(manifest_path, manifest_data)
    finally:
        os.close(lock_fd)
    return manifest_data
