"""Sandbox-side bootstrap and worker for near-data task evaluation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import traceback
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ._secrets import inject_env, read_and_delete_secrets


_MODULE_NAME = "ale_run.executors._sandbox_eval_entry"


def _write_result(spec: dict[str, Any], payload: dict[str, Any]) -> None:
    result = Path(spec["result_path"])
    result.parent.mkdir(parents=True, exist_ok=True)
    tmp = result.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(result)


def _worker_argv(python: str, spec_path: str) -> list[str]:
    return [python, "-m", _MODULE_NAME, spec_path]


def _extract_repo(spec: dict[str, Any]) -> Path:
    archive = Path(spec["archive_path"])
    payload = archive.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != spec["archive_sha256"]:
        raise RuntimeError(f"task archive digest mismatch: {digest}")
    repo = Path(spec["repo_dir"])
    with _cache_lock(repo.with_suffix(".lock")):
        marker = repo / ".archive.sha256"
        if marker.is_file() and marker.read_text(encoding="utf-8").strip() == digest:
            return repo
        tmp = repo.with_name(repo.name + ".tmp-" + uuid.uuid4().hex)
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        with tarfile.open(archive, "r:gz") as tf:
            root = tmp.resolve()
            members = tf.getmembers()
            for member in members:
                target = (tmp / member.name).resolve()
                if target != root and root not in target.parents:
                    raise RuntimeError(f"unsafe task archive member: {member.name}")
                if member.issym() or member.islnk():
                    raise RuntimeError(f"task archive links are not allowed: {member.name}")
            tf.extractall(tmp, members=members, filter="data")
        shutil.rmtree(repo, ignore_errors=True)
        tmp.replace(repo)
        marker.write_text(digest + "\n", encoding="utf-8")
    return repo


def _ensure_venv(spec: dict[str, Any], repo: Path) -> Path:
    venv = Path(spec["venv_dir"])
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    marker = venv / ".ready"
    with _cache_lock(venv.with_suffix(".lock")):
        if python.is_file() and marker.is_file():
            return python
        shutil.rmtree(venv, ignore_errors=True)
        subprocess.run(
            [sys.executable, "-m", "venv", "--system-site-packages", str(venv)],
            check=True,
        )
        install = [
            str(python), "-m", "pip", "install", "--disable-pip-version-check",
            "cua-bench==0.2.7",
        ]
        if (repo / "pyproject.toml").is_file():
            install.append(str(repo))
        subprocess.run(install, check=True)
        marker.write_text("ok\n", encoding="utf-8")
    return python


@contextmanager
def _cache_lock(path: Path, timeout_s: float = 600) -> Any:
    deadline = time.monotonic() + timeout_s
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            path.mkdir()
            (path / "owner.json").write_text(
                json.dumps({"pid": os.getpid(), "created": time.time()}),
                encoding="utf-8",
            )
            break
        except FileExistsError:
            if _lock_owner_is_dead(path):
                shutil.rmtree(path, ignore_errors=True)
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for evaluator cache lock: {path}")
            time.sleep(0.5)
    try:
        yield
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _lock_owner_is_dead(path: Path) -> bool:
    try:
        owner = json.loads((path / "owner.json").read_text(encoding="utf-8"))
        pid = int(owner["pid"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except (PermissionError, OSError):
        return False
    return False


async def _worker(spec: dict[str, Any]) -> dict[str, Any]:
    repo = Path(spec["repo_dir"])
    os.environ["REMOTE_ROOT_DIR"] = str(spec["task_data_root"])
    sys.path.insert(0, str(repo))
    sys.path.insert(0, spec["ale_src_root"])
    from cua_bench.computers.remote import RemoteDesktopSession
    from ale_run.tasks.driver import TaskDriver

    session = RemoteDesktopSession(
        api_url=spec["cua_url"], os_type=spec["os_type"],
    )
    await _start_remote_session(session)
    task_path = repo / Path(spec["task_rel"])
    driver = TaskDriver(
        task_path=str(task_path),
        session=session,
        variant=int(spec["variant"]),
        skip_setup=True,
        os_type=spec["os_type"],
    )
    try:
        return await driver.evaluate()
    finally:
        await driver.close()


async def _start_remote_session(session: Any) -> None:
    """Initialize the localhost CUA client before resilient wrapping."""
    await session.start(headless=True)


def main() -> int:
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if spec.get("secrets_path"):
        inject_env(read_and_delete_secrets(Path(spec["secrets_path"]).parent))
    log_path = Path(spec["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    is_worker = os.environ.get("ALE_EVAL_WORKER") == "1"
    try:
        repo = _extract_repo(spec)
        if not is_worker:
            python = _ensure_venv(spec, repo)
            env = os.environ.copy()
            env["ALE_EVAL_WORKER"] = "1"
            env["PYTHONPATH"] = os.pathsep.join(
                [spec["ale_src_root"], str(repo), env.get("PYTHONPATH", "")]
            )
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    _worker_argv(str(python), sys.argv[1]),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    check=False,
                )
            if completed.returncode != 0 and not Path(spec["result_path"]).exists():
                raise RuntimeError(f"evaluator worker exited {completed.returncode}")
            return completed.returncode
        result = asyncio.run(_worker(spec))
        _write_result(spec, {"ok": True, "result": result})
        return 0
    except BaseException as exc:  # noqa: BLE001
        _write_result(
            spec,
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        return 1
    finally:
        if not is_worker:
            Path(spec["done_path"]).write_text("done\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
