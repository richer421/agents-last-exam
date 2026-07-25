"""Run task evaluation beside sandbox-resident task artifacts.

The host ships code only, starts a detached evaluator in the sandbox, and
reads back a small JSON result plus text log. PSDs, images, and references
remain in the data environment.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import tarfile
import time
import tomllib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..base_interface import SandboxHandle


_POLL_INTERVAL_S = 2.0
_MAX_RESULT_BYTES = 4 * 1024 * 1024
_MAX_LOG_BYTES = 2 * 1024 * 1024
_MAX_CODE_FILE_BYTES = 8 * 1024 * 1024
_MAX_CODE_FILES = 4096
_MAX_CODE_SOURCE_BYTES = 32 * 1024 * 1024
_MAX_CODE_ARCHIVE_BYTES = 16 * 1024 * 1024
_CODE_SUFFIXES = frozenset(
    {".cfg", ".ini", ".js", ".json", ".lock", ".md", ".py", ".scm", ".toml", ".txt", ".yaml", ".yml"}
)
_EXECUTABLE_CODE_SUFFIXES = frozenset({".js", ".py", ".scm"})
_MEDIA_OR_ARCHIVE_SUFFIXES = frozenset(
    {
        ".7z", ".avi", ".bmp", ".gif", ".gz", ".jpeg", ".jpg", ".mov",
        ".mp3", ".mp4", ".pdf", ".png", ".psb", ".psd", ".tar", ".tif",
        ".tiff", ".wav", ".webm", ".webp", ".zip",
    }
)
_SENSITIVE_ARCHIVE_NAME = re.compile(
    r"(?i)(^|[._-])(?:api[_-]?keys?|auth(?:orization)?|credentials?|passwords?|"
    r"private[_-]?keys?|references?|outputs?|secrets?|signed[_-]?urls?|tokens?)"
    r"(?:[._-]|$)"
)
_MEDIA_MAGIC_PREFIXES = (
    b"8BPS",
    b"7z\xbc\xaf'\x1c",
    b"BM",
    b"\x1aE\xdf\xa3",
    b"\x1f\x8b",
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
    b"GIF87a",
    b"GIF89a",
    b"%PDF-",
    b"PK\x03\x04",
    b"II*\x00",
    b"MM\x00*",
)
_EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "artifacts",
        "node_modules",
    }
)


def _extra_archive_patterns(repo_root: Path) -> tuple[str, ...]:
    pyproject = repo_root / "pyproject.toml"
    if not pyproject.is_file():
        return ()
    config = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    include = (
        config.get("tool", {})
        .get("ale", {})
        .get("evaluator_archive", {})
        .get("include", [])
    )
    if not isinstance(include, list) or not all(isinstance(item, str) for item in include):
        raise ValueError("tool.ale.evaluator_archive.include must be a list of paths/globs")
    for pattern in include:
        normalized = pattern.replace("\\", "/")
        if (
            not normalized
            or normalized.startswith(("/", "../"))
            or "/../" in normalized
            or "**" in normalized.split("/")
        ):
            raise ValueError(f"unsafe evaluator archive include pattern: {pattern}")
    return tuple(item.replace("\\", "/") for item in include)


def _archive_path_denied(relative: Path) -> bool:
    parts = tuple(part.lower() for part in relative.parts)
    name = relative.name.lower()
    return (
        any(part in _EXCLUDED_DIRS for part in parts[:-1])
        or any(part in {"output", "outputs", "reference", "references"} for part in parts[:-1])
        or name.startswith(".")
        or (
            relative.suffix.lower() not in _EXECUTABLE_CODE_SUFFIXES
            and _SENSITIVE_ARCHIVE_NAME.search(name) is not None
        )
        or relative.suffix.lower() in _MEDIA_OR_ARCHIVE_SUFFIXES
    )


def _has_media_magic(path: Path) -> bool:
    with path.open("rb") as stream:
        head = stream.read(16)
    return (
        any(head.startswith(prefix) for prefix in _MEDIA_MAGIC_PREFIXES)
        or (head.startswith(b"RIFF") and head[8:12] == b"WEBP")
        or (head.startswith(b"RIFF") and head[8:12] in {b"AVI ", b"WAVE"})
        or (len(head) >= 12 and head[4:8] == b"ftyp")
    )


@dataclass(frozen=True)
class TaskArchive:
    payload: bytes
    digest: str
    files: int


@dataclass(frozen=True)
class SandboxEvaluationResult:
    result: dict[str, Any]
    log: str


class SandboxEvaluationError(RuntimeError):
    """Remote evaluator failure carrying its bounded sandbox log."""

    def __init__(self, message: str, *, evaluator_log: str = "") -> None:
        self.evaluator_log = evaluator_log
        tail = evaluator_log[-20_000:]
        detail = f"{message}\nremote evaluator log:\n{tail}" if tail else message
        super().__init__(detail)


def _build_task_archive(repo_root: Path) -> TaskArchive:
    """Build a deterministic code/metadata archive for a task repository."""
    repo_root = repo_root.resolve()
    extra_patterns = _extra_archive_patterns(repo_root)
    files: list[Path] = []
    source_bytes = 0
    for root, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = sorted(d for d in dirnames if d not in _EXCLUDED_DIRS)
        root_path = Path(root)
        for filename in sorted(filenames):
            path = root_path / filename
            relative = path.relative_to(repo_root)
            explicitly_included = any(relative.match(pattern) for pattern in extra_patterns)
            if (
                path.is_symlink()
                or _archive_path_denied(relative)
                or (
                    path.suffix.lower() not in _CODE_SUFFIXES
                    and not explicitly_included
                )
            ):
                continue
            if _has_media_magic(path):
                raise ValueError(f"task archive contains media content: {relative}")
            size = path.stat().st_size
            if size > _MAX_CODE_FILE_BYTES:
                raise ValueError(f"task code file exceeds {_MAX_CODE_FILE_BYTES} bytes: {path}")
            source_bytes += size
            if len(files) + 1 > _MAX_CODE_FILES:
                raise ValueError(f"task code archive exceeds {_MAX_CODE_FILES} files")
            if source_bytes > _MAX_CODE_SOURCE_BYTES:
                raise ValueError(
                    f"task code archive exceeds {_MAX_CODE_SOURCE_BYTES} source bytes"
                )
            files.append(path)

    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb", filename="", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tf:
            for path in files:
                data = path.read_bytes()
                info = tarfile.TarInfo(path.relative_to(repo_root).as_posix())
                info.size = len(data)
                info.mode = stat.S_IMODE(path.stat().st_mode)
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                tf.addfile(info, io.BytesIO(data))
    payload = out.getvalue()
    if len(payload) > _MAX_CODE_ARCHIVE_BYTES:
        raise ValueError(
            f"task code archive exceeds {_MAX_CODE_ARCHIVE_BYTES} compressed bytes"
        )
    return TaskArchive(
        payload=payload,
        digest=hashlib.sha256(payload).hexdigest(),
        files=len(files),
    )


def _task_repo_root(task_path: Path) -> Path:
    resolved = task_path.resolve()
    for parent in (resolved, *resolved.parents):
        if (parent / "tasks").is_dir() and resolved.is_relative_to(parent / "tasks"):
            return parent
    raise ValueError(f"task path is not inside a task repository: {task_path}")


async def evaluate_in_sandbox(
    *,
    sandbox: SandboxHandle,
    ale_src_root: str,
    task_path: Path,
    variant: int,
    timeout_s: float,
) -> SandboxEvaluationResult:
    """Execute a task's evaluator in the sandbox and return its small result."""
    repo_root = _task_repo_root(task_path)
    archive = _build_task_archive(repo_root)
    task_rel = task_path.resolve().relative_to(repo_root).as_posix()
    sep = "/" if sandbox.is_linux else "\\"
    root = f"{sandbox.work_dir_base.rstrip(sep)}{sep}evaluator"
    job = f"{root}{sep}jobs{sep}{archive.digest[:16]}-{uuid.uuid4().hex}"
    archive_path = f"{job}{sep}tasks.tar.gz"
    spec_path = f"{job}{sep}spec.json"
    result_path = f"{job}{sep}result.json"
    log_path = f"{job}{sep}eval.log"
    done_path = f"{job}{sep}done.marker"
    pid_path = f"{job}{sep}pid"

    spec = {
        "archive_path": archive_path,
        "archive_sha256": archive.digest,
        "repo_dir": f"{root}{sep}repos{sep}{archive.digest[:16]}",
        "venv_dir": f"{root}{sep}venvs{sep}{archive.digest[:16]}",
        "task_rel": task_rel,
        "variant": variant,
        "cua_url": f"http://127.0.0.1:{sandbox.cua_server_port}",
        "os_type": "linux" if sandbox.is_linux else "windows",
        "ale_src_root": ale_src_root,
        "result_path": result_path,
        "log_path": log_path,
        "done_path": done_path,
        "pid_path": pid_path,
        "task_data_root": sandbox.task_data_root,
    }
    await sandbox.mkdir(job)
    await sandbox.write_file(archive_path, archive.payload)
    await sandbox.write_file(
        spec_path,
        (json.dumps(spec, ensure_ascii=True, separators=(",", ":")) + "\n").encode(),
    )

    command = _launch_command(
        sandbox,
        ale_src_root=ale_src_root,
        spec_path=spec_path,
        pid_path=pid_path,
    )
    launched = await sandbox.run_command(command, timeout=30)
    if launched.returncode != 0:
        detail = (launched.stderr or launched.stdout or "unknown launch failure").strip()
        raise RuntimeError(f"sandbox evaluator launch failed: {detail[:500]}")

    pid = _parse_pid(launched.stdout)
    if pid is None:
        raise RuntimeError("sandbox evaluator launcher did not return a PID")

    completed = False
    try:
        deadline = time.monotonic() + timeout_s
        while not await sandbox.exists(done_path):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"sandbox evaluator exceeded {timeout_s:.0f}s")
            if not await _process_alive(sandbox, pid):
                await asyncio.sleep(1)
                if not await sandbox.exists(done_path):
                    raise RuntimeError("sandbox evaluator exited before done.marker")
                break
            await asyncio.sleep(_POLL_INTERVAL_S)
        completed = True
    finally:
        if not completed:
            await _kill_process_tree(sandbox, pid)

    result_size = await _remote_file_size(sandbox, result_path)
    if result_size is None:
        raise RuntimeError("sandbox evaluator did not produce result.json")
    if result_size > _MAX_RESULT_BYTES:
        raise RuntimeError(f"sandbox evaluator result is too large: {result_size} bytes")
    raw = await sandbox.read_file(result_path)
    payload = json.loads(raw)
    log_size = await _remote_file_size(sandbox, log_path)
    try:
        if log_size is None:
            raise FileNotFoundError(log_path)
        if log_size > _MAX_LOG_BYTES:
            raise RuntimeError(f"sandbox evaluator log is too large: {log_size} bytes")
        log = (await sandbox.read_file(log_path)).decode("utf-8", errors="replace")
    except FileNotFoundError:
        log = ""
    if not payload.get("ok"):
        raise SandboxEvaluationError(
            str(payload.get("error") or "sandbox evaluator failed"),
            evaluator_log=log,
        )
    result = payload.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("sandbox evaluator returned a non-object result")
    return SandboxEvaluationResult(result=result, log=log)


def _launch_command(
    sandbox: SandboxHandle, *, ale_src_root: str, spec_path: str, pid_path: str,
) -> str:
    module = "ale_run.executors._sandbox_eval_entry"
    if sandbox.is_linux:
        import shlex

        python = shlex.quote(sandbox.python)
        env = shlex.quote(ale_src_root)
        spec = shlex.quote(spec_path)
        pid_file = shlex.quote(pid_path)
        return (
            f"PYTHONPATH={env} nohup setsid {python} -m {module} {spec} "
            f">/dev/null 2>&1 & pid=$!; echo $pid > {pid_file}; "
            "echo __ALE_EVAL_PID__=$pid"
        )
    args = subprocess.list2cmdline(["-m", module, spec_path])
    py = sandbox.python.replace("'", "''")
    ale = ale_src_root.replace("'", "''")
    pid_file = pid_path.replace("'", "''")
    escaped_args = args.replace("'", "''")
    return (
        "powershell -NoProfile -NonInteractive -Command \""
        f"$env:PYTHONPATH='{ale}'; $p=Start-Process -FilePath '{py}' "
        f"-ArgumentList '{escaped_args}' -WindowStyle Hidden -PassThru; "
        f"Set-Content -LiteralPath '{pid_file}' -Value $p.Id; "
        "Write-Output ('__ALE_EVAL_PID__=' + $p.Id)\""
    )


def _parse_pid(stdout: str | None) -> int | None:
    match = re.search(r"__ALE_EVAL_PID__=(\d+)", stdout or "")
    return int(match.group(1)) if match else None


async def _process_alive(sandbox: SandboxHandle, pid: int) -> bool:
    command = (
        f"kill -0 {pid}"
        if sandbox.is_linux
        else f'tasklist /FI "PID eq {pid}" /NH | findstr /R /C:"[ ]{pid}[ ]"'
    )
    result = await sandbox.run_command(command, timeout=30)
    return result.returncode == 0


async def _kill_process_tree(sandbox: SandboxHandle, pid: int) -> None:
    command = (
        f"kill -TERM -{pid}; sleep 2; kill -KILL -{pid} 2>/dev/null || true"
        if sandbox.is_linux
        else f"taskkill /PID {pid} /T /F"
    )
    try:
        await sandbox.run_command(command, timeout=30)
    except Exception:  # noqa: BLE001
        pass


async def _remote_file_size(sandbox: SandboxHandle, path: str) -> int | None:
    if sandbox.is_linux:
        import shlex

        command = f"stat -c %s {shlex.quote(path)}"
    else:
        quoted = path.replace("'", "''")
        command = (
            "powershell -NoProfile -NonInteractive -Command \""
            f"if (Test-Path -LiteralPath '{quoted}') {{ "
            f"(Get-Item -LiteralPath '{quoted}').Length }} else {{ exit 3 }}\""
        )
    last_error = ""
    for attempt in range(3):
        result = await sandbox.run_command(command, timeout=30)
        if result.returncode == 0:
            try:
                return int((result.stdout or "").strip().splitlines()[-1])
            except (ValueError, IndexError):
                raise RuntimeError(
                    f"invalid remote file size for {path}: {result.stdout!r}"
                ) from None
        if result.returncode == 3:
            return None
        last_error = (result.stderr or result.stdout or f"rc={result.returncode}").strip()
        if attempt < 2:
            await asyncio.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"remote stat failed for {path}: {last_error[:500]}")
