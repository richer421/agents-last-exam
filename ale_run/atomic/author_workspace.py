"""Isolated Git workspace and task-local authoring policy."""

from __future__ import annotations

import hashlib
import os
import re
import selectors
import shutil
import signal
import stat
import subprocess
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .author_registry import author_registry_key
from .contracts import (
    AtomicInfrastructureError,
    AuthorEvaluatorRegistryIdentity,
    AuthorEvaluatorRequest,
)
from .evaluator_registry import materialize_git_checkout

_PROCESS_TIMEOUT_SECONDS = 120
_MAX_PROCESS_OUTPUT_BYTES = 1024 * 1024
_MAX_WORKSPACE_BYTES = 32 * 1024 * 1024
_MAX_WORKSPACE_FILE_BYTES = 8 * 1024 * 1024
_MAX_WORKSPACE_ENTRIES = 4096
_PROCESS_KILL_WAIT_SECONDS = 1


@dataclass(frozen=True)
class ProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes


@dataclass(frozen=True)
class AuthorWorkspace:
    root: Path
    task_directory: Path
    branch_name: str
    task_commit: str
    _environment: Mapping[str, str]
    _control_fingerprint: str

    def validate_diff(
        self,
        *,
        allowed_task_dependencies: Sequence[str] = (),
    ) -> tuple[str, ...]:
        """Return allowed changed paths or fail closed on any policy violation."""
        if _git_control_fingerprint(self.root) != self._control_fingerprint:
            raise AtomicInfrastructureError(
                "author_workspace",
                "author modified protected Git control state",
            )
        _validate_workspace_entries(self.root)
        result = _run_bounded_process(
            (
                "git",
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                "--ignored=matching",
            ),
            cwd=self.root,
            environment=self._environment,
        )
        if result.returncode != 0:
            raise AtomicInfrastructureError(
                "author_workspace",
                _process_failure("git status", result),
            )
        task_prefix = PurePosixPath(self.task_directory.relative_to(self.root).as_posix())
        allowed_dependencies = {
            _validate_relative_policy_path(path) for path in allowed_task_dependencies
        }
        changed: list[str] = []
        records = result.stdout[:-1].split(b"\0") if result.stdout else []
        if result.stdout and not result.stdout.endswith(b"\0"):
            raise AtomicInfrastructureError(
                "author_workspace",
                "malformed Git status output",
            )
        for record in records:
            if len(record) < 4 or record[2:3] != b" ":
                raise AtomicInfrastructureError(
                    "author_workspace",
                    "malformed Git status entry",
                )
            status_code = record[:2]
            if status_code == b"!!":
                raise AtomicInfrastructureError(
                    "author_workspace",
                    "ignored authoring output would not be committed",
                )
            if b"R" in status_code or b"C" in status_code:
                raise AtomicInfrastructureError(
                    "author_workspace",
                    "renamed or copied paths are not allowed during authoring",
                )
            path = PurePosixPath(os.fsdecode(record[3:]))
            if not _is_canonical_relative(path):
                raise AtomicInfrastructureError(
                    "author_workspace",
                    "Git reported an unsafe changed path",
                )
            if not _is_allowed_task_change(path, task_prefix, allowed_dependencies):
                raise AtomicInfrastructureError(
                    "author_workspace",
                    f"authoring change is not allowed: {path.as_posix()}",
                )
            changed.append(path.as_posix())
        return tuple(sorted(changed))


@contextmanager
def isolated_author_workspace(
    source_repository: Path,
    request: AuthorEvaluatorRequest,
) -> Iterator[AuthorWorkspace]:
    """Yield a clean isolated branch rooted at the exact requested commit."""
    branch_name = _author_branch_name(request)
    with materialize_git_checkout(
        source_repository,
        request.task_commit,
        category="author_workspace",
        prefix="ale-author-workspace-",
    ) as root:
        empty_template = root.parent / "empty-git-template"
        empty_template.mkdir(mode=0o700)
        private_home = root.parent / "home"
        private_home.mkdir(mode=0o700)
        environment = _isolated_git_environment(private_home)
        commands = (
            ("git", "init", "--quiet", f"--template={empty_template}", str(root)),
            (
                "git",
                "fetch",
                "--quiet",
                "--no-tags",
                "--no-recurse-submodules",
                "--depth=1",
                str(source_repository.resolve(strict=True)),
                request.task_commit,
            ),
            (
                "git",
                "update-ref",
                f"refs/heads/{branch_name}",
                request.task_commit,
            ),
            ("git", "symbolic-ref", "HEAD", f"refs/heads/{branch_name}"),
            ("git", "read-tree", request.task_commit),
        )
        for command in commands:
            result = _run_bounded_process(
                command,
                cwd=root,
                environment=environment,
            )
            if result.returncode != 0:
                raise AtomicInfrastructureError(
                    "author_workspace",
                    _process_failure(" ".join(command[:2]), result),
                )
        status_result = _run_bounded_process(
            ("git", "status", "--porcelain=v1", "-z", "--untracked-files=all"),
            cwd=root,
            environment=environment,
        )
        if status_result.returncode != 0 or status_result.stdout:
            raise AtomicInfrastructureError(
                "author_workspace",
                "materialized author workspace is not clean",
            )
        task_directory = root / "tasks" / request.task_path
        if not task_directory.is_dir():
            raise AtomicInfrastructureError(
                "author_workspace",
                "requested task is absent from task_commit",
            )
        _scrub_legacy_evaluator(task_directory)
        workspace = AuthorWorkspace(
            root=root,
            task_directory=task_directory,
            branch_name=branch_name,
            task_commit=request.task_commit,
            _environment=environment,
            _control_fingerprint=_git_control_fingerprint(root),
        )
        yield workspace


def _scrub_legacy_evaluator(task_directory: Path) -> None:
    evaluator = task_directory / "evaluator"
    if evaluator.exists():
        shutil.rmtree(evaluator)
    tests_directory = task_directory / "tests"
    if not tests_directory.is_dir():
        return
    evaluator_tests = tests_directory / "evaluator"
    if evaluator_tests.exists():
        shutil.rmtree(evaluator_tests)
    for legacy_test in tests_directory.glob("test_evaluator*"):
        if legacy_test.is_dir():
            shutil.rmtree(legacy_test)
        else:
            legacy_test.unlink()


def _author_branch_name(request: AuthorEvaluatorRequest) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", request.task_path.lower()).strip("-")
    slug = slug[:80] or "task"
    identity = request.model_dump(
        include={
            "task_commit",
            "rubric_hash",
            "reference_manifest_hash",
            "evaluator_sdk_version",
            "image_id",
        }
    )
    digest = author_registry_key(AuthorEvaluatorRegistryIdentity(**identity))[:12]
    return f"ale/author-{slug}-{digest}"


def _isolated_git_environment(private_home: Path) -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(
        {
            "HOME": str(private_home),
            "XDG_CONFIG_HOME": str(private_home / ".config"),
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_COUNT": "0",
            "GIT_ATTR_GLOBAL": os.devnull,
            "GIT_ATTR_SYSTEM": os.devnull,
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_NO_LAZY_FETCH": "1",
        }
    )
    return environment


def _git_control_fingerprint(root: Path) -> str:
    git_directory = root / ".git"
    protected = (
        git_directory / "config",
        git_directory / "HEAD",
        git_directory / "index",
        git_directory / "packed-refs",
        git_directory / "shallow",
        git_directory / "hooks",
        git_directory / "info",
        git_directory / "refs",
        git_directory / "objects" / "info",
    )
    digest = hashlib.sha256()
    for candidate in protected:
        if candidate.is_symlink():
            raise AtomicInfrastructureError(
                "author_workspace",
                "protected Git control state contains an unsafe entry",
            )
        if not candidate.exists():
            continue
        entries = [candidate]
        if candidate.is_dir():
            entries = sorted(candidate.rglob("*"))
        for entry in entries:
            relative = entry.relative_to(git_directory).as_posix().encode("utf-8")
            status_result = entry.lstat()
            digest.update(relative)
            digest.update(stat.S_IFMT(status_result.st_mode).to_bytes(4, "big"))
            digest.update(stat.S_IMODE(status_result.st_mode).to_bytes(2, "big"))
            if stat.S_ISREG(status_result.st_mode):
                payload = entry.read_bytes()
                if len(payload) > _MAX_PROCESS_OUTPUT_BYTES:
                    raise AtomicInfrastructureError(
                        "author_workspace",
                        "protected Git control file exceeds 1 MiB",
                    )
                digest.update(payload)
            elif not stat.S_ISDIR(status_result.st_mode):
                raise AtomicInfrastructureError(
                    "author_workspace",
                    "protected Git control state contains an unsafe entry",
                )
    return digest.hexdigest()


def _validate_workspace_entries(root: Path) -> None:
    entries = 0
    total_bytes = 0
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        relative_directory = Path(directory).relative_to(root)
        if relative_directory == Path(".") and ".git" in directory_names:
            directory_names.remove(".git")
        for name in (*directory_names, *file_names):
            path = Path(directory) / name
            relative = path.relative_to(root)
            status_result = path.lstat()
            entries += 1
            if entries > _MAX_WORKSPACE_ENTRIES:
                raise AtomicInfrastructureError(
                    "author_workspace",
                    "author workspace exceeds 4096 entries",
                )
            if stat.S_ISLNK(status_result.st_mode):
                raise AtomicInfrastructureError(
                    "author_workspace",
                    f"author workspace contains a symlink: {relative.as_posix()}",
                )
            if name == ".git":
                raise AtomicInfrastructureError(
                    "author_workspace",
                    f"author workspace contains nested Git metadata: {relative.as_posix()}",
                )
            if stat.S_ISREG(status_result.st_mode):
                if status_result.st_size > _MAX_WORKSPACE_FILE_BYTES:
                    raise AtomicInfrastructureError(
                        "author_workspace",
                        f"author workspace file exceeds 8 MiB: {relative.as_posix()}",
                    )
                total_bytes += status_result.st_size
                if total_bytes > _MAX_WORKSPACE_BYTES:
                    raise AtomicInfrastructureError(
                        "author_workspace",
                        "author workspace exceeds 32 MiB",
                    )
            elif not stat.S_ISDIR(status_result.st_mode):
                raise AtomicInfrastructureError(
                    "author_workspace",
                    f"author workspace contains a non-regular entry: {relative.as_posix()}",
                )


def _is_allowed_task_change(
    path: PurePosixPath,
    task_prefix: PurePosixPath,
    allowed_dependencies: set[PurePosixPath],
) -> bool:
    try:
        task_relative = path.relative_to(task_prefix)
    except ValueError:
        return False
    if not task_relative.parts:
        return False
    if task_relative.parts[0] == "evaluator":
        return True
    if task_relative.parts[0] == "tests" and len(task_relative.parts) >= 2:
        return task_relative.parts[1] == "evaluator" or task_relative.name.startswith(
            "test_evaluator"
        )
    return task_relative in allowed_dependencies


def _validate_relative_policy_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not _is_canonical_relative(path) or ".git" in path.parts:
        raise AtomicInfrastructureError(
            "author_workspace",
            f"invalid allowed dependency path: {value}",
        )
    return path


def _is_canonical_relative(path: PurePosixPath) -> bool:
    return (
        bool(path.parts)
        and not path.is_absolute()
        and all(part not in {"", ".", ".."} for part in path.parts)
        and path.as_posix() != "."
    )


def _process_failure(label: str, result: ProcessResult) -> str:
    detail = (result.stderr or result.stdout).decode("utf-8", errors="replace").strip()
    return f"{label} failed" + (f": {detail[:1000]}" if detail else "")


def _run_bounded_process(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: float = _PROCESS_TIMEOUT_SECONDS,
    max_output_bytes: int = _MAX_PROCESS_OUTPUT_BYTES,
) -> ProcessResult:
    deadline = time.monotonic() + timeout_seconds
    try:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise AtomicInfrastructureError(
            "author_workspace",
            f"cannot start {command[0]}: {exc}",
        ) from exc
    assert process.stdout is not None
    assert process.stderr is not None
    outputs = {process.stdout: bytearray(), process.stderr: bytearray()}
    selector = selectors.DefaultSelector()
    try:
        for stream in outputs:
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AtomicInfrastructureError(
                    "author_workspace",
                    f"{command[0]} timed out",
                )
            ready = selector.select(remaining)
            if not ready:
                raise AtomicInfrastructureError(
                    "author_workspace",
                    f"{command[0]} timed out",
                )
            for key, _events in ready:
                stream = key.fileobj
                output = outputs[stream]
                try:
                    chunk = os.read(
                        stream.fileno(),
                        min(64 * 1024, max_output_bytes - len(output) + 1),
                    )
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(stream)
                    stream.close()
                    continue
                output.extend(chunk)
                if len(output) > max_output_bytes:
                    raise AtomicInfrastructureError(
                        "author_workspace",
                        f"{command[0]} output exceeds its limit",
                    )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AtomicInfrastructureError(
                "author_workspace",
                f"{command[0]} timed out",
            )
        try:
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            raise AtomicInfrastructureError(
                "author_workspace",
                f"{command[0]} timed out",
            ) from exc
        return ProcessResult(
            returncode=returncode,
            stdout=bytes(outputs[process.stdout]),
            stderr=bytes(outputs[process.stderr]),
        )
    except BaseException:
        if process.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        with suppress(OSError, subprocess.TimeoutExpired):
            process.wait(timeout=_PROCESS_KILL_WAIT_SECONDS)
        raise
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()
