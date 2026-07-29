"""Trusted evaluator registry validation and immutable Git materialization."""

from __future__ import annotations

import hashlib
import json
import os
import selectors
import stat
import subprocess
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path, PurePosixPath

from pydantic import ValidationError

from .contracts import (
    AtomicInfrastructureError,
    EvaluateRequest,
    EvaluatorRegistryRecord,
)

_MAX_REGISTRY_RECORD_BYTES = 1024 * 1024
_MAX_GIT_TREE_BYTES = 64 * 1024 * 1024
_MAX_GIT_CHECKOUT_BYTES = 32 * 1024 * 1024
_MAX_GIT_FILE_BYTES = 8 * 1024 * 1024
_MAX_GIT_FILES = 4096
_MAX_GIT_STATUS_BYTES = 1024 * 1024
_MAX_GIT_ERROR_BYTES = 500
_GIT_MATERIALIZATION_TIMEOUT_SECONDS = 120
_GIT_PROCESS_KILL_WAIT_SECONDS = 1
_GIT_BLOB_MODES = {
    b"100644": 0o644,
    b"100755": 0o755,
}


def validate_evaluator_registry(
    request: EvaluateRequest,
    registry_root: str | Path | None = None,
) -> EvaluatorRegistryRecord:
    """Fail closed unless the external ready record authorizes this checkout."""
    raw = _read_registry_record(request, registry_root)
    if len(raw) > _MAX_REGISTRY_RECORD_BYTES:
        raise AtomicInfrastructureError(
            "evaluator_registry",
            "evaluator registry record exceeds the 1 MiB limit",
        )
    try:
        record = EvaluatorRegistryRecord.model_validate_json(raw)
    except ValidationError as exc:
        raise AtomicInfrastructureError(
            "evaluator_registry",
            f"invalid evaluator registry record: {exc}",
        ) from exc

    expected = {
        "task_path": request.task_path,
        "variant_index": request.variant_index,
        "task_commit": request.task_commit,
        "evaluator_id": request.evaluator_id,
        "evaluator_version": request.evaluator_version,
        "image_id": request.image_id,
    }
    mismatches = [field for field, value in expected.items() if getattr(record, field) != value]
    if mismatches:
        raise AtomicInfrastructureError(
            "evaluator_registry",
            f"evaluator registry identity mismatch: {', '.join(mismatches)}",
        )

    if _read_git_status(
        request.task_repo,
        category="evaluator_registry",
        include_ignored=False,
    ):
        raise AtomicInfrastructureError(
            "evaluator_registry",
            "task repository has tracked or untracked changes",
        )
    main = _run_git(
        request.task_repo,
        "rev-parse",
        "--verify",
        "refs/heads/main^{commit}",
    ).stdout.strip()
    ancestry = subprocess.run(
        [
            "git",
            "-C",
            str(request.task_repo),
            "merge-base",
            "--is-ancestor",
            request.evaluator_version,
            main,
        ],
        capture_output=True,
        check=False,
        text=True,
        env=_git_environment(),
    )
    if ancestry.returncode != 0:
        detail = (ancestry.stderr or ancestry.stdout).strip()
        raise AtomicInfrastructureError(
            "evaluator_registry",
            "evaluator_version is not an ancestor of local main"
            + (f": {detail[:500]}" if detail else ""),
        )
    return record


@contextmanager
def materialize_evaluator_checkout(
    request: EvaluateRequest,
) -> Iterator[Path]:
    """Yield a bounded, link-free task repository from the exact Git commit."""
    with materialize_git_checkout(
        request.task_repo,
        request.evaluator_version,
        category="evaluator_registry",
        prefix="ale-evaluator-checkout-",
    ) as checkout:
        task_dir = checkout / "tasks" / request.task_path
        if not task_dir.is_dir():
            raise AtomicInfrastructureError(
                "evaluator_registry",
                "evaluator commit does not contain the requested task",
            )
        yield checkout


def validate_git_checkout(
    repo: Path,
    commit: str,
    *,
    category: str,
) -> None:
    """Require a clean repository whose HEAD is the requested commit."""
    if _read_git_status(repo, category=category, include_ignored=True):
        raise AtomicInfrastructureError(
            category,
            "task repository has tracked or untracked changes",
        )
    head = _run_git(repo, "rev-parse", "HEAD^{commit}", category=category).stdout.strip()
    if head != commit:
        raise AtomicInfrastructureError(
            category,
            f"task checkout mismatch: expected {commit}, got {head!r}",
        )


@contextmanager
def materialize_git_checkout(
    repo: Path,
    commit: str,
    *,
    category: str,
    prefix: str,
) -> Iterator[Path]:
    """Yield a bounded, link-free checkout containing exact Git commit bytes."""
    with tempfile.TemporaryDirectory(prefix=prefix) as temp_dir:
        deadline = time.monotonic() + _GIT_MATERIALIZATION_TIMEOUT_SECONDS
        resolved_commit_bytes = _run_git_bounded(
            repo,
            "rev-parse",
            "--verify",
            f"{commit}^{{commit}}",
            category=category,
            max_stdout=65,
            deadline=deadline,
        )
        try:
            resolved_commit = resolved_commit_bytes.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise AtomicInfrastructureError(
                category,
                "cannot decode resolved Git commit",
            ) from exc
        if resolved_commit != commit:
            raise AtomicInfrastructureError(
                category,
                f"Git commit mismatch: expected {commit}, got {resolved_commit!r}",
            )
        root = Path(temp_dir)
        checkout = root / "checkout"
        tree = _run_git_bounded(
            repo,
            "ls-tree",
            "-r",
            "-z",
            "--long",
            "--full-tree",
            resolved_commit,
            category=category,
            max_stdout=_MAX_GIT_TREE_BYTES,
            deadline=deadline,
        )
        entries = _preflight_git_tree(tree, category=category, deadline=deadline)
        checkout.mkdir(mode=0o700)
        try:
            for path, object_id, size, permissions in entries:
                if time.monotonic() >= deadline:
                    raise AtomicInfrastructureError(
                        category,
                        "Git materialization timed out",
                    )
                payload = _run_git_bounded(
                    repo,
                    "cat-file",
                    "blob",
                    object_id,
                    category=category,
                    max_stdout=size,
                    deadline=deadline,
                )
                if len(payload) != size:
                    raise AtomicInfrastructureError(
                        category,
                        f"Git blob size changed during materialization: {path}",
                    )
                destination = checkout.joinpath(*path.parts)
                destination.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
                descriptor = os.open(
                    destination,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    permissions,
                )
                try:
                    with os.fdopen(descriptor, "wb") as stream:
                        descriptor = -1
                        stream.write(payload)
                    destination.chmod(permissions, follow_symlinks=False)
                finally:
                    if descriptor != -1:
                        os.close(descriptor)
                if time.monotonic() >= deadline:
                    raise AtomicInfrastructureError(
                        category,
                        "Git materialization timed out",
                    )
        except AtomicInfrastructureError:
            raise
        except OSError as exc:
            raise AtomicInfrastructureError(
                category,
                f"cannot materialize Git objects: {exc}",
            ) from exc
        yield checkout


def _preflight_git_tree(
    tree: bytes,
    *,
    category: str,
    deadline: float,
) -> list[tuple[PurePosixPath, str, int, int]]:
    if tree and not tree.endswith(b"\0"):
        raise AtomicInfrastructureError(category, "malformed Git tree output")
    records = tree[:-1].split(b"\0") if tree else []
    entries: list[tuple[PurePosixPath, str, int, int]] = []
    file_paths: set[PurePosixPath] = set()
    directory_paths: set[PurePosixPath] = set()
    source_bytes = 0
    for record in records:
        if time.monotonic() >= deadline:
            raise AtomicInfrastructureError(category, "Git materialization timed out")
        metadata, separator, raw_path = record.partition(b"\t")
        fields = metadata.split()
        if not separator or len(fields) != 4:
            raise AtomicInfrastructureError(category, "malformed Git tree entry")
        mode, object_type, raw_object_id, raw_size = fields
        raw_parts = raw_path.split(b"/")
        if (
            not raw_path
            or raw_path.startswith(b"/")
            or any(part in {b"", b".", b".."} for part in raw_parts)
        ):
            raise AtomicInfrastructureError(
                category,
                f"unsafe Git tree path: {os.fsdecode(raw_path)}",
            )
        path = PurePosixPath(os.fsdecode(raw_path))
        if object_type != b"blob" or mode not in _GIT_BLOB_MODES:
            raise AtomicInfrastructureError(
                category,
                f"unsafe Git tree entry: {path}",
            )
        if len(raw_object_id) not in {40, 64} or any(
            character not in b"0123456789abcdef" for character in raw_object_id
        ):
            raise AtomicInfrastructureError(category, "malformed Git object id")
        if not raw_size.isdigit():
            raise AtomicInfrastructureError(category, "malformed Git blob size")
        size = int(raw_size)
        if size > _MAX_GIT_FILE_BYTES:
            raise AtomicInfrastructureError(
                category,
                f"Git file exceeds 8 MiB: {path}",
            )
        source_bytes += size
        if source_bytes > _MAX_GIT_CHECKOUT_BYTES:
            raise AtomicInfrastructureError(
                category,
                "Git checkout exceeds 32 MiB of files",
            )
        if path in file_paths:
            raise AtomicInfrastructureError(category, f"duplicate Git tree path: {path}")
        file_paths.add(path)
        directory_paths.update(path.parents[:-1])
        if len(file_paths) + len(directory_paths) > _MAX_GIT_FILES:
            raise AtomicInfrastructureError(
                category,
                f"Git checkout exceeds {_MAX_GIT_FILES} entries",
            )
        entries.append(
            (
                path,
                raw_object_id.decode("ascii"),
                size,
                _GIT_BLOB_MODES[mode],
            )
        )
    if file_paths & directory_paths:
        raise AtomicInfrastructureError(category, "conflicting Git tree paths")
    return entries


def _run_git_bounded(
    repo: Path,
    *arguments: str,
    category: str,
    max_stdout: int,
    deadline: float,
) -> bytes:
    command = ["git", "-C", str(repo), *arguments]
    if time.monotonic() >= deadline:
        raise AtomicInfrastructureError(category, "Git materialization timed out")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_git_environment(),
        )
    except OSError as exc:
        raise AtomicInfrastructureError(
            category,
            f"cannot start git {arguments[0]}: {exc}",
        ) from exc
    assert process.stdout is not None
    assert process.stderr is not None
    stdout = bytearray()
    stderr = bytearray()
    selector = selectors.DefaultSelector()
    streams = {
        process.stdout: (stdout, max_stdout),
        process.stderr: (stderr, _MAX_GIT_ERROR_BYTES),
    }
    try:
        for stream in streams:
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ)
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise AtomicInfrastructureError(category, "Git materialization timed out")
            ready = selector.select(remaining)
            if not ready:
                raise AtomicInfrastructureError(category, "Git materialization timed out")
            for key, _events in ready:
                stream = key.fileobj
                output, limit = streams[stream]
                try:
                    chunk = os.read(stream.fileno(), min(64 * 1024, limit - len(output) + 1))
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(stream)
                    stream.close()
                    continue
                output.extend(chunk)
                if len(output) > limit:
                    label = "output" if stream is process.stdout else "error output"
                    raise AtomicInfrastructureError(
                        category,
                        f"git {arguments[0]} {label} exceeds its limit",
                    )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AtomicInfrastructureError(category, "Git materialization timed out")
        try:
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            raise AtomicInfrastructureError(
                category,
                "Git materialization timed out",
            ) from exc
        if returncode != 0:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise AtomicInfrastructureError(
                category,
                f"git {arguments[0]} failed" + (f": {detail}" if detail else ""),
            )
        return bytes(stdout)
    except BaseException:
        if process.poll() is None:
            with suppress(OSError):
                process.kill()
        with suppress(OSError, subprocess.TimeoutExpired):
            process.wait(timeout=_GIT_PROCESS_KILL_WAIT_SECONDS)
        raise
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()


def _read_registry_record(
    request: EvaluateRequest,
    registry_root: str | Path | None,
) -> bytes:
    configured_root = (
        registry_root
        if registry_root is not None
        else os.environ.get("ALE_EVALUATOR_REGISTRY_ROOT")
    )
    if configured_root is None or not str(configured_root):
        raise AtomicInfrastructureError(
            "evaluator_registry",
            "ALE_EVALUATOR_REGISTRY_ROOT is not configured",
        )
    root = Path(configured_root)
    if not root.is_absolute():
        raise AtomicInfrastructureError(
            "evaluator_registry",
            "evaluator registry root must be absolute",
        )

    identity = {
        field: getattr(request, field)
        for field in (
            "task_path",
            "variant_index",
            "task_commit",
            "evaluator_id",
            "evaluator_version",
            "image_id",
        )
    }
    encoded_identity = json.dumps(
        identity,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    filename = f"{hashlib.sha256(encoded_identity).hexdigest()}.json"
    record_path = root / filename
    if record_path.parent != root:
        raise AtomicInfrastructureError(
            "evaluator_registry",
            "evaluator registry record escapes the configured root",
        )

    root_fd: int | None = None
    record_fd: int | None = None
    try:
        root_fd = _open_registry_root(root)
        record_fd = os.open(
            filename,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=root_fd,
        )
        record_status = os.fstat(record_fd)
        if not stat.S_ISREG(record_status.st_mode):
            raise AtomicInfrastructureError(
                "evaluator_registry",
                "evaluator registry record is not a regular file",
            )
        with os.fdopen(record_fd, "rb") as stream:
            record_fd = None
            return stream.read(_MAX_REGISTRY_RECORD_BYTES + 1)
    except AtomicInfrastructureError:
        raise
    except OSError as exc:
        raise AtomicInfrastructureError(
            "evaluator_registry",
            f"cannot read evaluator registry record: {exc}",
        ) from exc
    finally:
        try:
            if record_fd is not None:
                os.close(record_fd)
        finally:
            if root_fd is not None:
                os.close(root_fd)


def _open_registry_root(root: Path) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    current_fd = os.open("/", flags)
    try:
        for component in root.parts[1:]:
            if component in {"", ".", ".."}:
                raise AtomicInfrastructureError(
                    "evaluator_registry",
                    "evaluator registry root contains an unsafe path component",
                )
            component_status = os.stat(
                component,
                dir_fd=current_fd,
                follow_symlinks=False,
            )
            if stat.S_ISLNK(component_status.st_mode):
                raise AtomicInfrastructureError(
                    "evaluator_registry",
                    "evaluator registry root must not contain symlinks",
                )
            if not stat.S_ISDIR(component_status.st_mode):
                raise AtomicInfrastructureError(
                    "evaluator_registry",
                    "evaluator registry root is not a directory",
                )
            next_fd = os.open(component, flags, dir_fd=current_fd)
            previous_fd = current_fd
            current_fd = next_fd
            os.close(previous_fd)
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _run_git(
    repo: Path,
    *arguments: str,
    category: str = "evaluator_registry",
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        capture_output=True,
        check=False,
        text=True,
        env=_git_environment(),
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise AtomicInfrastructureError(
            category,
            f"git {' '.join(arguments)} failed: {detail[:500]}",
        )
    return result


def _read_git_status(
    repo: Path,
    *,
    category: str,
    include_ignored: bool,
) -> bytes:
    arguments = [
        "git",
        "-C",
        str(repo),
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    ]
    if include_ignored:
        arguments.append("--ignored=matching")
    with tempfile.TemporaryFile() as stderr:
        try:
            process = subprocess.Popen(
                arguments,
                stdout=subprocess.PIPE,
                stderr=stderr,
                env=_git_environment(),
            )
        except OSError as exc:
            raise AtomicInfrastructureError(
                category,
                f"cannot start git status: {exc}",
            ) from exc
        assert process.stdout is not None
        try:
            output = process.stdout.read(_MAX_GIT_STATUS_BYTES + 1)
            if len(output) > _MAX_GIT_STATUS_BYTES:
                process.kill()
                process.wait()
                raise AtomicInfrastructureError(
                    category,
                    "git status output exceeds the 1 MiB limit",
                )
            returncode = process.wait()
        except BaseException:
            if process.poll() is None:
                process.kill()
                process.wait()
            raise
        if returncode != 0:
            stderr.seek(0)
            detail = stderr.read(501).decode("utf-8", errors="replace").strip()
            raise AtomicInfrastructureError(
                category,
                "git status failed" + (f": {detail[:500]}" if detail else ""),
            )
        return output


def _git_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    return environment
