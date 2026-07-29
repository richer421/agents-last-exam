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
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory as _PrivateTemporaryDirectory

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
_MAX_GIT_COMMAND_BYTES = 1024 * 1024
_MAX_GIT_CONTROL_FILE_BYTES = 1024 * 1024
_MAX_GIT_INDEX_BYTES = 64 * 1024 * 1024
_MAX_GIT_SHARED_INDEXES = 16
_GIT_MATERIALIZATION_TIMEOUT_SECONDS = 120
_GIT_OPERATION_TIMEOUT_SECONDS = 120
_GIT_PROCESS_KILL_WAIT_SECONDS = 1
_GIT_BLOB_MODES = {
    b"100644": 0o644,
    b"100755": 0o755,
}


@dataclass(frozen=True)
class _IsolatedGitRepository:
    work_tree: Path
    git_dir: Path


@dataclass(frozen=True)
class _GitProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes


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

    with _isolated_git_repository(
        request.task_repo,
        category="evaluator_registry",
    ) as repository:
        if _read_git_status(
            repository,
            category="evaluator_registry",
            include_ignored=False,
        ):
            raise AtomicInfrastructureError(
                "evaluator_registry",
                "task repository has tracked or untracked changes",
            )
        main = _run_git(
            repository,
            "rev-parse",
            "--verify",
            "refs/heads/main^{commit}",
        ).stdout.strip()
        ancestry = _run_git_process_bounded(
            repository,
            "merge-base",
            "--is-ancestor",
            request.evaluator_version,
            main,
            category="evaluator_registry",
            max_stdout=_MAX_GIT_COMMAND_BYTES,
            deadline=time.monotonic() + _GIT_OPERATION_TIMEOUT_SECONDS,
        )
        if ancestry.returncode != 0:
            detail = (
                (ancestry.stderr or ancestry.stdout)
                .decode(
                    "utf-8",
                    errors="replace",
                )
                .strip()
            )
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
    with _isolated_git_repository(repo, category=category) as repository:
        if _read_git_status(repository, category=category, include_ignored=True):
            raise AtomicInfrastructureError(
                category,
                "task repository has tracked or untracked changes",
            )
        head = _run_git(
            repository,
            "rev-parse",
            "HEAD^{commit}",
            category=category,
        ).stdout.strip()
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
    with (
        _isolated_git_repository(repo, category=category) as repository,
        _materialize_isolated_git_checkout(
            repository,
            commit,
            category=category,
            prefix=prefix,
        ) as checkout,
    ):
        yield checkout


@contextmanager
def _materialize_isolated_git_checkout(
    repo: _IsolatedGitRepository,
    commit: str,
    *,
    category: str,
    prefix: str,
) -> Iterator[Path]:
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
    repo: _IsolatedGitRepository,
    *arguments: str,
    category: str,
    max_stdout: int,
    deadline: float,
) -> bytes:
    result = _run_git_process_bounded(
        repo,
        *arguments,
        category=category,
        max_stdout=max_stdout,
        deadline=deadline,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise AtomicInfrastructureError(
            category,
            f"git {arguments[0]} failed" + (f": {detail}" if detail else ""),
        )
    return result.stdout


def _run_git_process_bounded(
    repo: _IsolatedGitRepository,
    *arguments: str,
    category: str,
    max_stdout: int,
    deadline: float,
) -> _GitProcessResult:
    command = [
        "git",
        f"--git-dir={repo.git_dir}",
        f"--work-tree={repo.work_tree}",
        *arguments,
    ]
    if time.monotonic() >= deadline:
        raise AtomicInfrastructureError(category, "Git operation timed out")
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_git_environment(repo),
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
                raise AtomicInfrastructureError(category, "Git operation timed out")
            ready = selector.select(remaining)
            if not ready:
                raise AtomicInfrastructureError(category, "Git operation timed out")
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
            raise AtomicInfrastructureError(category, "Git operation timed out")
        try:
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            raise AtomicInfrastructureError(
                category,
                "Git operation timed out",
            ) from exc
        return _GitProcessResult(
            returncode=returncode,
            stdout=bytes(stdout),
            stderr=bytes(stderr),
        )
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


@contextmanager
def _isolated_git_repository(
    repo: Path,
    *,
    category: str,
) -> Iterator[_IsolatedGitRepository]:
    deadline = time.monotonic() + _GIT_OPERATION_TIMEOUT_SECONDS
    _check_git_operation_deadline(deadline, category=category)
    work_tree = repo.resolve(strict=True)
    _check_git_operation_deadline(deadline, category=category)
    source_git_dir = _resolve_git_directory(
        work_tree,
        category=category,
        deadline=deadline,
    )
    common_dir_file = source_git_dir / "commondir"
    if common_dir_file.exists():
        common_dir_value = (
            _read_regular_file_bounded(
                common_dir_file,
                max_bytes=4096,
                category=category,
                label="Git commondir",
                deadline=deadline,
            )
            .decode("utf-8", errors="strict")
            .strip()
        )
        if not common_dir_value:
            raise AtomicInfrastructureError(category, "Git commondir is empty")
        common_dir = (source_git_dir / common_dir_value).resolve(strict=True)
    else:
        common_dir = source_git_dir
    _check_git_operation_deadline(deadline, category=category)
    if not common_dir.is_dir():
        raise AtomicInfrastructureError(category, "Git common directory is not a directory")

    source_config = _read_regular_file_bounded(
        common_dir / "config",
        max_bytes=_MAX_GIT_CONTROL_FILE_BYTES,
        category=category,
        label="Git config",
        deadline=deadline,
    )
    head = _read_regular_file_bounded(
        source_git_dir / "HEAD",
        max_bytes=4096,
        category=category,
        label="Git HEAD",
        deadline=deadline,
    )
    index = _read_regular_file_bounded(
        source_git_dir / "index",
        max_bytes=_MAX_GIT_INDEX_BYTES,
        category=category,
        label="Git index",
        deadline=deadline,
    )
    config = _build_isolated_git_config(source_config, category=category)
    _check_git_operation_deadline(deadline, category=category)

    with _PrivateTemporaryDirectory(prefix="ale-git-view-") as temp_dir:
        _check_git_operation_deadline(deadline, category=category)
        git_dir = Path(temp_dir) / "git"
        git_dir.mkdir(mode=0o700)
        (git_dir / "refs").mkdir(mode=0o700)
        (git_dir / "info").mkdir(mode=0o700)
        (git_dir / "config").write_text(config, encoding="utf-8")
        (git_dir / "HEAD").write_bytes(head)
        (git_dir / "index").write_bytes(index)
        _check_git_operation_deadline(deadline, category=category)

        objects = (common_dir / "objects").resolve(strict=True)
        _check_git_operation_deadline(deadline, category=category)
        if not objects.is_dir():
            raise AtomicInfrastructureError(category, "Git object directory is not a directory")
        os.symlink(objects, git_dir / "objects", target_is_directory=True)

        head_namespace = _head_ref_namespace(head, category=category)
        namespaces = {
            "heads",
            "tags",
            "remotes",
            "notes",
            "bisect",
            "rewritten",
            "worktree",
        }
        if head_namespace is not None:
            namespaces.add(head_namespace)
        for namespace in sorted(namespaces):
            _check_git_operation_deadline(deadline, category=category)
            source_namespace = source_git_dir / "refs" / namespace
            if not source_namespace.exists():
                source_namespace = common_dir / "refs" / namespace
            if source_namespace.exists():
                resolved_namespace = source_namespace.resolve(strict=True)
                if not resolved_namespace.is_dir():
                    raise AtomicInfrastructureError(
                        category,
                        f"Git ref namespace is not a directory: {namespace}",
                    )
                os.symlink(
                    resolved_namespace,
                    git_dir / "refs" / namespace,
                    target_is_directory=True,
                )
                _check_git_operation_deadline(deadline, category=category)

        for name, max_bytes in (
            ("packed-refs", _MAX_GIT_CONTROL_FILE_BYTES),
            ("shallow", _MAX_GIT_CONTROL_FILE_BYTES),
        ):
            _check_git_operation_deadline(deadline, category=category)
            source = common_dir / name
            if source.exists():
                (git_dir / name).write_bytes(
                    _read_regular_file_bounded(
                        source,
                        max_bytes=max_bytes,
                        category=category,
                        label=f"Git {name}",
                        deadline=deadline,
                    )
                )
                _check_git_operation_deadline(deadline, category=category)

        exclude = common_dir / "info" / "exclude"
        if exclude.exists():
            (git_dir / "info" / "exclude").write_bytes(
                _read_regular_file_bounded(
                    exclude,
                    max_bytes=_MAX_GIT_CONTROL_FILE_BYTES,
                    category=category,
                    label="Git exclude file",
                    deadline=deadline,
                )
            )
            _check_git_operation_deadline(deadline, category=category)

        shared_index_directories = [source_git_dir]
        if common_dir != source_git_dir:
            shared_index_directories.append(common_dir)
        _copy_shared_indexes_bounded(
            shared_index_directories,
            git_dir,
            category=category,
            deadline=deadline,
        )

        reftable = common_dir / "reftable"
        if "refstorage = reftable" in config and reftable.exists():
            resolved_reftable = reftable.resolve(strict=True)
            if not resolved_reftable.is_dir():
                raise AtomicInfrastructureError(category, "Git reftable is not a directory")
            os.symlink(resolved_reftable, git_dir / "reftable", target_is_directory=True)
        _check_git_operation_deadline(deadline, category=category)

        yield _IsolatedGitRepository(work_tree=work_tree, git_dir=git_dir)


def _resolve_git_directory(
    work_tree: Path,
    *,
    category: str,
    deadline: float,
) -> Path:
    _check_git_operation_deadline(deadline, category=category)
    dot_git = work_tree / ".git"
    try:
        dot_git_status = dot_git.lstat()
    except OSError as exc:
        raise AtomicInfrastructureError(category, f"cannot inspect Git directory: {exc}") from exc
    if stat.S_ISDIR(dot_git_status.st_mode):
        resolved = dot_git.resolve(strict=True)
        _check_git_operation_deadline(deadline, category=category)
        return resolved
    if not stat.S_ISREG(dot_git_status.st_mode):
        raise AtomicInfrastructureError(category, "Git metadata pointer is not a regular file")
    git_file = _read_regular_file_bounded(
        dot_git,
        max_bytes=4096,
        category=category,
        label="Git metadata pointer",
        deadline=deadline,
    ).decode("utf-8", errors="strict")
    prefix = "gitdir: "
    if not git_file.startswith(prefix) or "\n" in git_file.strip():
        raise AtomicInfrastructureError(category, "invalid Git metadata pointer")
    target = git_file[len(prefix) :].strip()
    if not target:
        raise AtomicInfrastructureError(category, "Git metadata pointer is empty")
    git_dir = (work_tree / target).resolve(strict=True)
    _check_git_operation_deadline(deadline, category=category)
    if not git_dir.is_dir():
        raise AtomicInfrastructureError(category, "Git metadata target is not a directory")
    return git_dir


def _read_regular_file_bounded(
    path: Path,
    *,
    max_bytes: int,
    category: str,
    label: str,
    deadline: float,
) -> bytes:
    _check_git_operation_deadline(deadline, category=category)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        file_status = os.fstat(descriptor)
        if not stat.S_ISREG(file_status.st_mode):
            raise AtomicInfrastructureError(category, f"{label} is not a regular file")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            payload = stream.read(max_bytes + 1)
    except AtomicInfrastructureError:
        raise
    except OSError as exc:
        raise AtomicInfrastructureError(category, f"cannot read {label}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if len(payload) > max_bytes:
        raise AtomicInfrastructureError(category, f"{label} exceeds its size limit")
    _check_git_operation_deadline(deadline, category=category)
    return payload


def _copy_shared_indexes_bounded(
    source_directories: list[Path],
    destination: Path,
    *,
    category: str,
    deadline: float,
) -> None:
    destination_fd: int | None = None
    seen: set[str] = set()
    matched = 0
    try:
        destination_fd = os.open(
            destination,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        for source_directory in source_directories:
            _check_git_operation_deadline(deadline, category=category)
            source_fd: int | None = None
            try:
                source_fd = os.open(
                    source_directory,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                )
                with os.scandir(source_fd) as entries:
                    for entry in entries:
                        _check_git_operation_deadline(deadline, category=category)
                        if not entry.name.startswith("sharedindex."):
                            continue
                        matched += 1
                        if matched > _MAX_GIT_SHARED_INDEXES:
                            raise AtomicInfrastructureError(
                                category,
                                "Git split index has too many shared files",
                            )
                        _validate_shared_index_name(entry.name, category=category)
                        if entry.name in seen:
                            raise AtomicInfrastructureError(
                                category,
                                f"duplicate Git shared index: {entry.name}",
                            )
                        seen.add(entry.name)
                        if not entry.is_file(follow_symlinks=False):
                            raise AtomicInfrastructureError(
                                category,
                                "Git shared index is not a regular file",
                            )
                        payload = _read_regular_file_at_bounded(
                            source_fd,
                            entry.name,
                            max_bytes=_MAX_GIT_INDEX_BYTES,
                            category=category,
                            label="Git shared index",
                            deadline=deadline,
                        )
                        _write_private_file_at(
                            destination_fd,
                            entry.name,
                            payload,
                            category=category,
                            deadline=deadline,
                        )
            except AtomicInfrastructureError:
                raise
            except OSError as exc:
                raise AtomicInfrastructureError(
                    category,
                    f"cannot inspect Git shared indexes: {exc}",
                ) from exc
            finally:
                if source_fd is not None:
                    os.close(source_fd)
    except AtomicInfrastructureError:
        raise
    except OSError as exc:
        raise AtomicInfrastructureError(
            category,
            f"cannot prepare Git shared indexes: {exc}",
        ) from exc
    finally:
        if destination_fd is not None:
            os.close(destination_fd)


def _validate_shared_index_name(name: str, *, category: str) -> None:
    prefix = "sharedindex."
    object_id = name.removeprefix(prefix)
    if (
        not name.startswith(prefix)
        or len(object_id) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in object_id)
    ):
        raise AtomicInfrastructureError(category, "Git shared index has an unsafe name")


def _read_regular_file_at_bounded(
    directory_fd: int,
    name: str,
    *,
    max_bytes: int,
    category: str,
    label: str,
    deadline: float,
) -> bytes:
    _check_git_operation_deadline(deadline, category=category)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory_fd,
        )
        file_status = os.fstat(descriptor)
        if not stat.S_ISREG(file_status.st_mode):
            raise AtomicInfrastructureError(category, f"{label} is not a regular file")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            payload = stream.read(max_bytes + 1)
    except AtomicInfrastructureError:
        raise
    except OSError as exc:
        raise AtomicInfrastructureError(category, f"cannot read {label}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if len(payload) > max_bytes:
        raise AtomicInfrastructureError(category, f"{label} exceeds its size limit")
    _check_git_operation_deadline(deadline, category=category)
    return payload


def _write_private_file_at(
    directory_fd: int,
    name: str,
    payload: bytes,
    *,
    category: str,
    deadline: float,
) -> None:
    _check_git_operation_deadline(deadline, category=category)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(payload)
    except OSError as exc:
        raise AtomicInfrastructureError(
            category,
            f"cannot copy Git shared index: {exc}",
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    _check_git_operation_deadline(deadline, category=category)


def _check_git_operation_deadline(deadline: float, *, category: str) -> None:
    if time.monotonic() >= deadline:
        raise AtomicInfrastructureError(category, "Git operation timed out")


def _build_isolated_git_config(source: bytes, *, category: str) -> str:
    try:
        lines = source.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as exc:
        raise AtomicInfrastructureError(category, "Git config is not UTF-8") from exc
    section = ""
    values: dict[tuple[str, str], str] = {}
    allowed = {
        ("core", "repositoryformatversion"),
        ("core", "filemode"),
        ("core", "ignorecase"),
        ("core", "symlinks"),
        ("core", "precomposeunicode"),
        ("extensions", "objectformat"),
        ("extensions", "compatobjectformat"),
        ("extensions", "refstorage"),
    }
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";")):
            continue
        if stripped.startswith("[") and "]" in stripped:
            section = stripped[1 : stripped.index("]")].split(maxsplit=1)[0].lower()
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        identity = (section, key.strip().lower())
        if identity in allowed:
            values[identity] = value.strip().strip('"')

    repository_format = values.get(("core", "repositoryformatversion"), "0")
    if repository_format not in {"0", "1"}:
        raise AtomicInfrastructureError(category, "unsupported Git repository format")
    object_format = values.get(("extensions", "objectformat"), "sha1").lower()
    if object_format not in {"sha1", "sha256"}:
        raise AtomicInfrastructureError(category, "unsupported Git object format")
    ref_storage = values.get(("extensions", "refstorage"), "files").lower()
    if ref_storage not in {"files", "reftable"}:
        raise AtomicInfrastructureError(category, "unsupported Git ref storage")

    config_lines = [
        "[core]",
        f"\trepositoryformatversion = {repository_format}",
        "\tbare = false",
    ]
    for key, default in (
        ("filemode", "true"),
        ("ignorecase", "false"),
        ("symlinks", "true"),
        ("precomposeunicode", "false"),
    ):
        value = values.get(("core", key), default).lower()
        if value not in {"true", "false", "yes", "no", "on", "off", "1", "0"}:
            raise AtomicInfrastructureError(category, f"invalid passive Git config: core.{key}")
        config_lines.append(f"\t{key} = {value}")
    if repository_format == "1" or object_format != "sha1" or ref_storage != "files":
        config_lines.append("[extensions]")
        if object_format != "sha1":
            config_lines.append(f"\tobjectformat = {object_format}")
        compat_object_format = values.get(("extensions", "compatobjectformat"))
        if compat_object_format is not None:
            compat_object_format = compat_object_format.lower()
            if compat_object_format not in {"sha1", "sha256"}:
                raise AtomicInfrastructureError(category, "unsupported compatible object format")
            config_lines.append(f"\tcompatobjectformat = {compat_object_format}")
        if ref_storage != "files":
            config_lines.append(f"\trefstorage = {ref_storage}")
    return "\n".join(config_lines) + "\n"


def _head_ref_namespace(head: bytes, *, category: str) -> str | None:
    try:
        value = head.decode("ascii", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise AtomicInfrastructureError(category, "Git HEAD is not ASCII") from exc
    if not value.startswith("ref: "):
        return None
    ref = PurePosixPath(value.removeprefix("ref: "))
    if (
        len(ref.parts) < 3
        or ref.parts[0] != "refs"
        or any(part in {"", ".", ".."} for part in ref.parts)
        or ref.parts[1] == "replace"
    ):
        raise AtomicInfrastructureError(category, "Git HEAD contains an unsafe ref")
    return ref.parts[1]


def _run_git(
    repo: _IsolatedGitRepository,
    *arguments: str,
    category: str = "evaluator_registry",
) -> subprocess.CompletedProcess[str]:
    result = _run_git_process_bounded(
        repo,
        *arguments,
        category=category,
        max_stdout=_MAX_GIT_COMMAND_BYTES,
        deadline=time.monotonic() + _GIT_OPERATION_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        detail = (
            (result.stderr or result.stdout)
            .decode(
                "utf-8",
                errors="replace",
            )
            .strip()
        )
        raise AtomicInfrastructureError(
            category,
            f"git {' '.join(arguments)} failed: {detail[:500]}",
        )
    return subprocess.CompletedProcess(
        args=list(arguments),
        returncode=result.returncode,
        stdout=result.stdout.decode("utf-8", errors="strict"),
        stderr=result.stderr.decode("utf-8", errors="replace"),
    )


def _read_git_status(
    repo: _IsolatedGitRepository,
    *,
    category: str,
    include_ignored: bool,
) -> bytes:
    arguments = [
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    ]
    if include_ignored:
        arguments.append("--ignored=matching")
    result = _run_git_process_bounded(
        repo,
        *arguments,
        category=category,
        max_stdout=_MAX_GIT_STATUS_BYTES,
        deadline=time.monotonic() + _GIT_OPERATION_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise AtomicInfrastructureError(
            category,
            "git status failed" + (f": {detail[:500]}" if detail else ""),
        )
    return result.stdout


def _git_environment(repo: _IsolatedGitRepository) -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(
        {
            "GIT_DIR": str(repo.git_dir),
            "GIT_WORK_TREE": str(repo.work_tree),
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
