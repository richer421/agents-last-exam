"""Trusted evaluator registry validation and immutable Git materialization."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import tarfile
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from pydantic import ValidationError

from .contracts import (
    AtomicInfrastructureError,
    EvaluateRequest,
    EvaluatorRegistryRecord,
)

_MAX_REGISTRY_RECORD_BYTES = 1024 * 1024
_MAX_GIT_ARCHIVE_BYTES = 64 * 1024 * 1024
_MAX_GIT_CHECKOUT_BYTES = 32 * 1024 * 1024
_MAX_GIT_FILE_BYTES = 8 * 1024 * 1024
_MAX_GIT_FILES = 4096
_MAX_GIT_STATUS_BYTES = 1024 * 1024


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

    if _read_git_status(request.task_repo, category="evaluator_registry"):
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
    if _read_git_status(repo, category=category):
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
        resolved_commit = _run_git(
            repo,
            "rev-parse",
            "--verify",
            f"{commit}^{{commit}}",
            category=category,
        ).stdout.strip()
        if resolved_commit != commit:
            raise AtomicInfrastructureError(
                category,
                f"Git commit mismatch: expected {commit}, got {resolved_commit!r}",
            )
        root = Path(temp_dir)
        archive = root / "checkout.tar"
        checkout = root / "checkout"
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "archive",
                "--format=tar",
                "-o",
                str(archive),
                resolved_commit,
            ],
            capture_output=True,
            check=False,
            text=True,
            env=_git_environment(),
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise AtomicInfrastructureError(
                category,
                f"cannot archive Git commit: {detail[:500]}",
            )
        if archive.stat().st_size > _MAX_GIT_ARCHIVE_BYTES:
            raise AtomicInfrastructureError(
                category,
                "Git archive exceeds the 64 MiB limit",
            )
        checkout.mkdir(mode=0o700)
        try:
            with tarfile.open(archive, mode="r:") as bundle:
                members = bundle.getmembers()
                if len(members) > _MAX_GIT_FILES:
                    raise AtomicInfrastructureError(
                        category,
                        f"Git checkout exceeds {_MAX_GIT_FILES} entries",
                    )
                source_bytes = 0
                for member in members:
                    path = PurePosixPath(member.name)
                    if (
                        path.is_absolute()
                        or any(part in {"", ".", ".."} for part in path.parts)
                        or member.issym()
                        or member.islnk()
                        or not (member.isfile() or member.isdir())
                    ):
                        raise AtomicInfrastructureError(
                            category,
                            f"unsafe Git archive member: {member.name}",
                        )
                    if member.isfile():
                        if member.size > _MAX_GIT_FILE_BYTES:
                            raise AtomicInfrastructureError(
                                category,
                                f"Git file exceeds 8 MiB: {member.name}",
                            )
                        source_bytes += member.size
                        if source_bytes > _MAX_GIT_CHECKOUT_BYTES:
                            raise AtomicInfrastructureError(
                                category,
                                "Git checkout exceeds 32 MiB of files",
                            )
                bundle.extractall(checkout, members=members, filter="data")
        except AtomicInfrastructureError:
            raise
        except (OSError, tarfile.TarError) as exc:
            raise AtomicInfrastructureError(
                category,
                f"cannot materialize Git archive: {exc}",
            ) from exc
        yield checkout


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


def _read_git_status(repo: Path, *, category: str) -> bytes:
    with tempfile.TemporaryFile() as stderr:
        try:
            process = subprocess.Popen(
                [
                    "git",
                    "-C",
                    str(repo),
                    "status",
                    "--porcelain=v1",
                    "-z",
                    "--untracked-files=all",
                    "--ignored=matching",
                ],
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
