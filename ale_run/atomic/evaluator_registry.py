"""Trusted evaluator registry validation and immutable Git materialization."""

from __future__ import annotations

import hashlib
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
_MAX_EVALUATOR_ARCHIVE_BYTES = 64 * 1024 * 1024
_MAX_EVALUATOR_SOURCE_BYTES = 32 * 1024 * 1024
_MAX_EVALUATOR_FILE_BYTES = 8 * 1024 * 1024
_MAX_EVALUATOR_FILES = 4096


def validate_evaluator_registry(
    request: EvaluateRequest,
) -> EvaluatorRegistryRecord:
    """Fail closed unless the external ready record authorizes this checkout."""
    record_path = _external_record_path(request)
    try:
        with record_path.open("rb") as stream:
            raw = stream.read(_MAX_REGISTRY_RECORD_BYTES + 1)
    except OSError as exc:
        raise AtomicInfrastructureError(
            "evaluator_registry",
            f"cannot read evaluator registry record: {exc}",
        ) from exc
    if len(raw) > _MAX_REGISTRY_RECORD_BYTES:
        raise AtomicInfrastructureError(
            "evaluator_registry",
            "evaluator registry record exceeds the 1 MiB limit",
        )
    if hashlib.sha256(raw).hexdigest() != request.evaluator_registry_record_sha256:
        raise AtomicInfrastructureError(
            "evaluator_registry",
            "evaluator registry record SHA-256 mismatch",
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

    status = _run_git(
        request.task_repo,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status.stdout:
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
    with tempfile.TemporaryDirectory(prefix="ale-evaluator-checkout-") as temp_dir:
        root = Path(temp_dir)
        archive = root / "evaluator.tar"
        checkout = root / "checkout"
        result = subprocess.run(
            [
                "git",
                "-C",
                str(request.task_repo),
                "archive",
                "--format=tar",
                "-o",
                str(archive),
                request.evaluator_version,
            ],
            capture_output=True,
            check=False,
            text=True,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise AtomicInfrastructureError(
                "evaluator_registry",
                f"cannot archive evaluator commit: {detail[:500]}",
            )
        if archive.stat().st_size > _MAX_EVALUATOR_ARCHIVE_BYTES:
            raise AtomicInfrastructureError(
                "evaluator_registry",
                "evaluator Git archive exceeds the 64 MiB limit",
            )
        checkout.mkdir(mode=0o700)
        try:
            with tarfile.open(archive, mode="r:") as bundle:
                members = bundle.getmembers()
                if len(members) > _MAX_EVALUATOR_FILES:
                    raise AtomicInfrastructureError(
                        "evaluator_registry",
                        f"evaluator checkout exceeds {_MAX_EVALUATOR_FILES} entries",
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
                            "evaluator_registry",
                            f"unsafe evaluator archive member: {member.name}",
                        )
                    if member.isfile():
                        if member.size > _MAX_EVALUATOR_FILE_BYTES:
                            raise AtomicInfrastructureError(
                                "evaluator_registry",
                                f"evaluator file exceeds 8 MiB: {member.name}",
                            )
                        source_bytes += member.size
                        if source_bytes > _MAX_EVALUATOR_SOURCE_BYTES:
                            raise AtomicInfrastructureError(
                                "evaluator_registry",
                                "evaluator checkout exceeds 32 MiB of files",
                            )
                bundle.extractall(checkout, members=members, filter="data")
        except AtomicInfrastructureError:
            raise
        except (OSError, tarfile.TarError) as exc:
            raise AtomicInfrastructureError(
                "evaluator_registry",
                f"cannot materialize evaluator archive: {exc}",
            ) from exc
        task_dir = checkout / "tasks" / request.task_path
        if not task_dir.is_dir():
            raise AtomicInfrastructureError(
                "evaluator_registry",
                "evaluator commit does not contain the requested task",
            )
        yield checkout


def _external_record_path(request: EvaluateRequest) -> Path:
    try:
        repo = request.task_repo.resolve(strict=True)
        raw = request.evaluator_registry_record_path
        if raw.is_symlink():
            raise AtomicInfrastructureError(
                "evaluator_registry",
                "evaluator registry record path must not be a symlink",
            )
        record = raw.resolve(strict=True)
    except AtomicInfrastructureError:
        raise
    except OSError as exc:
        raise AtomicInfrastructureError(
            "evaluator_registry",
            f"cannot resolve evaluator registry boundary: {exc}",
        ) from exc
    if record == repo or record.is_relative_to(repo):
        raise AtomicInfrastructureError(
            "evaluator_registry",
            "evaluator registry record must live outside the task checkout",
        )
    if not record.is_file():
        raise AtomicInfrastructureError(
            "evaluator_registry",
            "evaluator registry record is not a regular file",
        )
    return record


def _run_git(repo: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise AtomicInfrastructureError(
            "evaluator_registry",
            f"git {' '.join(arguments)} failed: {detail[:500]}",
        )
    return result
