"""Host-mediated task-data staging for atomic capabilities."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shlex
import stat
import tempfile
import zipfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from uuid import uuid4

from pydantic import ValidationError

from ..base_interface import SandboxHandle, TaskDataSpec
from ..environments import task_data as task_data_pkg
from ..environments.task_data import join, task_subdir
from ..orchestration.experiment_spec import EnvironmentSpec
from .contracts import AtomicInfrastructureError, EvaluatorRegistryRecord, ReferenceManifest
from .host_oss import (
    host_command_diagnostic,
    read_host_oss_object,
    run_host_ossutil,
)

_MAX_STAGED_FILES = 4096
_MAX_STAGED_FILE_BYTES = 512 * 1024 * 1024
_MAX_STAGED_SOURCE_BYTES = 1024 * 1024 * 1024
_MAX_STAGED_ARCHIVE_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True)
class PreparedInput:
    archive_path: Path
    archive_sha256: str


@dataclass(frozen=True)
class PreparedReference:
    archive_path: Path | None
    archive_sha256: str | None
    manifest: ReferenceManifest


def sanitize_solve_environment(environment: EnvironmentSpec) -> EnvironmentSpec:
    """Clone an environment and remove instance identities capable of OSS access."""
    sanitized = copy.deepcopy(environment)
    for provider in sanitized.provider_specs.values():
        if provider.kind == "aliyun":
            provider.config["ram_role_name"] = ""
            provider.config["output_to_bucket"] = False
    return sanitized


@asynccontextmanager
async def prepare_atomic_input(
    *,
    source: str,
    task_data: TaskDataSpec,
    declared_input_paths: tuple[str, ...],
) -> AsyncIterator[PreparedInput | None]:
    """Download remote task data on the host before any VM receives it."""
    _validate_declared_input_paths(declared_input_paths)
    if not task_data.requires_task_data or source == "baked_in_sandbox":
        yield None
        return
    if not source.startswith("oss://"):
        raise AtomicInfrastructureError(
            "input",
            f"unsupported atomic remote task-data backend: {source.split(':', 1)[0]}",
        )

    with tempfile.TemporaryDirectory(prefix="ale-atomic-input-") as temp_dir:
        root = Path(temp_dir)
        os.chmod(root, 0o700)
        payload = root / "payload"
        payload.mkdir(mode=0o700)
        prefix = (
            f"{source.rstrip('/')}/{task_data.domain_name}/"
            f"{task_data.task_name}/{task_data.variant_name}"
        )
        await _download_prefix(
            f"{prefix}/input/",
            payload / "input",
            required=True,
        )
        await _download_prefix(
            f"{prefix}/software/",
            payload / "software",
            required=False,
        )
        _audit_staged_tree(payload, declared_input_paths)
        archive = root / "input.zip"
        _build_archive(payload, archive, category="input")
        yield PreparedInput(
            archive_path=archive,
            archive_sha256=_file_sha256(archive),
        )


@asynccontextmanager
async def prepare_atomic_reference(
    *,
    record: EvaluatorRegistryRecord,
    source: str,
    task_data: TaskDataSpec,
    declared_reference_paths: tuple[str, ...],
) -> AsyncIterator[PreparedReference]:
    """Verify the registry manifest and remote reference bytes on the host."""
    declared = tuple(_normalize_reference_path(path) for path in declared_reference_paths)
    raw = await read_host_oss_object(
        record.reference_manifest_uri,
        limit=1024 * 1024,
        missing_ok=False,
        integrity_category="reference",
    )
    assert raw is not None
    if hashlib.sha256(raw).hexdigest() != record.reference_manifest_hash:
        raise AtomicInfrastructureError(
            "reference",
            "reference manifest SHA-256 mismatch",
        )
    try:
        manifest = ReferenceManifest.model_validate_json(raw)
    except ValidationError as exc:
        raise AtomicInfrastructureError(
            "reference",
            f"invalid reference manifest: {exc}",
        ) from exc
    normalized_entries: list[str] = []
    for entry in manifest.files:
        path = _normalize_reference_path(entry.path)
        if path in normalized_entries:
            raise AtomicInfrastructureError(
                "reference",
                f"duplicate reference manifest path: {path}",
            )
        normalized_entries.append(path)
    missing = [path for path in declared if path not in normalized_entries]
    if missing:
        raise AtomicInfrastructureError(
            "reference",
            f"reference manifest omits declared files: {', '.join(missing)}",
        )
    if source == "baked_in_sandbox":
        yield PreparedReference(
            archive_path=None,
            archive_sha256=None,
            manifest=manifest,
        )
        return
    if not source.startswith("oss://"):
        raise AtomicInfrastructureError(
            "reference",
            f"unsupported atomic remote reference backend: {source.split(':', 1)[0]}",
        )

    with tempfile.TemporaryDirectory(prefix="ale-atomic-reference-") as temp_dir:
        root = Path(temp_dir)
        os.chmod(root, 0o700)
        payload = root / "payload"
        payload.mkdir(mode=0o700)
        object_root = record.reference_manifest_uri.rsplit("/", 1)[0]
        total = 0
        for entry, normalized in zip(manifest.files, normalized_entries, strict=True):
            destination = payload.joinpath(*PurePosixPath(normalized).parts)
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            object_name = PurePosixPath(normalized).relative_to("reference").as_posix()
            object_url = f"{object_root}/{object_name}"
            stat_result = await run_host_ossutil("stat", object_url)
            if stat_result[0] != 0:
                raise AtomicInfrastructureError(
                    "reference",
                    f"reference object is missing: {normalized}",
                )
            stat_output = stat_result[1].decode("utf-8", errors="replace")
            import re

            size_match = re.search(
                r"(?im)^\s*(?:content[- ]?length|size)\s*[:=]\s*(\d+)\s*$",
                stat_output,
            )
            if size_match is None or int(size_match.group(1)) != entry.size_bytes:
                raise AtomicInfrastructureError(
                    "reference",
                    f"reference size mismatch: {normalized}",
                )
            if entry.size_bytes > _MAX_STAGED_FILE_BYTES:
                raise AtomicInfrastructureError(
                    "reference",
                    f"reference file exceeds 512 MiB: {normalized}",
                )
            total += entry.size_bytes
            if total > _MAX_STAGED_SOURCE_BYTES:
                raise AtomicInfrastructureError(
                    "reference",
                    "reference files exceed the total-size limit",
                )
            downloaded = await run_host_ossutil(
                "cp",
                object_url,
                str(destination),
                "-f",
            )
            if downloaded[0] != 0:
                raise AtomicInfrastructureError(
                    "reference",
                    f"reference download failed: {normalized}",
                )
            if (
                destination.is_symlink()
                or not destination.is_file()
                or destination.stat().st_size != entry.size_bytes
                or _file_sha256(destination) != entry.sha256
            ):
                raise AtomicInfrastructureError(
                    "reference",
                    f"reference hash or size mismatch: {normalized}",
                )
            os.chmod(destination, 0o600)
        archive = root / "reference.zip"
        _build_archive(payload, archive, category="reference")
        yield PreparedReference(
            archive_path=archive,
            archive_sha256=_file_sha256(archive),
            manifest=manifest,
        )


async def stage_atomic_input(
    sandbox: SandboxHandle,
    task_data: TaskDataSpec,
    *,
    source: str,
    declared_input_paths: tuple[str, ...],
    prepared: PreparedInput | None,
) -> None:
    """Transfer one trusted archive, or verify baked input, inside the VM."""
    if not task_data.requires_task_data:
        return
    _validate_declared_input_paths(declared_input_paths)
    if source == "baked_in_sandbox":
        backend = task_data_pkg.select(source)
        try:
            report = await backend.stage_input(sandbox, task_data, source=source)
        except Exception as exc:
            raise AtomicInfrastructureError(
                "input",
                f"baked input staging failed: {type(exc).__name__}: {exc}"[:2000],
            ) from exc
        if not isinstance(report, dict) or report.get("skipped"):
            raise AtomicInfrastructureError(
                "input",
                "baked input backend did not confirm complete staging",
            )
    elif source.startswith("oss://"):
        if prepared is None:
            raise AtomicInfrastructureError(
                "input",
                "trusted host input archive is missing",
            )
    else:
        raise AtomicInfrastructureError(
            "input",
            f"unsupported atomic remote task-data backend: {source.split(':', 1)[0]}",
        )

    base = task_subdir(sandbox, task_data)
    await sandbox.mkdir(base)
    nonce = uuid4().hex
    archive_path = join(sandbox, base, f".ale-input-{nonce}.zip")
    script_path = join(sandbox, base, f".ale-input-{nonce}.py")
    config_path = join(sandbox, base, f".ale-input-{nonce}.json")
    archive_sha256: str | None = None
    if prepared is not None:
        await sandbox.upload_local_file(str(prepared.archive_path), archive_path)
        archive_sha256 = prepared.archive_sha256
    config = {
        "archive_path": archive_path if prepared is not None else None,
        "archive_sha256": archive_sha256,
        "base": base,
        "declared_input_paths": declared_input_paths,
        "max_files": _MAX_STAGED_FILES,
        "max_file_bytes": _MAX_STAGED_FILE_BYTES,
        "max_source_bytes": _MAX_STAGED_SOURCE_BYTES,
    }
    await sandbox.write_file(script_path, _STAGE_SCRIPT.encode())
    await sandbox.write_file(
        config_path,
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode(),
    )
    python = sandbox.python or ("python3" if sandbox.is_linux else "python")
    command = (
        f"{shlex.quote(python)} {shlex.quote(script_path)} {shlex.quote(config_path)}"
        if sandbox.is_linux
        else f'"{python}" "{script_path}" "{config_path}"'
    )
    result = await sandbox.run_command(command, timeout=600)
    try:
        payload = json.loads(result.stdout or "")
    except (TypeError, json.JSONDecodeError) as exc:
        raise AtomicInfrastructureError(
            "input",
            "VM input verifier returned invalid JSON",
        ) from exc
    if result.returncode != 0 or not isinstance(payload, dict) or payload.get("ok") is not True:
        detail = payload.get("error") if isinstance(payload, dict) else None
        raise AtomicInfrastructureError(
            "input",
            str(detail or result.stderr or "VM input verifier failed")[:2000],
        )


async def stage_atomic_reference(
    sandbox: SandboxHandle,
    task_data: TaskDataSpec,
    *,
    source: str,
    declared_reference_paths: tuple[str, ...],
    prepared: PreparedReference,
) -> None:
    """Stage reference bytes and verify registry size/hash inside the VM."""
    declared = tuple(_normalize_reference_path(path) for path in declared_reference_paths)
    if source == "baked_in_sandbox":
        backend = task_data_pkg.select(source)
        try:
            report = await backend.stage_reference(sandbox, task_data, source=source)
        except Exception as exc:
            raise AtomicInfrastructureError(
                "reference",
                f"baked reference staging failed: {type(exc).__name__}: {exc}"[:2000],
            ) from exc
        if not isinstance(report, dict) or report.get("skipped"):
            reason = (
                report.get("reason", "unknown") if isinstance(report, dict) else "invalid report"
            )
            raise AtomicInfrastructureError(
                "reference",
                f"baked reference staging was incomplete: {reason}",
            )
    elif source.startswith("oss://"):
        if prepared.archive_path is None or prepared.archive_sha256 is None:
            raise AtomicInfrastructureError(
                "reference",
                "trusted host reference archive is missing",
            )
    else:
        raise AtomicInfrastructureError(
            "reference",
            f"unsupported atomic remote reference backend: {source.split(':', 1)[0]}",
        )

    base = task_subdir(sandbox, task_data)
    await sandbox.mkdir(base)
    nonce = uuid4().hex
    archive_path = join(sandbox, base, f".ale-reference-{nonce}.zip")
    script_path = join(sandbox, base, f".ale-reference-{nonce}.py")
    config_path = join(sandbox, base, f".ale-reference-{nonce}.json")
    if prepared.archive_path is not None:
        await sandbox.upload_local_file(str(prepared.archive_path), archive_path)
    entries = []
    for entry in prepared.manifest.files:
        entries.append(
            {
                "path": _normalize_reference_path(entry.path),
                "size_bytes": entry.size_bytes,
                "sha256": entry.sha256,
            }
        )
    config = {
        "archive_path": archive_path if prepared.archive_path is not None else None,
        "archive_sha256": prepared.archive_sha256,
        "base": base,
        "declared_reference_paths": declared,
        "entries": entries,
        "max_files": _MAX_STAGED_FILES,
        "max_file_bytes": _MAX_STAGED_FILE_BYTES,
        "max_source_bytes": _MAX_STAGED_SOURCE_BYTES,
    }
    await sandbox.write_file(script_path, _REFERENCE_STAGE_SCRIPT.encode())
    await sandbox.write_file(
        config_path,
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode(),
    )
    python = sandbox.python or ("python3" if sandbox.is_linux else "python")
    command = (
        f"{shlex.quote(python)} {shlex.quote(script_path)} {shlex.quote(config_path)}"
        if sandbox.is_linux
        else f'"{python}" "{script_path}" "{config_path}"'
    )
    result = await sandbox.run_command(command, timeout=600)
    try:
        payload = json.loads(result.stdout or "")
    except (TypeError, json.JSONDecodeError) as exc:
        raise AtomicInfrastructureError(
            "reference",
            "VM reference verifier returned invalid JSON",
        ) from exc
    if result.returncode != 0 or not isinstance(payload, dict) or payload.get("ok") is not True:
        detail = payload.get("error") if isinstance(payload, dict) else None
        raise AtomicInfrastructureError(
            "reference",
            str(detail or result.stderr or "VM reference verifier failed")[:2000],
        )


async def _download_prefix(
    source: str,
    destination: Path,
    *,
    required: bool,
) -> None:
    listed = await run_host_ossutil(
        "ls",
        "--payer",
        "requester",
        source,
        "--limited-num",
        "1",
    )
    if listed[0] != 0:
        if required:
            raise AtomicInfrastructureError(
                "input",
                f"cannot list required OSS prefix: {host_command_diagnostic(listed)}",
            )
        return
    listing = listed[1].decode("utf-8", errors="replace")
    empty = "Object Number is: 0" in listing or not (
        "oss://" in listing or "Object Number is:" in listing
    )
    if empty:
        if required:
            raise AtomicInfrastructureError(
                "input",
                "required OSS input prefix is empty",
            )
        return
    destination.mkdir(parents=True, mode=0o700)
    synced = await run_host_ossutil(
        "sync",
        "--payer",
        "requester",
        source,
        str(destination),
    )
    if synced[0] != 0:
        raise AtomicInfrastructureError(
            "input",
            f"host OSS staging failed: {host_command_diagnostic(synced)}",
        )


def _audit_staged_tree(
    payload: Path,
    declared_input_paths: tuple[str, ...],
) -> None:
    files = 0
    total = 0
    for root, dirnames, filenames in os.walk(payload, followlinks=False):
        root_path = Path(root)
        for name in (*dirnames, *filenames):
            if (root_path / name).is_symlink():
                raise AtomicInfrastructureError(
                    "input",
                    f"staged task data contains a symlink: {(root_path / name).relative_to(payload)}",
                )
        for name in filenames:
            path = root_path / name
            size = path.stat().st_size
            if size > _MAX_STAGED_FILE_BYTES:
                raise AtomicInfrastructureError(
                    "input",
                    f"staged input file exceeds 512 MiB: {path.relative_to(payload)}",
                )
            files += 1
            total += size
            if files > _MAX_STAGED_FILES or total > _MAX_STAGED_SOURCE_BYTES:
                raise AtomicInfrastructureError(
                    "input",
                    "staged task data exceeds file-count or total-size limits",
                )
    for declared in declared_input_paths:
        if not (payload / PurePosixPath(declared)).is_file():
            raise AtomicInfrastructureError(
                "input",
                f"declared input file is missing after host staging: {declared}",
            )


def _build_archive(payload: Path, archive: Path, *, category: str) -> None:
    with zipfile.ZipFile(
        archive,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as bundle:
        for path in sorted(candidate for candidate in payload.rglob("*") if candidate.is_file()):
            relative = path.relative_to(payload).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            bundle.writestr(info, path.read_bytes())
    if archive.stat().st_size > _MAX_STAGED_ARCHIVE_BYTES:
        raise AtomicInfrastructureError(
            category,
            "trusted input archive exceeds 512 MiB",
        )


def _validate_declared_input_paths(paths: tuple[str, ...]) -> None:
    for raw in paths:
        path = PurePosixPath(raw)
        if (
            not raw
            or path.is_absolute()
            or len(path.parts) < 2
            or path.parts[0] != "input"
            or any(part in {"", ".", ".."} for part in path.parts)
            or path.as_posix() != raw
        ):
            raise AtomicInfrastructureError(
                "input",
                f"unsafe declared input path: {raw!r}",
            )


def _normalize_reference_path(raw: str) -> str:
    path = PurePosixPath(raw)
    if (
        not raw
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != raw
    ):
        raise AtomicInfrastructureError(
            "reference",
            f"unsafe declared reference path: {raw!r}",
        )
    if path.parts[0] == "reference":
        if len(path.parts) < 2:
            raise AtomicInfrastructureError(
                "reference",
                f"unsafe declared reference path: {raw!r}",
            )
        return raw
    return PurePosixPath("reference", *path.parts).as_posix()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_STAGE_SCRIPT = r"""
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

def fail(message):
    print(json.dumps({"ok": False, "error": str(message)[:2000]}, separators=(",", ":")))
    raise SystemExit(2)

def remove_path(path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)

def validate_declared(root, declared_paths):
    for raw in declared_paths:
        path = PurePosixPath(raw)
        candidate = root.joinpath(*path.parts)
        current = root
        for part in path.parts:
            current = current / part
            if current.is_symlink():
                fail(f"declared input is or traverses a symlink: {raw}")
        if not candidate.is_file():
            fail(f"declared input is missing: {raw}")

config_path = Path(sys.argv[1])
archive = None
staging_root = None
try:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base = Path(config["base"])
    archive_name = config.get("archive_path")
    if archive_name:
        archive = Path(archive_name)
        digest = hashlib.sha256()
        with archive.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != config["archive_sha256"]:
            fail("trusted input archive SHA-256 mismatch")
        with zipfile.ZipFile(archive) as bundle:
            infos = bundle.infolist()
            if len(infos) > config["max_files"]:
                fail("trusted input archive exceeds file-count limit")
            total = 0
            names = set()
            for info in infos:
                path = PurePosixPath(info.filename)
                if (
                    not info.filename
                    or path.is_absolute()
                    or any(part in {"", ".", ".."} for part in path.parts)
                    or info.is_dir()
                    or len(path.parts) < 2
                    or path.parts[0] not in {"input", "software"}
                    or path.as_posix() != info.filename
                    or "\\" in info.filename
                ):
                    fail(f"unsafe trusted input archive member: {info.filename}")
                if info.filename in names:
                    fail(f"duplicate trusted input archive member: {info.filename}")
                names.add(info.filename)
                if info.file_size > config["max_file_bytes"]:
                    fail(f"trusted input member exceeds size limit: {info.filename}")
                total += info.file_size
                if total > config["max_source_bytes"]:
                    fail("trusted input archive exceeds total-size limit")
                mode = info.external_attr >> 16
                if (mode & 0o170000) != 0o100000:
                    fail(f"trusted input archive member is not regular: {info.filename}")
                if info.flag_bits & 1:
                    fail(f"trusted input archive member is encrypted: {info.filename}")
            staging_root = Path(tempfile.mkdtemp(prefix=".ale-input-stage-", dir=base))
            staging_root.chmod(0o700)
            for info in infos:
                target = staging_root.joinpath(*PurePosixPath(info.filename).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(info) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination, 1024 * 1024)
        validate_declared(staging_root, config["declared_input_paths"])
        backup_root = staging_root / ".previous"
        backup_root.mkdir(mode=0o700)
        try:
            for name in ("input", "software"):
                live = base / name
                if live.is_symlink() or live.exists():
                    live.replace(backup_root / name)
            for name in ("input", "software"):
                staged = staging_root / name
                if staged.exists():
                    staged.replace(base / name)
        except Exception:
            for name in ("input", "software"):
                remove_path(base / name)
            for name in ("input", "software"):
                previous = backup_root / name
                if previous.is_symlink() or previous.exists():
                    previous.replace(base / name)
            raise
    else:
        validate_declared(base, config["declared_input_paths"])
    print(json.dumps({"ok": True}, separators=(",", ":")))
except SystemExit:
    raise
except Exception as exc:
    fail(exc)
finally:
    if staging_root is not None:
        shutil.rmtree(staging_root, ignore_errors=True)
    if archive is not None:
        archive.unlink(missing_ok=True)
    config_path.unlink(missing_ok=True)
    Path(__file__).unlink(missing_ok=True)
""".strip()


_REFERENCE_STAGE_SCRIPT = r"""
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath

def fail(message):
    print(json.dumps({"ok": False, "error": str(message)[:2000]}, separators=(",", ":")))
    raise SystemExit(2)

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

try:
    config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    base = Path(config["base"])
    archive_name = config.get("archive_path")
    if archive_name:
        archive = Path(archive_name)
        if digest(archive) != config["archive_sha256"]:
            fail("trusted reference archive SHA-256 mismatch")
        reference_root = base / "reference"
        shutil.rmtree(reference_root, ignore_errors=True)
        with zipfile.ZipFile(archive) as bundle:
            infos = bundle.infolist()
            if len(infos) > config["max_files"]:
                fail("trusted reference archive exceeds file-count limit")
            total = 0
            for info in infos:
                path = PurePosixPath(info.filename)
                if (
                    path.is_absolute()
                    or any(part in {"", ".", ".."} for part in path.parts)
                    or info.is_dir()
                    or path.parts[0] != "reference"
                ):
                    fail(f"unsafe trusted reference archive member: {info.filename}")
                if info.file_size > config["max_file_bytes"]:
                    fail(f"trusted reference member exceeds size limit: {info.filename}")
                total += info.file_size
                if total > config["max_source_bytes"]:
                    fail("trusted reference archive exceeds total-size limit")
                mode = info.external_attr >> 16
                if mode and not (mode & 0o170000) == 0o100000:
                    fail(f"trusted reference archive member is not regular: {info.filename}")
            for info in infos:
                target = base.joinpath(*PurePosixPath(info.filename).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(info) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination, 1024 * 1024)
        archive.unlink(missing_ok=True)

    expected = {entry["path"]: entry for entry in config["entries"]}
    for raw, entry in expected.items():
        candidate = base.joinpath(*PurePosixPath(raw).parts)
        current = base
        for part in PurePosixPath(raw).parts:
            current = current / part
            if current.is_symlink():
                fail(f"reference is or traverses a symlink: {raw}")
        if not candidate.is_file():
            fail(f"reference file is missing: {raw}")
        if candidate.stat().st_size != entry["size_bytes"]:
            fail(f"reference size mismatch: {raw}")
        if digest(candidate) != entry["sha256"]:
            fail(f"reference SHA-256 mismatch: {raw}")
    for raw in config["declared_reference_paths"]:
        candidate = base.joinpath(*PurePosixPath(raw).parts)
        if raw not in expected or candidate.is_symlink() or not candidate.is_file():
            fail(f"declared reference is unavailable: {raw}")
    print(json.dumps({"ok": True}, separators=(",", ":")))
except SystemExit:
    raise
except Exception as exc:
    fail(exc)
""".strip()
