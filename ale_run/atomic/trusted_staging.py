"""Host-mediated task-data staging for atomic capabilities."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import re
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
_STAGING_CLEANUP_TIMEOUT_SECONDS = 35
_OSS_LIST_PAGE_SIZE = 16
_MAX_OSS_LIST_OUTPUT_BYTES = 64 * 1024

_OSS_LIST_HEADER_RE = re.compile(
    r"^LastModifiedTime\s+Size\(B\)\s+StorageClass\s+ETAG\s+ObjectName\s*$"
)
_OSS_LIST_ROW_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} [+-]\d{4} \S+\s+"
    r"(?P<size>\d+)\s+\S+\s+\S+\s+(?P<url>oss://.*)$"
)
_OSS_LIST_SUMMARY_RE = re.compile(r"^Object Number is: (?P<count>\d+)$")
_OSS_LIST_ELAPSED_RE = re.compile(r"^\s*\d+(?:\.\d+)?\(s\) elapsed\s*$")


@dataclass(frozen=True)
class PreparedInput:
    archive_path: Path
    archive_sha256: str


@dataclass(frozen=True)
class PreparedReference:
    archive_path: Path | None
    archive_sha256: str | None
    manifest: ReferenceManifest


@dataclass(frozen=True)
class _OssObject:
    url: str
    key: str
    relative_path: PurePosixPath
    size_bytes: int


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
        input_objects = await _list_prefix_objects(
            f"{prefix}/input/",
            required=True,
        )
        software_objects = await _list_prefix_objects(
            f"{prefix}/software/",
            required=False,
        )
        objects = [*input_objects, *software_objects]
        if len(objects) > _MAX_STAGED_FILES:
            raise AtomicInfrastructureError(
                "input",
                "OSS task data exceeds the file-count limit",
            )
        total = 0
        for item in objects:
            if item.size_bytes > _MAX_STAGED_FILE_BYTES:
                raise AtomicInfrastructureError(
                    "input",
                    f"OSS input file exceeds 512 MiB: {item.relative_path}",
                )
            total += item.size_bytes
            if total > _MAX_STAGED_SOURCE_BYTES:
                raise AtomicInfrastructureError(
                    "input",
                    "OSS task data exceeds the total-size limit",
                )
        actual_total = await _download_prefix(
            input_objects,
            payload / "input",
            actual_total=0,
        )
        await _download_prefix(
            software_objects,
            payload / "software",
            actual_total=actual_total,
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
    staging_path = join(sandbox, base, f".ale-input-{nonce}.stage")
    archive_sha256: str | None = None
    primary_error: BaseException | None = None
    run_attempted = False
    terminal_success = False
    try:
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
        run_attempted = True
        result = await sandbox.run_command(command, timeout=600)
        try:
            payload = json.loads(result.stdout or "")
        except (TypeError, json.JSONDecodeError) as exc:
            raise AtomicInfrastructureError(
                "input",
                "VM input verifier returned invalid JSON",
            ) from exc
        terminal_success = (
            result.returncode == 0
            and isinstance(payload, dict)
            and set(payload) == {"ok"}
            and payload["ok"] is True
        )
        if not terminal_success:
            detail = payload.get("error") if isinstance(payload, dict) else None
            raise AtomicInfrastructureError(
                "input",
                str(detail or result.stderr or "VM input verifier failed")[:2000],
            )
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        cleanup_paths = [
            archive_path,
            f"{archive_path}.b64",
            script_path,
            f"{script_path}.b64",
            config_path,
            f"{config_path}.b64",
        ]
        if terminal_success:
            cleanup_paths.append(staging_path)
        elif run_attempted and primary_error is not None:
            primary_error.add_note(
                f"VM input staging path preserved for operator recovery: {staging_path}"
            )
        cleanup_errors = []
        for cleanup_path in cleanup_paths:
            try:
                await asyncio.wait_for(
                    _checked_remove_staging_path(sandbox, cleanup_path),
                    timeout=_STAGING_CLEANUP_TIMEOUT_SECONDS,
                )
            except Exception as exc:  # noqa: BLE001
                cleanup_errors.append(f"{cleanup_path}: {type(exc).__name__}: {exc}"[:500])
        if cleanup_errors:
            detail = "VM input staging cleanup failed: " + "; ".join(cleanup_errors)
            if primary_error is not None:
                primary_error.add_note(detail[:2000])
            else:
                raise AtomicInfrastructureError("input", detail[:2000])


async def _checked_remove_staging_path(
    sandbox: SandboxHandle,
    path: str,
) -> None:
    if sandbox.is_linux:
        target = shlex.quote(path)
        command = (
            "# ale-input-cleanup\n"
            f"if [ -L {target} ] || [ -e {target} ]; then rm -rf -- {target}; fi\n"
            f"if [ -L {target} ] || [ -e {target} ]; then "
            "echo 'cleanup target remains' >&2; exit 1; fi"
        )
    else:
        target = path.replace("'", "''")
        command = (
            'powershell -NoProfile -Command "'
            "$ErrorActionPreference='Stop';"
            "$aleInputCleanup='ale-input-cleanup';"
            f"$p='{target}';"
            "$item=Get-Item -LiteralPath $p -Force -ErrorAction SilentlyContinue;"
            "if($null -ne $item){"
            "if($item.LinkType){"
            "Remove-Item -LiteralPath $p -Force -ErrorAction Stop"
            "}else{"
            "Remove-Item -LiteralPath $p -Recurse -Force -ErrorAction Stop"
            "}"
            "};"
            "$remaining=Get-Item -LiteralPath $p -Force -ErrorAction SilentlyContinue;"
            "if($null -ne $remaining){Write-Error 'cleanup target remains';exit 1}"
            '"'
        )
    result = await sandbox.run_command(command, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(
            f"checked cleanup failed rc={result.returncode}: "
            f"{str(result.stderr or result.stdout or '')[:500]}"
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


async def _list_prefix_objects(
    source: str,
    *,
    required: bool,
) -> list[_OssObject]:
    if not source.startswith("oss://") or not source.endswith("/"):
        raise AtomicInfrastructureError("input", f"invalid OSS prefix: {source}")
    bucket_and_prefix = source.removeprefix("oss://")
    bucket, separator, prefix_key = bucket_and_prefix.partition("/")
    if not bucket or not separator or not prefix_key:
        raise AtomicInfrastructureError("input", f"invalid OSS prefix: {source}")

    objects: list[_OssObject] = []
    seen_keys: set[str] = set()
    marker: str | None = None
    while True:
        arguments = [
            "ls",
            "--payer",
            "requester",
            source,
            "--limited-num",
            str(_OSS_LIST_PAGE_SIZE),
        ]
        if marker is not None:
            arguments.extend(("--marker", marker))
        listed = await run_host_ossutil(*arguments)
        if listed[0] != 0:
            raise AtomicInfrastructureError(
                "input",
                f"cannot list OSS prefix {source}: {host_command_diagnostic(listed)}",
            )
        if (
            len(listed[1]) > _MAX_OSS_LIST_OUTPUT_BYTES
            or len(listed[2]) > _MAX_OSS_LIST_OUTPUT_BYTES
        ):
            raise AtomicInfrastructureError(
                "input",
                f"OSS metadata listing output exceeded 64 KiB: {source}",
            )
        if listed[2]:
            raise AtomicInfrastructureError(
                "input",
                f"OSS metadata listing reported an error for {source}: "
                f"{host_command_diagnostic(listed)}",
            )
        try:
            lines = listed[1].decode("utf-8", errors="strict").splitlines()
        except UnicodeDecodeError as exc:
            raise AtomicInfrastructureError(
                "input",
                f"OSS metadata listing is not UTF-8: {source}",
            ) from exc
        while lines and not lines[-1]:
            lines.pop()
        if lines and _OSS_LIST_ELAPSED_RE.fullmatch(lines[-1]):
            lines.pop()
        if lines == ["Object Number is: 0"]:
            row_lines: list[str] = []
            page_count = 0
        elif not lines or _OSS_LIST_HEADER_RE.fullmatch(lines[0]) is None:
            raise AtomicInfrastructureError(
                "input",
                f"malformed OSS metadata listing header: {source}",
            )
        elif len(lines) < 2:
            raise AtomicInfrastructureError(
                "input",
                f"malformed OSS metadata listing summary: {source}",
            )
        else:
            summary = _OSS_LIST_SUMMARY_RE.fullmatch(lines[-1])
            if summary is None:
                raise AtomicInfrastructureError(
                    "input",
                    f"malformed OSS metadata listing summary: {source}",
                )
            row_lines = lines[1:-1]
            page_count = int(summary.group("count"))
        if page_count != len(row_lines) or page_count > _OSS_LIST_PAGE_SIZE:
            raise AtomicInfrastructureError(
                "input",
                f"ambiguous OSS metadata listing count: {source}",
            )

        page: list[_OssObject] = []
        previous_key = marker
        for line in row_lines:
            match = _OSS_LIST_ROW_RE.fullmatch(line)
            if match is None:
                raise AtomicInfrastructureError(
                    "input",
                    f"malformed OSS metadata row: {source}",
                )
            url = match.group("url")
            if not url.startswith(source):
                raise AtomicInfrastructureError(
                    "input",
                    f"OSS metadata object is outside the requested prefix: {url}",
                )
            relative = url[len(source) :]
            path = PurePosixPath(relative)
            key = f"{prefix_key}{relative}"
            if (
                not relative
                or relative.endswith("/")
                or path.is_absolute()
                or any(part in {"", ".", ".."} for part in path.parts)
                or path.as_posix() != relative
                or "\\" in relative
                or any(ord(character) < 32 or ord(character) == 127 for character in relative)
            ):
                raise AtomicInfrastructureError(
                    "input",
                    f"unsafe or noncanonical OSS object key: {url}",
                )
            if key in seen_keys:
                raise AtomicInfrastructureError(
                    "input",
                    f"duplicate OSS object key: {url}",
                )
            if previous_key is not None and key <= previous_key:
                raise AtomicInfrastructureError(
                    "input",
                    f"OSS metadata listing marker did not advance: {url}",
                )
            item = _OssObject(
                url=url,
                key=key,
                relative_path=path,
                size_bytes=int(match.group("size")),
            )
            page.append(item)
            seen_keys.add(key)
            previous_key = key
        objects.extend(page)
        if len(objects) > _MAX_STAGED_FILES:
            raise AtomicInfrastructureError(
                "input",
                "OSS task data exceeds the file-count limit",
            )
        if page_count < _OSS_LIST_PAGE_SIZE:
            break
        if not page:
            raise AtomicInfrastructureError(
                "input",
                f"OSS metadata listing marker did not advance: {source}",
            )
        marker = page[-1].key

    if required and not objects:
        raise AtomicInfrastructureError(
            "input",
            "required OSS input prefix is empty",
        )
    return objects


async def _download_prefix(
    objects: list[_OssObject],
    destination: Path,
    *,
    actual_total: int,
) -> int:
    if not objects:
        return actual_total
    try:
        destination.mkdir(mode=0o700)
    except FileExistsError:
        pass
    destination_metadata = destination.lstat()
    if stat.S_ISLNK(destination_metadata.st_mode) or not stat.S_ISDIR(destination_metadata.st_mode):
        raise AtomicInfrastructureError("input", "unsafe host OSS staging root")
    destination_root = destination.resolve(strict=True)

    for item in objects:
        target = destination.joinpath(*item.relative_path.parts)
        current = destination
        for part in item.relative_path.parts[:-1]:
            current /= part
            try:
                current.mkdir(mode=0o700)
            except FileExistsError:
                metadata = current.lstat()
                if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                    raise AtomicInfrastructureError(
                        "input",
                        f"unsafe host OSS staging parent: {item.relative_path}",
                    )
        if not target.parent.resolve(strict=True).is_relative_to(destination_root):
            raise AtomicInfrastructureError(
                "input",
                f"host OSS staging path escaped its private root: {item.relative_path}",
            )
        if os.path.lexists(target):
            raise AtomicInfrastructureError(
                "input",
                f"duplicate local OSS staging target: {item.relative_path}",
            )
        downloaded = await run_host_ossutil(
            "cp",
            "--payer",
            "requester",
            item.url,
            str(target),
            "-f",
        )
        if downloaded[0] != 0:
            raise AtomicInfrastructureError(
                "input",
                f"host OSS object download failed: {host_command_diagnostic(downloaded)}",
            )
        current = destination
        for part in item.relative_path.parts[:-1]:
            current /= part
            try:
                parent_metadata = current.lstat()
            except OSError as exc:
                raise AtomicInfrastructureError(
                    "input",
                    f"unsafe host OSS staging parent: {item.relative_path}",
                ) from exc
            if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
                raise AtomicInfrastructureError(
                    "input",
                    f"unsafe host OSS staging parent: {item.relative_path}",
                )
        if not target.parent.resolve(strict=True).is_relative_to(destination_root):
            raise AtomicInfrastructureError(
                "input",
                f"unsafe host OSS staging parent: {item.relative_path}",
            )
        try:
            local = target.lstat()
        except OSError as exc:
            raise AtomicInfrastructureError(
                "input",
                f"host OSS download did not create a regular file: {item.relative_path}",
            ) from exc
        if stat.S_ISLNK(local.st_mode) or not stat.S_ISREG(local.st_mode):
            raise AtomicInfrastructureError(
                "input",
                f"host OSS download did not create a regular file: {item.relative_path}",
            )
        actual_total += local.st_size
        if actual_total > _MAX_STAGED_SOURCE_BYTES:
            raise AtomicInfrastructureError(
                "input",
                "downloaded OSS task data exceeds the total-size limit",
            )
        if local.st_size != item.size_bytes:
            raise AtomicInfrastructureError(
                "input",
                f"OSS object size changed during download: {item.relative_path}",
            )
        os.chmod(target, 0o600, follow_symlinks=False)
    return actual_total


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
import zipfile
from pathlib import Path, PurePosixPath

def reject(message):
    raise RuntimeError(str(message))

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
                reject(f"declared input is or traverses a symlink: {raw}")
        if not candidate.is_file():
            reject(f"declared input is missing: {raw}")

config_path = Path(sys.argv[1])
archive = config_path.with_suffix(".zip")
script_path = Path(__file__)
known_staging_path = config_path.with_suffix(".stage")
staging_root = None
staging_created = False
preserve_staging = False
primary_error = None
cleanup_errors = []
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
            reject("trusted input archive SHA-256 mismatch")
        with zipfile.ZipFile(archive) as bundle:
            infos = bundle.infolist()
            if len(infos) > config["max_files"]:
                reject("trusted input archive exceeds file-count limit")
            total = 0
            names = set()
            input_members = 0
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
                    reject(f"unsafe trusted input archive member: {info.filename}")
                if info.filename in names:
                    reject(f"duplicate trusted input archive member: {info.filename}")
                names.add(info.filename)
                if info.file_size > config["max_file_bytes"]:
                    reject(f"trusted input member exceeds size limit: {info.filename}")
                total += info.file_size
                if total > config["max_source_bytes"]:
                    reject("trusted input archive exceeds total-size limit")
                mode = info.external_attr >> 16
                if (mode & 0o170000) != 0o100000:
                    reject(f"trusted input archive member is not regular: {info.filename}")
                if info.flag_bits & 1:
                    reject(f"trusted input archive member is encrypted: {info.filename}")
                if path.parts[0] == "input":
                    input_members += 1
            if input_members == 0:
                reject("trusted input archive contains no input files")
            staging_root = known_staging_path
            staging_root.mkdir(mode=0o700)
            staging_created = True
            for info in infos:
                target = staging_root.joinpath(*PurePosixPath(info.filename).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(info) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination, 1024 * 1024)
        input_root = staging_root / "input"
        if input_root.is_symlink() or not input_root.is_dir():
            reject("trusted input archive did not produce a real input directory")
        validate_declared(staging_root, config["declared_input_paths"])
        backup_root = staging_root / ".previous"
        backup_root.mkdir(mode=0o700)
        moved_old = {"input": False, "software": False}
        installed_new = {"input": False, "software": False}
        transaction_succeeded = False
        try:
            for name in ("input", "software"):
                live = base / name
                if live.is_symlink() or live.exists():
                    live.replace(backup_root / name)
                    moved_old[name] = True
            for name in ("input", "software"):
                staged = staging_root / name
                if staged.exists():
                    staged.replace(base / name)
                    installed_new[name] = True
            transaction_succeeded = True
        except Exception as replacement_error:
            rollback_errors = []
            for name in ("input", "software"):
                if installed_new[name]:
                    try:
                        remove_path(base / name)
                        installed_new[name] = False
                    except Exception as exc:
                        rollback_errors.append(
                            f"remove new {name}: {type(exc).__name__}: {exc}"
                        )
            for name in ("input", "software"):
                if moved_old[name]:
                    previous = backup_root / name
                    try:
                        previous.replace(base / name)
                        moved_old[name] = False
                    except Exception as exc:
                        rollback_errors.append(
                            f"restore old {name}: {type(exc).__name__}: {exc}"
                        )
            rollback_incomplete = (
                any(installed_new.values())
                or any(moved_old.values())
                or bool(rollback_errors)
            )
            if rollback_incomplete:
                preserve_staging = True
                reject(
                    f"{replacement_error}; rollback backup preserved at {staging_root}; "
                    + "; ".join(rollback_errors)
                )
            raise
    else:
        validate_declared(base, config["declared_input_paths"])
except Exception as exc:
    primary_error = exc
finally:
    if staging_created and not preserve_staging:
        try:
            shutil.rmtree(staging_root)
        except Exception as exc:
            cleanup_errors.append(f"staging root: {type(exc).__name__}: {exc}")
    try:
        archive.unlink(missing_ok=True)
    except Exception as exc:
        cleanup_errors.append(f"archive: {type(exc).__name__}: {exc}")
    try:
        config_path.unlink(missing_ok=True)
    except Exception as exc:
        cleanup_errors.append(f"config: {type(exc).__name__}: {exc}")
    try:
        script_path.unlink(missing_ok=True)
    except Exception as exc:
        cleanup_errors.append(f"script: {type(exc).__name__}: {exc}")

if primary_error is not None or cleanup_errors:
    details = str(primary_error) if primary_error is not None else "VM input cleanup failed"
    if cleanup_errors:
        details += "; cleanup errors: " + "; ".join(cleanup_errors)
    print(
        json.dumps(
            {
                "ok": False,
                "error": details[:2000],
                "preserve_staging": preserve_staging,
            },
            separators=(",", ":"),
        )
    )
    raise SystemExit(2)
print(json.dumps({"ok": True}, separators=(",", ":")))
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
