"""Immutable OSS commit protocol for atomic solve/evaluate artifacts."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import math
import mimetypes
import os
import re
import shlex
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from time import monotonic as _monotonic
from typing import Any
from urllib.parse import quote
from uuid import uuid4

from pydantic import ValidationError

from ..base_interface import RangeResult, SandboxHandle, TaskDataSpec
from ..environments.task_data import join, ossbucket, task_subdir
from .contracts import (
    AtomicInfrastructureError,
    EvaluateRequest,
    EvaluationResult,
    SolveRequest,
    SubmissionManifest,
)
from .host_oss import (
    host_command_diagnostic,
    read_host_oss_object,
    run_host_ossutil,
)

_COMMAND_TIMEOUT_SECONDS = 3600
_MAX_DECLARED_ARTIFACTS = 1024
_MAX_TASK_CARD_BYTES = 1024 * 1024
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_RESULT_BYTES = 64 * 1024
_MAX_SCRIPT_OUTPUT_BYTES = 2 * 1024 * 1024
_MAX_PATH_BYTES = 4096
_MAX_ARTIFACT_BYTES = 4 * 1024 * 1024 * 1024
_MAX_ARTIFACT_TOTAL_BYTES = 16 * 1024 * 1024 * 1024
_DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024
_OSS_ARGUMENT_TOKEN = "__ALE_OSS_ARGUMENTS__"
_PROVENANCE_FIELDS = frozenset(
    {"ale_run_id", "model_id", "config_digest", "started_at", "completed_at"}
)


async def read_existing_submission_manifest(
    request: SolveRequest,
) -> SubmissionManifest | None:
    """Read the canonical commit marker on the trusted host before provisioning."""
    raw = await read_host_oss_object(
        f"{_submission_prefix(request)}/manifest.json",
        limit=_MAX_MANIFEST_BYTES,
        missing_ok=True,
        integrity_category="idempotency_conflict",
    )
    if raw is None:
        return None
    try:
        manifest = SubmissionManifest.model_validate_json(raw)
    except ValidationError as exc:
        raise AtomicInfrastructureError(
            "idempotency_conflict",
            f"occupied submission manifest is invalid: {exc}",
        ) from exc
    if raw != _canonical_json(manifest.model_dump(mode="json")):
        raise AtomicInfrastructureError(
            "idempotency_conflict",
            "occupied submission manifest is not canonical",
        )
    return manifest


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _submission_prefix(request: SolveRequest | EvaluateRequest) -> str:
    return f"{request.submission_root.rstrip('/')}/output/{request.submission_id}"


def _sandbox_temp_path(sandbox: SandboxHandle, name: str) -> str:
    root = "/tmp" if sandbox.is_linux else r"C:\Windows\Temp"
    return join(sandbox, root, name)


def _python_command(sandbox: SandboxHandle, script_path: str, config_path: str) -> str:
    python = sandbox.python or ("python3" if sandbox.is_linux else "python")
    if sandbox.is_linux:
        return f"{shlex.quote(python)} {shlex.quote(script_path)} {shlex.quote(config_path)}"
    return f'"{python}" "{script_path}" "{config_path}"'


def _quote_oss_argument(sandbox: SandboxHandle, value: str) -> str:
    if sandbox.is_linux:
        return shlex.quote(value)
    return ossbucket.powershell_literal(value)


def _oss_call(sandbox: SandboxHandle, arguments: str) -> str:
    command = ossbucket.oss_command(sandbox, arguments)
    if sandbox.is_linux:
        return command
    return _encode_windows_powershell(command)


def _encode_windows_powershell(command: str) -> str:
    prefix = 'powershell -NoProfile -Command "'
    if not command.startswith(prefix) or not command.endswith('"'):
        raise AtomicInfrastructureError(
            "submission_integrity",
            "shared Windows OSS command has an unsupported shell shape",
        )
    script = command[len(prefix) : -1]
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    return f"powershell -NoProfile -EncodedCommand {encoded}"


def _command_diagnostic(result: Any) -> str:
    diagnostic = result.stderr or result.stdout or "unknown command failure"
    return str(diagnostic).strip()[:1000]


def _parse_script_output(
    result: Any,
    *,
    default_category: str,
) -> dict[str, Any]:
    stdout = (result.stdout or "").encode("utf-8", errors="replace")
    if len(stdout) > _MAX_SCRIPT_OUTPUT_BYTES:
        raise AtomicInfrastructureError(
            default_category, "VM script output exceeded the 2 MiB limit"
        )
    try:
        payload = json.loads(stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise AtomicInfrastructureError(
            default_category,
            f"VM script returned invalid JSON: {_command_diagnostic(result)}",
        ) from exc
    if not isinstance(payload, dict):
        raise AtomicInfrastructureError(default_category, "VM script returned a non-object result")
    if result.returncode != 0 or payload.get("ok") is not True:
        category = payload.get("category")
        message = payload.get("error")
        raise AtomicInfrastructureError(
            category if isinstance(category, str) else default_category,
            str(message or _command_diagnostic(result))[:1000],
        )
    return payload


def _task_card_path(request: SolveRequest | EvaluateRequest) -> Path:
    raw_path = request.task_path
    posix_path = PurePosixPath(raw_path)
    if (
        not raw_path
        or "\x00" in raw_path
        or "\\" in raw_path
        or posix_path.is_absolute()
        or PureWindowsPath(raw_path).is_absolute()
        or any(part in {"", ".", ".."} for part in posix_path.parts)
        or posix_path.as_posix() != raw_path
    ):
        raise AtomicInfrastructureError(
            "submission_integrity",
            f"task_path must be a canonical relative identity: {raw_path!r}",
        )
    try:
        tasks_root_raw = request.task_repo / "tasks"
        if tasks_root_raw.is_symlink():
            raise AtomicInfrastructureError(
                "submission_integrity", "task repository tasks root is a symlink"
            )
        tasks_root = tasks_root_raw.resolve(strict=True)
        task_dir = tasks_root
        for part in posix_path.parts:
            task_dir /= part
            if task_dir.is_symlink():
                raise AtomicInfrastructureError(
                    "submission_integrity",
                    f"task_path traverses a symlink: {raw_path!r}",
                )
            if not task_dir.is_dir():
                raise AtomicInfrastructureError(
                    "submission_integrity",
                    f"task_path does not name a task directory: {raw_path!r}",
                )
        candidate_raw = task_dir / "task_card.json"
        if candidate_raw.is_symlink():
            raise AtomicInfrastructureError(
                "submission_integrity",
                f"task_card.json is a symlink for task {raw_path!r}",
            )
        candidate = candidate_raw.resolve(strict=True)
    except AtomicInfrastructureError:
        raise
    except (FileNotFoundError, OSError) as exc:
        raise AtomicInfrastructureError(
            "submission_integrity",
            f"cannot resolve canonical task identity {raw_path!r}: {exc}",
        ) from exc
    try:
        candidate.relative_to(tasks_root)
    except ValueError as exc:
        raise AtomicInfrastructureError(
            "submission_integrity", "task_path escapes the task repository"
        ) from exc
    if not candidate.is_file():
        raise AtomicInfrastructureError(
            "submission_integrity", f"task card is not a file: {candidate}"
        )
    return candidate


def _declared_output_entries(
    request: SolveRequest,
) -> tuple[tuple[str, str], ...]:
    try:
        task_card_path = _task_card_path(request)
        with task_card_path.open("rb") as stream:
            raw = stream.read(_MAX_TASK_CARD_BYTES + 1)
    except AtomicInfrastructureError:
        raise
    except (FileNotFoundError, OSError) as exc:
        raise AtomicInfrastructureError(
            "submission_integrity", f"cannot read task_card.json: {exc}"
        ) from exc
    if len(raw) > _MAX_TASK_CARD_BYTES:
        raise AtomicInfrastructureError(
            "submission_integrity", "task_card.json exceeds the 1 MiB limit"
        )
    try:
        card = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AtomicInfrastructureError(
            "submission_integrity", f"invalid task_card.json: {exc}"
        ) from exc
    output_files = card.get("outputFiles") if isinstance(card, dict) else None
    if not isinstance(output_files, list) or not output_files:
        raise AtomicInfrastructureError(
            "submission_integrity", "task_card.json must declare outputFiles"
        )
    if len(output_files) > _MAX_DECLARED_ARTIFACTS:
        raise AtomicInfrastructureError(
            "submission_integrity",
            f"outputFiles exceeds the {_MAX_DECLARED_ARTIFACTS}-file limit",
        )

    entries: list[tuple[str, str]] = []
    for entry in output_files:
        raw_path = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(raw_path, str):
            raise AtomicInfrastructureError(
                "submission_integrity", "each outputFiles entry must have a string path"
            )
        path = _validate_artifact_path(raw_path)
        if any(existing_path == path for existing_path, _media_type in entries):
            raise AtomicInfrastructureError(
                "submission_integrity", f"duplicate declared output path: {path}"
            )
        raw_media_type = (
            entry.get("mediaType") or entry.get("media_type") if isinstance(entry, dict) else None
        )
        media_type = (
            str(raw_media_type)
            if raw_media_type
            else mimetypes.guess_type(path)[0] or "application/octet-stream"
        )
        entries.append((path, media_type))
    return tuple(entries)


def _declared_output_paths(request: SolveRequest) -> tuple[str, ...]:
    return tuple(path for path, _media_type in _declared_output_entries(request))


def _validate_artifact_path(raw_path: str) -> str:
    if (
        not raw_path
        or "\x00" in raw_path
        or "\\" in raw_path
        or len(raw_path.encode("utf-8")) > _MAX_PATH_BYTES
    ):
        raise AtomicInfrastructureError(
            "submission_integrity", f"unsafe artifact path: {raw_path!r}"
        )
    path = PurePosixPath(raw_path)
    if (
        path.is_absolute()
        or len(path.parts) < 2
        or path.parts[0] != "output"
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != raw_path
    ):
        raise AtomicInfrastructureError(
            "submission_integrity", f"unsafe artifact path: {raw_path!r}"
        )
    return raw_path


def _output_root(sandbox: SandboxHandle, task_data: TaskDataSpec) -> str:
    if task_data.remote_output_dir:
        return task_data.remote_output_dir
    return join(sandbox, task_subdir(sandbox, task_data), "output")


def _manifest_base(
    request: SolveRequest,
    provenance: Mapping[str, object],
) -> dict[str, object]:
    keys = frozenset(provenance)
    if keys != _PROVENANCE_FIELDS:
        missing = sorted(_PROVENANCE_FIELDS - keys)
        extra = sorted(keys - _PROVENANCE_FIELDS)
        raise AtomicInfrastructureError(
            "submission_integrity",
            f"invalid manifest provenance fields (missing={missing}, extra={extra})",
        )
    base: dict[str, object] = {
        "schema_version": 1,
        "status": "submitted",
        "submission_id": str(request.submission_id),
        "task_path": request.task_path,
        "variant_index": request.variant_index,
        "task_commit": request.task_commit,
        "image_id": request.image_id,
        "agent_id": request.agent_id,
    }
    for key in _PROVENANCE_FIELDS:
        value = provenance[key]
        base[key] = value.isoformat() if hasattr(value, "isoformat") else value
    return base


async def _write_script_bundle(
    sandbox: SandboxHandle,
    *,
    stem: str,
    script: str,
    config: Mapping[str, object],
) -> tuple[str, str]:
    nonce = uuid4().hex
    script_path = _sandbox_temp_path(sandbox, f"{stem}-{nonce}.py")
    config_path = _sandbox_temp_path(sandbox, f"{stem}-{nonce}.json")
    config_bytes = _canonical_json(config)
    if len(config_bytes) > _MAX_MANIFEST_BYTES * 2:
        raise AtomicInfrastructureError(
            "submission_integrity", "VM script configuration is too large"
        )
    await sandbox.write_file(script_path, script.encode("utf-8"))
    await sandbox.write_file(config_path, config_bytes)
    return script_path, config_path


async def download_range_to_local(
    sandbox: SandboxHandle,
    remote_path: str,
    local_path: str | os.PathLike[str],
    max_bytes: int,
    timeout: float,
) -> None:
    """Stream one size-bounded sandbox file into a private host file."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
        raise ValueError("max_bytes must be a non-negative integer")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise ValueError("timeout must be a positive finite number")

    deadline = _monotonic() + timeout
    destination = Path(local_path)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(destination, flags, 0o600)
    complete = False
    try:
        with os.fdopen(fd, "wb") as stream:
            offset = 0
            expected_size: int | None = None
            while expected_size is None or offset < expected_size:
                remaining = max_bytes - offset
                if expected_size is not None:
                    remaining = min(remaining, expected_size - offset)
                requested = min(_DOWNLOAD_CHUNK_BYTES, remaining)
                if requested == 0:
                    requested = 1
                remaining_timeout = deadline - _monotonic()
                if remaining_timeout <= 0:
                    raise AtomicInfrastructureError(
                        "submission_integrity",
                        f"range download deadline exhausted at offset {offset}",
                    )
                try:
                    async with asyncio.timeout(remaining_timeout):
                        result = await sandbox.download_range(
                            remote_path,
                            start=offset,
                            max_chunk_bytes=requested,
                            timeout=remaining_timeout,
                        )
                except TimeoutError as exc:
                    raise AtomicInfrastructureError(
                        "submission_integrity",
                        f"range download deadline exhausted at offset {offset}",
                    ) from exc
                if _monotonic() >= deadline:
                    raise AtomicInfrastructureError(
                        "submission_integrity",
                        f"range download deadline exhausted at offset {offset}",
                    )
                if not isinstance(result, RangeResult):
                    raise TypeError(f"range response has invalid shape at offset {offset}")
                if result.success is not True:
                    raise RuntimeError(
                        f"range download failed at offset {offset}: "
                        f"{result.error or 'unknown error'}"
                    )
                if (
                    isinstance(result.new_size, bool)
                    or not isinstance(result.new_size, int)
                    or result.new_size < 0
                ):
                    raise RuntimeError(f"range download returned a bad size at offset {offset}")
                if result.new_size > max_bytes:
                    raise RuntimeError(f"remote file exceeds the {max_bytes}-byte download limit")
                if expected_size is None:
                    expected_size = result.new_size
                elif result.new_size != expected_size:
                    raise RuntimeError(
                        f"remote file size changed from {expected_size} to {result.new_size} bytes"
                    )
                chunk = result.new_data
                if not isinstance(chunk, bytes):
                    raise TypeError(f"range download returned non-bytes data at offset {offset}")
                if len(chunk) > requested:
                    raise RuntimeError(
                        f"range download exceeded requested range at offset {offset}"
                    )
                if offset + len(chunk) > max_bytes:
                    raise RuntimeError(
                        f"range download exceeds the {max_bytes}-byte download limit"
                    )
                if offset + len(chunk) > expected_size:
                    raise RuntimeError(f"range download exceeds reported total at offset {offset}")
                if expected_size == 0:
                    if chunk:
                        raise RuntimeError("range download exceeds the 0-byte download limit")
                    break
                if not chunk:
                    raise RuntimeError(
                        f"range download returned empty before expected offset {expected_size}"
                    )
                stream.write(chunk)
                offset += len(chunk)
        complete = True
    finally:
        if not complete:
            destination.unlink(missing_ok=True)


def _validate_inspector_artifacts(
    artifacts: list[dict[str, object]] | None,
    declared_paths: tuple[str, ...],
) -> list[dict[str, object]]:
    if artifacts is None or len(artifacts) != len(declared_paths):
        raise AtomicInfrastructureError(
            "submission_integrity",
            "artifact inspector metadata does not match declared output count",
        )

    validated: list[dict[str, object]] = []
    total_bytes = 0
    expected_keys = {"path", "type", "size_bytes", "sha256"}
    for index, (row, declared_path) in enumerate(zip(artifacts, declared_paths, strict=True)):
        if not isinstance(row, dict) or set(row) != expected_keys:
            raise AtomicInfrastructureError(
                "submission_integrity",
                f"artifact inspector row {index} has unsafe metadata",
            )
        size = row["size_bytes"]
        digest = row["sha256"]
        if (
            row["path"] != declared_path
            or row["type"] != "file"
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise AtomicInfrastructureError(
                "submission_integrity",
                f"artifact inspector row {index} is invalid for {declared_path}",
            )
        if size > _MAX_ARTIFACT_BYTES:
            raise AtomicInfrastructureError(
                "submission_integrity",
                f"artifact inspector metadata exceeds per-file limit: {declared_path}",
            )
        total_bytes += size
        if total_bytes > _MAX_ARTIFACT_TOTAL_BYTES:
            raise AtomicInfrastructureError(
                "submission_integrity",
                "artifact inspector metadata exceeds total staging limit",
            )
        validated.append(row)
    return validated


async def publish_submission(
    sandbox: SandboxHandle,
    task_data: TaskDataSpec,
    request: SolveRequest,
    *,
    provenance: Mapping[str, object],
) -> SubmissionManifest:
    """Publish declared solve outputs and commit them with ``manifest.json``."""
    declared_entries = _declared_output_entries(request)
    declared_paths = tuple(path for path, _media_type in declared_entries)
    manifest_base = _manifest_base(request, provenance)
    submission_prefix = _submission_prefix(request)
    output_root = _output_root(sandbox, task_data)

    inspected_artifacts = _validate_inspector_artifacts(
        await _inspect_submission_artifacts(
            sandbox,
            output_root=output_root,
            declared_paths=declared_paths,
            allow_missing_root=False,
        ),
        declared_paths,
    )
    with tempfile.TemporaryDirectory(prefix="ale-submission-staging-") as temp_dir:
        staging = Path(temp_dir)
        os.chmod(staging, 0o700)
        artifact_root = staging / "artifacts"
        artifact_rows: list[dict[str, object]] = []
        for (declared_path, media_type), inspected in zip(
            declared_entries, inspected_artifacts, strict=True
        ):
            relative = PurePosixPath(declared_path).relative_to("output")
            remote_path = join(sandbox, output_root, *relative.parts)
            local_path = artifact_root.joinpath(*PurePosixPath(declared_path).parts)
            local_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            expected_size = int(inspected["size_bytes"])
            expected_sha256 = str(inspected["sha256"])
            try:
                await download_range_to_local(
                    sandbox,
                    remote_path,
                    local_path,
                    max_bytes=expected_size,
                    timeout=_COMMAND_TIMEOUT_SECONDS,
                )
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                raise AtomicInfrastructureError(
                    "submission_integrity",
                    f"failed to stream required output into trusted staging: "
                    f"{declared_path}: {exc}",
                ) from exc
            if local_path.is_symlink() or not local_path.is_file():
                raise AtomicInfrastructureError(
                    "submission_integrity",
                    f"invalid trusted staging file: {declared_path}",
                )
            size = local_path.stat().st_size
            digest = _local_sha256(local_path)
            if size != expected_size or digest != expected_sha256:
                raise AtomicInfrastructureError(
                    "submission_integrity",
                    f"streamed output does not match inspector metadata: {declared_path}",
                )
            artifact_rows.append(
                {
                    "path": declared_path,
                    "size_bytes": size,
                    "sha256": digest,
                    "media_type": media_type,
                    "local_path": local_path,
                }
            )

        try:
            manifest = SubmissionManifest.model_validate(
                manifest_base
                | {
                    "artifacts": [
                        {key: row[key] for key in ("path", "size_bytes", "sha256", "media_type")}
                        for row in artifact_rows
                    ]
                }
            )
        except ValidationError as exc:
            raise AtomicInfrastructureError(
                "submission_integrity",
                f"invalid submission manifest before commit: {exc}",
            ) from exc
        existing = await read_existing_submission_manifest(request)
        if existing is not None:
            if existing != manifest:
                raise AtomicInfrastructureError(
                    "idempotency_conflict",
                    "manifest.json already commits different submission bytes",
                )
            return existing

        for row in artifact_rows:
            artifact_url = f"{submission_prefix}/artifacts/{row['path']}"
            uploaded = await run_host_ossutil(
                "cp",
                str(row["local_path"]),
                artifact_url,
                "--forbid-overwrite",
                "--meta",
                f"x-oss-meta-sha256:{row['sha256']}",
            )
            if uploaded[0] != 0:
                await _verify_host_artifact(
                    artifact_url,
                    expected_size=int(row["size_bytes"]),
                    expected_sha256=str(row["sha256"]),
                    conflict=True,
                )
            await _verify_host_artifact(
                artifact_url,
                expected_size=int(row["size_bytes"]),
                expected_sha256=str(row["sha256"]),
                conflict=False,
            )

        manifest_bytes = _canonical_json(manifest.model_dump(mode="json"))
        if len(manifest_bytes) > _MAX_MANIFEST_BYTES:
            raise AtomicInfrastructureError(
                "submission_integrity",
                "generated manifest exceeds the 1 MiB limit",
            )
        manifest_path = staging / "manifest.json"
        manifest_path.write_bytes(manifest_bytes)
        os.chmod(manifest_path, 0o600)
        manifest_url = f"{submission_prefix}/manifest.json"
        committed = await run_host_ossutil(
            "cp",
            str(manifest_path),
            manifest_url,
            "--forbid-overwrite",
        )
        if committed[0] != 0:
            raced = await read_existing_submission_manifest(request)
            if raced != manifest:
                raise AtomicInfrastructureError(
                    "idempotency_conflict" if raced is not None else "submission_storage",
                    "manifest commit lost a race or failed without a committed result",
                )
        return manifest


def _local_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def _verify_host_artifact(
    url: str,
    *,
    expected_size: int,
    expected_sha256: str,
    conflict: bool,
) -> None:
    result = await run_host_ossutil("stat", url)
    if result[0] != 0:
        raise AtomicInfrastructureError(
            "submission_storage",
            f"artifact remote verification failed: {host_command_diagnostic(result)}",
        )
    output = result[1].decode("utf-8", errors="replace")
    size_match = re.search(
        r"(?im)^\s*(?:content[- ]?length|size)\s*[:=]\s*(\d+)\s*$",
        output,
    )
    digest_match = re.search(
        r"(?i)[\"']?x-oss-meta-sha256[\"']?\s*[:=]\s*[\"']?([0-9a-f]{64})",
        output,
    )
    matches = (
        size_match is not None
        and digest_match is not None
        and int(size_match.group(1)) == expected_size
        and digest_match.group(1).lower() == expected_sha256
    )
    if not matches:
        raise AtomicInfrastructureError(
            "idempotency_conflict" if conflict else "submission_storage",
            f"artifact metadata mismatch for {url}",
        )


async def _inspect_submission_artifacts(
    sandbox: SandboxHandle,
    *,
    output_root: str,
    declared_paths: tuple[str, ...],
    allow_missing_root: bool,
) -> list[dict[str, object]] | None:
    script_path, config_path = await _write_script_bundle(
        sandbox,
        stem="ale-atomic-inspect",
        script=_INSPECT_SCRIPT,
        config={
            "output_root": output_root,
            "declared_paths": declared_paths,
            "allow_missing_root": allow_missing_root,
        },
    )
    result = await sandbox.run_command(
        _python_command(sandbox, script_path, config_path),
        timeout=_COMMAND_TIMEOUT_SECONDS,
    )
    payload = _parse_script_output(result, default_category="submission_integrity")
    if payload.get("missing_root") is True:
        return None
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise AtomicInfrastructureError(
            "submission_integrity", "artifact inspector omitted artifact metadata"
        )
    return artifacts


async def _load_existing_submission_manifest(
    sandbox: SandboxHandle,
    request: SolveRequest,
    *,
    declared_paths: tuple[str, ...],
    manifest_base: Mapping[str, object],
) -> SubmissionManifest | None:
    nonce = uuid4().hex
    local_manifest = _sandbox_temp_path(
        sandbox, f"ale-atomic-existing-manifest-{request.submission_id}-{nonce}.json"
    )
    manifest_url = f"{_submission_prefix(request)}/manifest.json"
    downloaded = await _download(sandbox, manifest_url, local_manifest)
    if downloaded.returncode != 0:
        diagnostic = _command_diagnostic(downloaded)
        lowered = diagnostic.lower()
        if any(
            marker in lowered
            for marker in (
                "nosuchkey",
                "nosuchobject",
                "not found",
                "status=404",
                "status: 404",
            )
        ):
            return None
        raise AtomicInfrastructureError(
            "submission_storage",
            f"cannot inspect existing manifest: {diagnostic}",
        )
    raw = await _read_remote_bounded(
        sandbox,
        local_manifest,
        limit=_MAX_MANIFEST_BYTES,
        category="idempotency_conflict",
    )
    try:
        existing = SubmissionManifest.model_validate_json(raw)
        expected_with_existing_artifacts = SubmissionManifest.model_validate(
            dict(manifest_base) | {"artifacts": existing.artifacts}
        )
    except ValidationError as exc:
        raise AtomicInfrastructureError(
            "idempotency_conflict", f"existing or requested manifest is invalid: {exc}"
        ) from exc
    if [artifact.path for artifact in existing.artifacts] != list(
        declared_paths
    ) or existing != expected_with_existing_artifacts:
        raise AtomicInfrastructureError(
            "idempotency_conflict",
            "manifest.json already exists with conflicting submission identity",
        )
    return existing


def _validate_manifest_identity(
    manifest: SubmissionManifest,
    request: SolveRequest | EvaluateRequest,
) -> None:
    _task_card_path(request)
    identity = {
        "submission_id": request.submission_id,
        "task_path": request.task_path,
        "variant_index": request.variant_index,
        "task_commit": request.task_commit,
        "image_id": request.image_id,
    }
    if isinstance(request, SolveRequest):
        identity["agent_id"] = request.agent_id
    mismatches = [
        field for field, expected in identity.items() if getattr(manifest, field) != expected
    ]
    if mismatches:
        raise AtomicInfrastructureError(
            "submission_integrity",
            f"submission manifest identity mismatch: {', '.join(mismatches)}",
        )


async def _download(
    sandbox: SandboxHandle,
    source: str,
    destination: str,
) -> Any:
    command = _oss_call(
        sandbox,
        f"cp {_quote_oss_argument(sandbox, source)} {_quote_oss_argument(sandbox, destination)} -f",
    )
    return await sandbox.run_command(command, timeout=_COMMAND_TIMEOUT_SECONDS)


async def _read_remote_bounded(
    sandbox: SandboxHandle,
    path: str,
    *,
    limit: int,
    category: str,
) -> bytes:
    script_path, config_path = await _write_script_bundle(
        sandbox,
        stem="ale-atomic-read",
        script=_BOUNDED_READ_SCRIPT,
        config={"path": path, "limit": limit},
    )
    result = await sandbox.run_command(
        _python_command(sandbox, script_path, config_path), timeout=60
    )
    payload = _parse_script_output(result, default_category=category)
    encoded = payload.get("base64")
    if not isinstance(encoded, str):
        raise AtomicInfrastructureError(category, "bounded read omitted file bytes")
    try:
        data = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise AtomicInfrastructureError(category, "bounded read returned invalid base64") from exc
    if len(data) > limit:
        raise AtomicInfrastructureError(category, "bounded read exceeded its limit")
    return data


async def stage_submission(
    sandbox: SandboxHandle,
    task_data: TaskDataSpec,
    request: SolveRequest | EvaluateRequest,
) -> SubmissionManifest:
    """Stage a committed submission and reverify every artifact inside the VM."""
    await ossbucket.ensure_ossutil(sandbox)
    nonce = uuid4().hex
    manifest_path = _sandbox_temp_path(
        sandbox, f"ale-atomic-manifest-{request.submission_id}-{nonce}.json"
    )
    submission_prefix = _submission_prefix(request)
    manifest_url = f"{submission_prefix}/manifest.json"

    manifest_download = await _download(sandbox, manifest_url, manifest_path)
    if manifest_download.returncode != 0:
        raise AtomicInfrastructureError(
            "submission_integrity",
            f"submission manifest unavailable: {_command_diagnostic(manifest_download)}",
        )
    manifest_bytes = await _read_remote_bounded(
        sandbox,
        manifest_path,
        limit=_MAX_MANIFEST_BYTES,
        category="submission_integrity",
    )
    try:
        manifest = SubmissionManifest.model_validate_json(manifest_bytes)
    except ValidationError as exc:
        raise AtomicInfrastructureError(
            "submission_integrity", f"invalid submission manifest: {exc}"
        ) from exc
    _validate_manifest_identity(manifest, request)
    if len(manifest.artifacts) > _MAX_DECLARED_ARTIFACTS:
        raise AtomicInfrastructureError(
            "submission_integrity",
            f"manifest artifact count exceeds {_MAX_DECLARED_ARTIFACTS}",
        )

    paths: list[str] = []
    for artifact in manifest.artifacts:
        path = _validate_artifact_path(artifact.path)
        if path in paths:
            raise AtomicInfrastructureError(
                "submission_integrity", f"duplicate manifest artifact: {path}"
            )
        paths.append(path)

    output_root = _output_root(sandbox, task_data)
    verifier_config = {
        "output_root": output_root,
        "artifacts": [
            {
                "path": artifact.path,
                "relative_output_path": PurePosixPath(artifact.path)
                .relative_to("output")
                .as_posix(),
                "size_bytes": artifact.size_bytes,
                "sha256": artifact.sha256,
            }
            for artifact in manifest.artifacts
        ],
    }
    verifier_path, verifier_config_path = await _write_script_bundle(
        sandbox,
        stem="ale-atomic-verify",
        script=_VERIFY_SCRIPT,
        config=verifier_config,
    )
    prepare = await sandbox.run_command(
        _python_command(sandbox, verifier_path, verifier_config_path) + " prepare",
        timeout=60,
    )
    _parse_script_output(prepare, default_category="submission_integrity")

    for artifact in manifest.artifacts:
        relative = PurePosixPath(artifact.path).relative_to("output")
        destination = join(sandbox, output_root, *relative.parts)
        source = f"{submission_prefix}/artifacts/{artifact.path}"
        downloaded = await _download(sandbox, source, destination)
        if downloaded.returncode != 0:
            raise AtomicInfrastructureError(
                "submission_integrity",
                f"artifact download failed for {artifact.path}: {_command_diagnostic(downloaded)}",
            )

    verified = await sandbox.run_command(
        _python_command(sandbox, verifier_path, verifier_config_path) + " verify",
        timeout=_COMMAND_TIMEOUT_SECONDS,
    )
    _parse_script_output(verified, default_category="submission_integrity")
    return manifest


def _evaluator_segment(evaluator_id: str) -> str:
    if not evaluator_id or len(evaluator_id.encode("utf-8")) > _MAX_PATH_BYTES:
        raise AtomicInfrastructureError(
            "submission_integrity", "evaluator_id must be a non-empty bounded string"
        )
    return quote(evaluator_id, safe="")


async def publish_evaluation_result(
    request: EvaluateRequest,
    result: EvaluationResult,
    evidence_paths: Mapping[str, Path],
    *,
    on_result_committed: Callable[[], None] | None = None,
) -> str:
    """Publish immutable Harbor evidence and commit canonical result.json last."""
    if result.status != "scored":
        raise AtomicInfrastructureError(
            "submission_integrity",
            "only scored evaluation results may occupy the canonical result key",
        )
    expected_identity = {
        "submission_id": request.submission_id,
        "task_path": request.task_path,
        "variant_index": request.variant_index,
        "task_commit": request.task_commit,
        "image_id": request.image_id,
        "evaluator_id": request.evaluator_id,
        "evaluator_version": request.evaluator_version,
    }
    mismatches = [
        field for field, expected in expected_identity.items() if getattr(result, field) != expected
    ]
    if mismatches:
        raise AtomicInfrastructureError(
            "submission_integrity",
            f"evaluation result identity mismatch: {', '.join(mismatches)}",
        )
    result_bytes = _canonical_json(result.model_dump(mode="json"))
    if len(result_bytes) > _MAX_RESULT_BYTES:
        raise AtomicInfrastructureError(
            "submission_integrity", "evaluation result exceeds the 64 KiB limit"
        )
    if result.harbor is None:
        raise AtomicInfrastructureError(
            "submission_integrity", "scored result is missing Harbor provenance"
        )
    expected_evidence = {
        "reward.json": result.harbor.reward_sha256,
        "reward-details.json": result.harbor.details_sha256,
    }
    if set(evidence_paths) != set(expected_evidence):
        raise AtomicInfrastructureError(
            "submission_integrity", "evaluation evidence paths are incomplete"
        )

    evidence_rows: list[tuple[str, Path, int, str]] = []
    for name, expected_sha256 in expected_evidence.items():
        local_path = Path(evidence_paths[name])
        if local_path.is_symlink() or not local_path.is_file():
            raise AtomicInfrastructureError(
                "submission_integrity", f"staged {name} is missing or unsafe"
            )
        size = local_path.stat().st_size
        actual_sha256 = await asyncio.to_thread(_local_sha256, local_path)
        if actual_sha256 != expected_sha256:
            raise AtomicInfrastructureError(
                "submission_integrity", f"staged {name} SHA-256 mismatch"
            )
        evidence_rows.append((name, local_path, size, actual_sha256))

    reward_raw = evidence_rows[0][1].read_bytes()
    try:
        reward_value = json.loads(reward_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AtomicInfrastructureError(
            "submission_integrity", f"staged reward.json is invalid: {exc}"
        ) from exc
    if reward_value != result.harbor.reward:
        raise AtomicInfrastructureError(
            "submission_integrity",
            "staged reward.json does not match Harbor provenance",
        )

    result_url = (
        f"{_submission_prefix(request)}/evaluations/"
        f"{_evaluator_segment(request.evaluator_id)}/"
        f"{request.evaluator_version}/result.json"
    )
    evaluation_prefix = result_url.removesuffix("/result.json")
    for name, local_path, size, digest in evidence_rows:
        await _publish_immutable_host_object(
            local_path,
            f"{evaluation_prefix}/evidence/{name}",
            size=size,
            sha256=digest,
        )

    with tempfile.TemporaryDirectory(prefix="ale-evaluation-result-") as temp_dir:
        local_result = Path(temp_dir) / "result.json"
        local_result.write_bytes(result_bytes)
        uploaded = await run_host_ossutil(
            "cp",
            str(local_result),
            result_url,
            "--forbid-overwrite",
        )
        if uploaded[0] == 0 and on_result_committed is not None:
            on_result_committed()
    if uploaded[0] == 0:
        return result_url

    existing = await read_host_oss_object(
        result_url,
        limit=_MAX_RESULT_BYTES,
        missing_ok=True,
        integrity_category="idempotency_conflict",
    )
    if existing is None:
        raise AtomicInfrastructureError(
            "submission_storage",
            f"result upload failed: {host_command_diagnostic(uploaded)}",
        )
    try:
        EvaluationResult.model_validate_json(existing)
    except ValidationError as exc:
        raise AtomicInfrastructureError(
            "idempotency_conflict", f"existing result.json is invalid: {exc}"
        ) from exc
    if existing != result_bytes:
        raise AtomicInfrastructureError(
            "idempotency_conflict",
            "result.json already exists with different canonical bytes",
        )
    return result_url


async def _publish_immutable_host_object(
    local_path: Path,
    url: str,
    *,
    size: int,
    sha256: str,
) -> None:
    uploaded = await run_host_ossutil(
        "cp",
        str(local_path),
        url,
        "--forbid-overwrite",
        "--meta",
        f"x-oss-meta-sha256:{sha256}",
    )
    if uploaded[0] == 0:
        stat_result = await run_host_ossutil("stat", url)
        if stat_result[0] != 0:
            raise AtomicInfrastructureError(
                "submission_storage",
                f"cannot verify published evidence {url}: {host_command_diagnostic(stat_result)}",
            )
        output = stat_result[1].decode("utf-8", errors="replace")
        size_match = re.search(
            r"(?im)^\s*(?:content[- ]?length|size)\s*[:=]\s*(\d+)\s*$",
            output,
        )
        digest_match = re.search(
            r"(?i)x-oss-meta-sha256\s*[:=]\s*([0-9a-f]{64})",
            output,
        )
        if (
            size_match is None
            or digest_match is None
            or int(size_match.group(1)) != size
            or digest_match.group(1).lower() != sha256
        ):
            raise AtomicInfrastructureError(
                "submission_storage",
                f"published evidence verification mismatch: {url}",
            )
        remote = await read_host_oss_object(
            url,
            limit=size,
            missing_ok=False,
            integrity_category="submission_storage",
        )
        if remote is None or len(remote) != size or hashlib.sha256(remote).hexdigest() != sha256:
            raise AtomicInfrastructureError(
                "submission_storage",
                f"published evidence bytes mismatch: {url}",
            )
        return

    existing = await read_host_oss_object(
        url,
        limit=size,
        missing_ok=True,
        integrity_category="idempotency_conflict",
    )
    if existing is None:
        raise AtomicInfrastructureError(
            "submission_storage",
            f"evidence upload failed: {host_command_diagnostic(uploaded)}",
        )
    if len(existing) != size or hashlib.sha256(existing).hexdigest() != sha256:
        raise AtomicInfrastructureError(
            "idempotency_conflict",
            f"evidence already exists with different bytes: {url}",
        )


_INSPECT_SCRIPT = r"""
import hashlib
import json
import sys
from pathlib import Path

CHUNK_BYTES = 8 * 1024 * 1024

def fail(message):
    print(json.dumps(
        {"ok": False, "category": "submission_integrity", "error": str(message)[:1000]},
        separators=(",", ":"),
    ))
    raise SystemExit(2)

try:
    config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    output_root = Path(config["output_root"])
    if not output_root.exists() and config["allow_missing_root"]:
        print(json.dumps(
            {"ok": True, "missing_root": True},
            separators=(",", ":"),
        ))
        raise SystemExit(0)
    if output_root.is_symlink() or not output_root.is_dir():
        fail(f"output directory is missing or a symlink: {output_root}")
    resolved_root = output_root.resolve(strict=True)

    artifacts = []
    for declared_path in config["declared_paths"]:
        relative_parts = declared_path.split("/")[1:]
        candidate = output_root.joinpath(*relative_parts)
        current = output_root
        for part in relative_parts:
            current = current / part
            if current.is_symlink():
                fail(f"declared output is or traverses a symlink: {declared_path}")
        if not candidate.is_file():
            fail(f"missing required output: {declared_path}")
        resolved_candidate = candidate.resolve(strict=True)
        try:
            resolved_candidate.relative_to(resolved_root)
        except ValueError:
            fail(f"declared output escapes output directory: {declared_path}")
        digest = hashlib.sha256()
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
                digest.update(chunk)
        artifacts.append({
            "path": declared_path,
            "type": "file",
            "size_bytes": candidate.stat().st_size,
            "sha256": digest.hexdigest(),
        })
    print(json.dumps(
        {"ok": True, "artifacts": artifacts},
        separators=(",", ":"),
    ))
except SystemExit:
    raise
except Exception as exc:
    fail(exc)
""".strip()


_BOUNDED_READ_SCRIPT = r"""
import base64
import json
import sys
from pathlib import Path

try:
    config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    path = Path(config["path"])
    limit = int(config["limit"])
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise RuntimeError(f"file exceeds bounded read limit of {limit} bytes")
    print(json.dumps(
        {"ok": True, "base64": base64.b64encode(data).decode("ascii")},
        separators=(",", ":"),
    ))
except Exception as exc:
    print(json.dumps(
        {"ok": False, "error": str(exc)[:1000]},
        separators=(",", ":"),
    ))
    raise SystemExit(2)
""".strip()


_VERIFY_SCRIPT = r"""
import hashlib
import json
import sys
from pathlib import Path

CHUNK_BYTES = 8 * 1024 * 1024

def fail(message):
    print(json.dumps(
        {"ok": False, "error": str(message)[:1000]},
        separators=(",", ":"),
    ))
    raise SystemExit(2)

try:
    config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    mode = sys.argv[2]
    root = Path(config["output_root"])
    if root.exists() and root.is_symlink():
        fail(f"output root is a symlink: {root}")
    root.mkdir(parents=True, exist_ok=True)
    resolved_root = root.resolve(strict=True)

    destinations = []
    for artifact in config["artifacts"]:
        destination = root.joinpath(*artifact["relative_output_path"].split("/"))
        current = root
        for part in artifact["relative_output_path"].split("/")[:-1]:
            current = current / part
            if current.exists() and current.is_symlink():
                fail(f"artifact parent is a symlink: {artifact['path']}")
            current.mkdir(exist_ok=True)
        resolved_parent = destination.parent.resolve(strict=True)
        try:
            resolved_parent.relative_to(resolved_root)
        except ValueError:
            fail(f"artifact path escapes output root: {artifact['path']}")
        if destination.exists() and destination.is_symlink():
            fail(f"artifact is a symlink: {artifact['path']}")
        destinations.append((artifact, destination))

    if mode == "prepare":
        print(json.dumps(
            {"ok": True, "files": len(destinations)},
            separators=(",", ":"),
        ))
        raise SystemExit(0)
    if mode != "verify":
        fail("unknown verifier mode")

    total_bytes = 0
    for artifact, destination in destinations:
        if destination.is_symlink() or not destination.is_file():
            fail(f"missing or unsupported artifact: {artifact['path']}")
        size = destination.stat().st_size
        if size != artifact["size_bytes"]:
            fail(
                f"size mismatch for {artifact['path']}: "
                f"expected {artifact['size_bytes']}, got {size}"
            )
        digest = hashlib.sha256()
        with destination.open("rb") as stream:
            for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
                digest.update(chunk)
        actual_digest = digest.hexdigest()
        if actual_digest != artifact["sha256"]:
            fail(
                f"sha256 mismatch for {artifact['path']}: "
                f"expected {artifact['sha256']}, got {actual_digest}"
            )
        total_bytes += size
    print(json.dumps(
        {"ok": True, "files": len(destinations), "bytes": total_bytes},
        separators=(",", ":"),
    ))
except SystemExit:
    raise
except Exception as exc:
    fail(exc)
""".strip()


_PUBLISH_SCRIPT = r"""
import base64
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import threading
from pathlib import Path

CHUNK_BYTES = 8 * 1024 * 1024
MAX_STAT_BYTES = 64 * 1024

class ProtocolFailure(Exception):
    def __init__(self, category, message):
        self.category = category
        super().__init__(message)

def quote_argument(value):
    if not config["windows"]:
        return shlex.quote(value)
    return "'" + value.replace("'", "''") + "'"

def run_bounded(argv):
    process = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout = bytearray()
    stderr = bytearray()

    def drain(stream, destination):
        while True:
            chunk = stream.read(8192)
            if not chunk:
                return
            remaining = MAX_STAT_BYTES + 1 - len(destination)
            if remaining > 0:
                destination.extend(chunk[:remaining])

    stdout_thread = threading.Thread(
        target=drain,
        args=(process.stdout, stdout),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=drain,
        args=(process.stderr, stderr),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    try:
        returncode = process.wait(timeout=3600)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        process.wait()
        raise ProtocolFailure(
            "submission_storage", "ossutil command exceeded 3600 seconds"
        ) from exc
    finally:
        stdout_thread.join()
        stderr_thread.join()
    return subprocess.CompletedProcess(
        argv,
        returncode,
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
    )

def oss(arguments):
    command = config["oss_template"].replace(
        config["oss_argument_token"], arguments
    )
    if config["windows"]:
        prefix = 'powershell -NoProfile -Command "'
        if not command.startswith(prefix) or not command.endswith('"'):
            raise ProtocolFailure(
                "submission_integrity",
                "shared Windows OSS command has an unsupported shell shape",
            )
        powershell_script = command[len(prefix):-1]
        encoded = base64.b64encode(
            powershell_script.encode("utf-16-le")
        ).decode("ascii")
        argv = ["powershell", "-NoProfile", "-EncodedCommand", encoded]
    else:
        argv = shlex.split(command)
    result = run_bounded(argv)
    if len((result.stdout or "").encode("utf-8", errors="replace")) > MAX_STAT_BYTES:
        raise ProtocolFailure("submission_storage", "ossutil stdout exceeded 64 KiB")
    if len((result.stderr or "").encode("utf-8", errors="replace")) > MAX_STAT_BYTES:
        raise ProtocolFailure("submission_storage", "ossutil stderr exceeded 64 KiB")
    return result

def diagnostic(result):
    return (result.stderr or result.stdout or "unknown ossutil failure").strip()[:1000]

def stat_matches(url, expected_size, expected_digest):
    result = oss("stat " + quote_argument(url))
    if result.returncode != 0:
        return False, False, diagnostic(result)
    output = result.stdout or ""
    size_match = re.search(
        r"(?im)^\s*(?:content[- ]?length|size)\s*[:=]\s*(\d+)\s*$",
        output,
    )
    digest_match = re.search(
        r"(?i)[\"']?x-oss-meta-sha256[\"']?\s*[:=]\s*[\"']?([0-9a-f]{64})",
        output,
    )
    if size_match is None or digest_match is None:
        return False, True, "ossutil stat omitted size or x-oss-meta-sha256"
    actual_size = int(size_match.group(1))
    actual_digest = digest_match.group(1).lower()
    if actual_size != expected_size or actual_digest != expected_digest:
        return False, True, (
            f"remote verification mismatch for {url}: "
            f"size={actual_size} sha256={actual_digest}"
        )
    return True, True, ""

def load_existing_manifest(manifest_url, destination):
    stat_result = oss("stat " + quote_argument(manifest_url))
    if stat_result.returncode != 0:
        diagnostic_text = diagnostic(stat_result).lower()
        if any(marker in diagnostic_text for marker in (
            "nosuchkey", "nosuchobject", "not found", "status=404", "status: 404"
        )):
            return None
        raise ProtocolFailure(
            "submission_storage",
            f"cannot inspect existing manifest: {diagnostic(stat_result)}",
        )
    downloaded = oss(
        "cp "
        + quote_argument(manifest_url)
        + " "
        + quote_argument(str(destination))
        + " -f"
    )
    if downloaded.returncode != 0:
        raise ProtocolFailure(
            "submission_storage",
            f"cannot download existing manifest: {diagnostic(downloaded)}",
        )
    with destination.open("rb") as stream:
        raw = stream.read(config["max_manifest_bytes"] + 1)
    if len(raw) > config["max_manifest_bytes"]:
        raise ProtocolFailure(
            "idempotency_conflict", "existing manifest exceeds the size limit"
        )
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolFailure(
            "idempotency_conflict", f"existing manifest is invalid JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ProtocolFailure(
            "idempotency_conflict", "existing manifest is not a JSON object"
        )
    return value

try:
    config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    manifest_url = config["submission_prefix"] + "/manifest.json"
    local_manifest = Path(sys.argv[1]).with_suffix(".manifest.json")
    existing_manifest = load_existing_manifest(manifest_url, local_manifest)
    if existing_manifest is not None:
        if existing_manifest != config["expected_manifest"]:
            raise ProtocolFailure(
                "idempotency_conflict",
                "manifest.json already exists with conflicting submission data",
            )
        print(json.dumps(
            {"ok": True, "manifest": existing_manifest, "existing": True},
            ensure_ascii=False,
            separators=(",", ":"),
        ))
        raise SystemExit(0)

    output_root = Path(config["output_root"])
    if output_root.is_symlink() or not output_root.is_dir():
        raise ProtocolFailure(
            "submission_integrity", f"output directory is missing or a symlink: {output_root}"
        )
    resolved_root = output_root.resolve(strict=True)

    artifact_rows = []
    for declared_path in config["declared_paths"]:
        relative_parts = declared_path.split("/")[1:]
        candidate = output_root.joinpath(*relative_parts)
        current = output_root
        for part in relative_parts:
            current = current / part
            if current.is_symlink():
                raise ProtocolFailure(
                    "submission_integrity",
                    f"declared output is or traverses a symlink: {declared_path}",
                )
        if not candidate.is_file():
            raise ProtocolFailure(
                "submission_integrity", f"missing required output: {declared_path}"
            )
        resolved_candidate = candidate.resolve(strict=True)
        try:
            resolved_candidate.relative_to(resolved_root)
        except ValueError as exc:
            raise ProtocolFailure(
                "submission_integrity",
                f"declared output escapes output directory: {declared_path}",
            ) from exc
        digest = hashlib.sha256()
        with candidate.open("rb") as stream:
            for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
                digest.update(chunk)
        artifact_rows.append({
            "path": declared_path,
            "size_bytes": candidate.stat().st_size,
            "sha256": digest.hexdigest(),
            "local_path": str(candidate),
        })

    computed_artifacts = [
        {
            "path": row["path"],
            "size_bytes": row["size_bytes"],
            "sha256": row["sha256"],
        }
        for row in artifact_rows
    ]
    manifest = config["expected_manifest"]
    if manifest.get("artifacts") != computed_artifacts:
        raise ProtocolFailure(
            "submission_integrity",
            "artifact bytes changed after manifest prevalidation",
        )
    manifest_bytes = json.dumps(
        manifest,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(manifest_bytes) > config["max_manifest_bytes"]:
        raise ProtocolFailure(
            "submission_integrity", "generated manifest exceeds the size limit"
        )

    for row in artifact_rows:
        artifact_url = (
            config["submission_prefix"] + "/artifacts/" + row["path"]
        )
        upload = oss(
            "cp "
            + quote_argument(row["local_path"])
            + " "
            + quote_argument(artifact_url)
            + " --forbid-overwrite"
            + " --meta "
            + quote_argument("x-oss-meta-sha256:" + row["sha256"])
        )
        if upload.returncode != 0:
            matches, object_exists, reason = stat_matches(
                artifact_url, row["size_bytes"], row["sha256"]
            )
            if not matches:
                raise ProtocolFailure(
                    (
                        "idempotency_conflict"
                        if object_exists
                        else "submission_storage"
                    ),
                    f"artifact upload failed for {row['path']}: "
                    f"{diagnostic(upload)}; {reason}",
                )
        matches, _object_exists, reason = stat_matches(
            artifact_url, row["size_bytes"], row["sha256"]
        )
        if not matches:
            raise ProtocolFailure(
                "submission_storage",
                f"artifact remote verification failed for {row['path']}: {reason}",
            )

    temporary_manifest = local_manifest.with_suffix(".json.tmp")
    temporary_manifest.write_bytes(manifest_bytes)
    os.replace(temporary_manifest, local_manifest)
    commit = oss(
        "cp "
        + quote_argument(str(local_manifest))
        + " "
        + quote_argument(manifest_url)
        + " --forbid-overwrite"
    )
    if commit.returncode != 0:
        raced_manifest = load_existing_manifest(manifest_url, local_manifest)
        if raced_manifest is None:
            raise ProtocolFailure(
                "submission_storage",
                f"manifest commit failed: {diagnostic(commit)}",
            )
        if raced_manifest != manifest:
            raise ProtocolFailure(
                "idempotency_conflict",
                "manifest commit lost a race to conflicting submission data",
            )
    print(json.dumps(
        {"ok": True, "manifest": manifest, "existing": False},
        ensure_ascii=False,
        separators=(",", ":"),
    ))
except SystemExit:
    raise
except ProtocolFailure as exc:
    print(json.dumps(
        {"ok": False, "category": exc.category, "error": str(exc)[:1000]},
        ensure_ascii=False,
        separators=(",", ":"),
    ))
    raise SystemExit(2)
except Exception as exc:
    print(json.dumps(
        {"ok": False, "category": "submission_storage", "error": str(exc)[:1000]},
        ensure_ascii=False,
        separators=(",", ":"),
    ))
    raise SystemExit(2)
""".strip()
