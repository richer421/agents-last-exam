"""Bounded host-side OSS command execution for trusted atomic control planes."""

from __future__ import annotations

import asyncio
import re
import tempfile
from pathlib import Path

from .contracts import AtomicInfrastructureError

_HOST_OSS_TIMEOUT_S = 600
_MAX_HOST_OSS_OUTPUT_BYTES = 64 * 1024


async def run_host_ossutil(*arguments: str) -> tuple[int, bytes, bytes]:
    try:
        process = await asyncio.create_subprocess_exec(
            "ossutil",
            *arguments,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        raise AtomicInfrastructureError(
            "submission_storage",
            f"cannot start host ossutil: {exc}",
        ) from exc

    assert process.stdout is not None
    assert process.stderr is not None
    stdout_task = asyncio.create_task(_drain_bounded(process.stdout))
    stderr_task = asyncio.create_task(_drain_bounded(process.stderr))
    try:
        returncode = await asyncio.wait_for(
            process.wait(),
            timeout=_HOST_OSS_TIMEOUT_S,
        )
    except TimeoutError as exc:
        process.kill()
        await process.wait()
        await asyncio.gather(stdout_task, stderr_task)
        raise AtomicInfrastructureError(
            "submission_storage",
            f"host ossutil exceeded {_HOST_OSS_TIMEOUT_S} seconds",
        ) from exc
    except BaseException:
        process.kill()
        await process.wait()
        await asyncio.gather(stdout_task, stderr_task)
        raise

    stdout, stdout_exceeded = await stdout_task
    stderr, stderr_exceeded = await stderr_task
    if stdout_exceeded or stderr_exceeded:
        raise AtomicInfrastructureError(
            "submission_storage",
            "host ossutil output exceeded 64 KiB",
        )
    return returncode, stdout, stderr


async def _drain_bounded(
    stream: asyncio.StreamReader,
) -> tuple[bytes, bool]:
    kept = bytearray()
    exceeded = False
    while chunk := await stream.read(64 * 1024):
        remaining = _MAX_HOST_OSS_OUTPUT_BYTES - len(kept)
        if remaining > 0:
            kept.extend(chunk[:remaining])
        if len(chunk) > remaining:
            exceeded = True
    return bytes(kept), exceeded


def host_command_diagnostic(result: tuple[int, bytes, bytes]) -> str:
    output = result[2] or result[1]
    if not output:
        return f"ossutil exited {result[0]}"
    return output.decode("utf-8", errors="replace").strip()[:1000]


async def read_host_oss_object(
    url: str,
    *,
    limit: int,
    missing_ok: bool,
    integrity_category: str,
) -> bytes | None:
    stat_result = await run_host_ossutil("stat", url)
    if stat_result[0] != 0:
        diagnostic = host_command_diagnostic(stat_result)
        if _is_missing_object(diagnostic):
            if missing_ok:
                return None
            raise AtomicInfrastructureError(
                integrity_category,
                f"required OSS object is missing: {url}",
            )
        raise AtomicInfrastructureError(
            "submission_storage",
            f"cannot stat OSS object {url}: {diagnostic}",
        )
    stat_output = stat_result[1].decode("utf-8", errors="replace")
    size_match = re.search(
        r"(?im)^\s*(?:content[- ]?length|size)\s*[:=]\s*(\d+)\s*$",
        stat_output,
    )
    if size_match is None:
        raise AtomicInfrastructureError(
            "submission_storage",
            f"OSS stat omitted a trustworthy content length: {url}",
        )
    expected_size = int(size_match.group(1))
    if expected_size > limit:
        raise AtomicInfrastructureError(
            integrity_category,
            f"OSS object exceeds {limit} bytes: {url}",
        )
    with tempfile.TemporaryDirectory(prefix="ale-host-oss-read-") as temp_dir:
        path = Path(temp_dir) / "object"
        downloaded = await run_host_ossutil("cp", url, str(path), "-f")
        if downloaded[0] != 0:
            raise AtomicInfrastructureError(
                "submission_storage",
                f"cannot read OSS object {url}: {host_command_diagnostic(downloaded)}",
            )
        try:
            size = path.stat().st_size
            if size > limit:
                raise AtomicInfrastructureError(
                    integrity_category,
                    f"OSS object exceeds {limit} bytes: {url}",
                )
            if size != expected_size:
                raise AtomicInfrastructureError(
                    integrity_category,
                    f"OSS object size changed after stat: {url}",
                )
            return path.read_bytes()
        except AtomicInfrastructureError:
            raise
        except OSError as exc:
            raise AtomicInfrastructureError(
                "submission_storage",
                f"cannot read downloaded OSS object {url}: {exc}",
            ) from exc


def _is_missing_object(diagnostic: str) -> bool:
    lowered = diagnostic.lower()
    return any(
        marker in lowered
        for marker in (
            "nosuchkey",
            "nosuchobject",
            "not found",
            "status=404",
            "status: 404",
            "statuscode=404",
        )
    )
