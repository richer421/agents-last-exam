"""``task_data_source: oss://<bucket>`` — pull task data from an Alibaba Cloud
OSS bucket.

Mirror of :mod:`ale_run.environments.task_data.s3bucket` / ``gsbucket``, using
the in-box ``ossutil`` CLI instead of ``aws s3`` / ``gsutil``.

Behavior (identical to the others):

* ``stage_input``: pull ``<oss_prefix>/input`` and ``<oss_prefix>/software`` to
  the sandbox. **Skip if already on the sandbox** (image-baked data intact).
* ``stage_reference``: always wipe + fresh sync. Reference is eval truth.

OSS auth: ECS instances launched by :class:`~ale_run.environments.providers.aliyun.AliyunProvider`
carry an **instance RAM role** (``ram_role_name``), so the in-box ``ossutil``
authenticates via STS credentials pulled from instance metadata — there is no
key to inject (contrast gcloud, which pushes an SA key into each VM; like aws,
which relies on the instance profile). The image must therefore have ``ossutil``
on PATH AND a baked ``~/.ossutilconfig`` that sets ``mode=EcsRamRole`` +
``ramRoleName=<role>`` + the region OSS ``endpoint`` (use the
``oss-<region>-internal.aliyuncs.com`` endpoint from inside ECS to avoid public
egress); the AliyunProvider images bake both.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shlex
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from ...base_interface import SandboxHandle, TaskDataSpec
from . import join, task_subdir

logger = logging.getLogger(__name__)


async def stage_input(
    sandbox: SandboxHandle, task_data: TaskDataSpec, *, source: str,
) -> dict[str, Any]:
    await _ensure_ossutil(sandbox)
    oss_prefix = _oss_prefix(source, task_data)
    base = task_subdir(sandbox, task_data)
    await sandbox.mkdir(base)

    staged: list[str] = []
    for subdir in ("input", "software"):
        dst = join(sandbox, base, subdir)
        if await _has_baked_files(sandbox, dst):
            logger.info("ossbucket: %s already present on sandbox, skipping sync", dst)
            staged.append(f"{subdir}(baked)")
            continue
        src = f"{oss_prefix}/{subdir}"
        if not await _oss_exists(sandbox, src):
            await sandbox.mkdir(dst)
            continue
        r = await sandbox.run_command(_sync_cmd(sandbox, src, dst), timeout=600)
        if r.returncode != 0:
            raise RuntimeError(
                f"ossutil sync {subdir} failed (rc={r.returncode}): "
                f"{(r.stderr or '')[:300]}"
            )
        if subdir == "software" and sandbox.is_linux:
            await sandbox.run_command(
                f"find {shlex.quote(dst)} -type f -exec chmod +x {{}} +",
                timeout=60,
            )
        staged.append(subdir)

    await sandbox.mkdir(join(sandbox, base, "output"))
    return {"staged": staged, "source": source}


async def stage_reference(
    sandbox: SandboxHandle, task_data: TaskDataSpec, *, source: str,
) -> dict[str, Any]:
    await _ensure_ossutil(sandbox)
    oss_prefix = _oss_prefix(source, task_data)
    base = task_subdir(sandbox, task_data)
    src = f"{oss_prefix}/reference"
    dst = join(sandbox, base, "reference")

    if not await _oss_exists(sandbox, src):
        return {"skipped": True, "reason": "no_reference_on_oss"}

    await sandbox.rm([dst])
    r = await sandbox.run_command(_sync_cmd(sandbox, src, dst), timeout=600)
    if r.returncode != 0:
        raise RuntimeError(
            f"ossutil sync reference failed (rc={r.returncode}): "
            f"{(r.stderr or '')[:300]}"
        )
    # ossutil sync does not preserve POSIX mode bits; normalize like gsbucket so
    # grading sees predictable perms.
    if sandbox.is_linux:
        await sandbox.run_command(f"chmod -R 777 {shlex.quote(dst)}", timeout=60)
    return {"staged": ["reference"], "source": source}


# ---- helpers ----


def _oss_prefix(source: str, task_data: TaskDataSpec) -> str:
    return (
        f"{source.rstrip('/')}/{task_data.domain_name}/"
        f"{task_data.task_name}/{task_data.variant_name}"
    )


async def _has_baked_files(sandbox: SandboxHandle, path: str) -> bool:
    if not await sandbox.exists(path):
        return False
    entries = await sandbox.list_dir(path)
    return any(not e["is_dir"] for e in entries)


# The task-data bucket is requester-pays (the puller's account is billed, not
# the owner's — mirrors gcloud's ale-data-public / aws's requester-pays bucket).
# Every read must carry --payer requester or OSS returns AccessDenied.
_RP = "--payer requester"
_WINDOWS_OSSUTIL = r"C:\Windows\Temp\ale-ossutil.exe"
_WINDOWS_OSS_ENDPOINT = "oss-ap-southeast-1-internal.aliyuncs.com"
_OSSUTIL_VERSION = "1.7.18"
_OSSUTIL_WINDOWS_URL = (
    f"https://gosspublic.alicdn.com/ossutil/{_OSSUTIL_VERSION}/ossutil64.zip"
)
_OSSUTIL_WINDOWS_ZIP_SHA256 = (
    "6604343a846717a8ac4dbd77536cc6f802f4d1e6d2d51aa88524e92b8d6a0e42"
)
_OSSUTIL_WINDOWS_EXE_SHA256 = (
    "ac5b0b40f20f380a14ef2d9ee9a8ae7f06f37789f9c75b08fbb61e1173c101a4"
)


def _windows_ossutil_candidates() -> list[Path]:
    user_cache = Path(
        os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
    ) / "ale"
    return [
        user_cache / f"ossutil-{_OSSUTIL_VERSION}/ossutil64.exe",
    ]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_windows_ossutil() -> Path:
    configured = os.environ.get("ALE_OSSUTIL_WINDOWS_BIN")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_file():
            raise RuntimeError(
                f"ALE_OSSUTIL_WINDOWS_BIN does not exist: {path}"
            )
        return path

    candidates = _windows_ossutil_candidates()
    for path in candidates:
        if path.is_file() and _file_sha256(path) == _OSSUTIL_WINDOWS_EXE_SHA256:
            return path
        path.unlink(missing_ok=True)

    destination = candidates[-1]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        archive = Path(tmp.name)
    try:
        with urllib.request.urlopen(_OSSUTIL_WINDOWS_URL, timeout=60) as response:
            payload = response.read(20 * 1024 * 1024 + 1)
        if len(payload) > 20 * 1024 * 1024:
            raise RuntimeError("ossutil bootstrap archive exceeds 20 MiB")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != _OSSUTIL_WINDOWS_ZIP_SHA256:
            raise RuntimeError(
                f"ossutil bootstrap checksum mismatch: {digest}"
            )
        archive.write_bytes(payload)
        with zipfile.ZipFile(archive) as bundle:
            member = bundle.getinfo("ossutil64/ossutil64.exe")
            if member.file_size > 20 * 1024 * 1024:
                raise RuntimeError("ossutil executable exceeds 20 MiB")
            binary = bundle.read(member)
        binary_digest = hashlib.sha256(binary).hexdigest()
        if binary_digest != _OSSUTIL_WINDOWS_EXE_SHA256:
            raise RuntimeError(
                f"ossutil executable checksum mismatch: {binary_digest}"
            )
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, suffix=".tmp", delete=False,
        ) as tmp_binary:
            staged = Path(tmp_binary.name)
            tmp_binary.write(binary)
        try:
            os.replace(staged, destination)
        finally:
            staged.unlink(missing_ok=True)
        return destination
    except Exception as exc:
        raise RuntimeError(
            "unable to bootstrap Windows ossutil; set "
            "ALE_OSSUTIL_WINDOWS_BIN to a verified ossutil executable: "
            f"{exc}"
        ) from exc
    finally:
        archive.unlink(missing_ok=True)


async def _ensure_ossutil(sandbox: SandboxHandle) -> None:
    if sandbox.is_linux:
        return
    probe = await sandbox.run_command(
        f"powershell -NoProfile -Command \"if (Test-Path -LiteralPath "
        f"'{_WINDOWS_OSSUTIL}') {{ exit 0 }} else {{ exit 1 }}\"",
        timeout=30,
    )
    if probe.returncode == 0:
        return
    binary_path = await asyncio.to_thread(_resolve_windows_ossutil)
    binary = await asyncio.to_thread(binary_path.read_bytes)
    await sandbox.write_file(_WINDOWS_OSSUTIL, binary)


def _oss_command(sandbox: SandboxHandle, arguments: str) -> str:
    if sandbox.is_linux:
        return f"ossutil {arguments}"
    metadata_url = (
        "http://100.100.100.200/latest/meta-data/ram/security-credentials/"
    )
    return (
        'powershell -NoProfile -Command "'
        f"$role=(Invoke-RestMethod -UseBasicParsing -Uri '{metadata_url}').Trim(); "
        f"& '{_WINDOWS_OSSUTIL}' {arguments} --mode EcsRamRole "
        f"--ecs-role-name $role -e {_WINDOWS_OSS_ENDPOINT}"
        '"'
    )


# Output upload and task-data staging must use the same resolved executable,
# endpoint, and RAM-role authentication contract.
ensure_ossutil = _ensure_ossutil
oss_command = _oss_command


def powershell_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


async def _oss_exists(sandbox: SandboxHandle, oss_url: str) -> bool:
    """True if the prefix has at least one object.

    ``ossutil ls`` exits 0 even for an empty prefix (unlike ``aws s3 ls``), so —
    rather than rely on the exit code — we ask for at most one object and parse
    the ``Object Number is: N`` summary line ossutil prints."""
    url = oss_url.rstrip("/") + "/"
    quoted_url = shlex.quote(url) if sandbox.is_linux else powershell_literal(url)
    cmd = _oss_command(sandbox, f"ls {_RP} {quoted_url} --limited-num 1")
    r = await sandbox.run_command(cmd, timeout=30)
    if r.returncode != 0:
        diagnostic = (r.stderr or r.stdout or "unknown ossutil failure").strip()
        raise RuntimeError(
            f"ossutil ls failed for {url} (rc={r.returncode}): {diagnostic[:300]}"
        )
    out = (r.stdout or "")
    # "Object Number is: 0" → empty; any object line starts with the oss:// url.
    if "Object Number is: 0" in out:
        return False
    return "oss://" in out or "Object Number is:" in out and "Object Number is: 0" not in out


def _sync_cmd(sandbox: SandboxHandle, src: str, dst: str) -> str:
    # ossutil sync mirrors `aws s3 sync` / `gsutil rsync`: it copies the prefix
    # tree under src into dst. Trailing slashes force directory semantics.
    src = src.rstrip("/") + "/"
    if sandbox.is_linux:
        return (
            f"mkdir -p {shlex.quote(dst)} && "
            f"ossutil sync {_RP} {shlex.quote(src)} {shlex.quote(dst)}"
        )
    return _oss_command(
        sandbox,
        f"sync {_RP} {powershell_literal(src)} {powershell_literal(dst)}",
    )
