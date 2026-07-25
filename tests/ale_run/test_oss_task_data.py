from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ale_run.environments.task_data.ossbucket import (
    _ensure_ossutil,
    _oss_command,
    _oss_exists,
)


@pytest.mark.asyncio
async def test_oss_exists_surfaces_command_failures() -> None:
    sandbox = SimpleNamespace(
        is_linux=False,
        run_command=AsyncMock(
            return_value=SimpleNamespace(
                returncode=1,
                stdout="",
                stderr="AccessDenied: RAM role is not authorized",
            )
        ),
    )

    with pytest.raises(RuntimeError, match="AccessDenied"):
        await _oss_exists(sandbox, "oss://ale-artifacts/domain/task/variant/input")


def test_windows_oss_command_uses_ecs_role_and_internal_endpoint() -> None:
    sandbox = SimpleNamespace(is_linux=False)

    command = _oss_command(sandbox, "ls --payer requester 'oss://bucket/input/'")

    assert r"C:\Windows\Temp\ale-ossutil.exe" in command
    assert "--mode EcsRamRole" in command
    assert "--ecs-role-name $role" in command
    assert "oss-ap-southeast-1-internal.aliyuncs.com" in command


@pytest.mark.asyncio
async def test_ensure_ossutil_uses_runtime_cache_without_env(
    monkeypatch, tmp_path,
) -> None:
    monkeypatch.delenv("ALE_OSSUTIL_WINDOWS_BIN", raising=False)
    binary = tmp_path / "ossutil64.exe"
    binary.write_bytes(b"ossutil")
    monkeypatch.setattr(
        "ale_run.environments.task_data.ossbucket._OSSUTIL_WINDOWS_EXE_SHA256",
        "2d9ccdf7354fcc9afd5651bb7c5c2d1e04a1fd4ebd1612a0786142af43e3c66a",
    )
    monkeypatch.setattr(
        "ale_run.environments.task_data.ossbucket._windows_ossutil_candidates",
        lambda: [binary],
    )
    sandbox = SimpleNamespace(
        is_linux=False,
        run_command=AsyncMock(
            return_value=SimpleNamespace(returncode=1, stdout="", stderr="")
        ),
        write_file=AsyncMock(),
    )

    await _ensure_ossutil(sandbox)

    sandbox.write_file.assert_awaited_once_with(
        r"C:\Windows\Temp\ale-ossutil.exe", b"ossutil"
    )
