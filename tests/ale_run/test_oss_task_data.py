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
async def test_ensure_ossutil_requires_windows_bootstrap_binary(monkeypatch) -> None:
    monkeypatch.delenv("ALE_OSSUTIL_WINDOWS_BIN", raising=False)
    sandbox = SimpleNamespace(
        is_linux=False,
        run_command=AsyncMock(
            return_value=SimpleNamespace(returncode=1, stdout="", stderr="")
        ),
    )

    with pytest.raises(RuntimeError, match="ALE_OSSUTIL_WINDOWS_BIN"):
        await _ensure_ossutil(sandbox)
