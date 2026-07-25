from types import SimpleNamespace

import pytest

from ale_run.base_interface import SandboxSpec
from ale_run.environments.env import ALEEnv


@pytest.mark.asyncio
async def test_cleanup_retries_release_before_dropping_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    class Provider:
        async def release(self, sandbox, *, mode):
            nonlocal calls
            calls += 1
            if calls < 3:
                raise RuntimeError("temporary delete failure")

    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("ale_run.environments.env.asyncio.sleep", no_sleep)
    env = ALEEnv(
        provider=Provider(),
        spec=SandboxSpec(snapshot="test", os="windows"),
    )
    env._sandbox = SimpleNamespace(id="instance-1")

    await env.close_async(mode="delete")

    assert calls == 3
    assert env._sandbox is None


@pytest.mark.asyncio
async def test_cleanup_keeps_handle_when_all_release_attempts_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Provider:
        async def release(self, sandbox, *, mode):
            raise RuntimeError("delete failed")

    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("ale_run.environments.env.asyncio.sleep", no_sleep)
    env = ALEEnv(
        provider=Provider(),
        spec=SandboxSpec(snapshot="test", os="windows"),
    )
    handle = SimpleNamespace(id="instance-2")
    env._sandbox = handle

    with pytest.raises(RuntimeError, match="delete failed"):
        await env.close_async(mode="delete")

    assert env._sandbox is handle
