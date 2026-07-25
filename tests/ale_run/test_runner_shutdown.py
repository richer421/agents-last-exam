import asyncio

import pytest

from ale_run.orchestration.runner import _gather_until_shutdown


@pytest.mark.asyncio
async def test_shutdown_cancels_workers_and_waits_for_cleanup() -> None:
    shutdown = asyncio.Event()
    started = asyncio.Event()
    cleaned = asyncio.Event()

    async def worker() -> str:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return "cancelled"
        finally:
            cleaned.set()

    task = asyncio.create_task(worker())
    gather = asyncio.create_task(
        _gather_until_shutdown([task], shutdown_event=shutdown)
    )
    await started.wait()

    shutdown.set()
    result = await asyncio.wait_for(gather, timeout=1)

    assert result == ["cancelled"]
    assert cleaned.is_set()
