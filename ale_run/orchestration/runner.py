"""Runner: yaml-described experiment → concurrent run units.

Per-unit isolation:
    - One fresh ``ale.make(task_path)`` per unit (env binds task at ctor)
    - One fresh deployer instance per unit (configs are per-run state)

Concurrency is a single ``asyncio.Semaphore`` sized to ``spec.concurrency``
(matches simprun's one-knob model). Each unit holds the slot for its
full lifetime — VM acquire + agent run + post-launch fan-out + eval —
so the cap is effectively "max VMs alive at once". Size to
``min(GCP quota, LLM rate-limit / N)``.

Provider is shared across units — real providers (gcloud) acquire
a fresh VM per ``acquire()`` call, so concurrent acquires give concurrent
VMs.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Iterable

from .factory import EnvironmentRouter
from .experiment_spec import ExperimentSpec, RunUnit, UnitResult

logger = logging.getLogger(__name__)


async def _gather_until_shutdown(
    tasks: list[asyncio.Task],
    *,
    shutdown_event: asyncio.Event,
) -> list[Any]:
    combined = asyncio.gather(*tasks)
    shutdown_wait = asyncio.create_task(shutdown_event.wait())
    done, _ = await asyncio.wait(
        {combined, shutdown_wait},
        return_when=asyncio.FIRST_COMPLETED,
    )
    if shutdown_wait in done and shutdown_event.is_set():
        for task in tasks:
            if not task.done():
                task.cancel()
        return list(await combined)
    shutdown_wait.cancel()
    await asyncio.gather(shutdown_wait, return_exceptions=True)
    return list(await combined)


class Runner:
    """Owns the provider; produces and executes run units."""

    def __init__(self, spec: ExperimentSpec):
        self._spec = spec
        # Router resolves each unit's snapshot to its provider, building +
        # caching provider instances lazily (keeps --dry-run provider-free).
        self._router = EnvironmentRouter(spec.environment)
        self._output_root = Path(spec.output.root) / spec.name

    @property
    def spec(self) -> ExperimentSpec:
        return self._spec

    @property
    def output_root(self) -> Path:
        return self._output_root

    # ---- enumeration ----

    def enumerate_units(self) -> list[RunUnit]:
        """Cartesian product of agents × tasks × variants."""
        out: list[RunUnit] = []
        for agent in self._spec.agents:
            for task in self._spec.tasks:
                for vi in task.variants:
                    out.append(RunUnit(
                        agent_id=agent.id,
                        agent_spec=agent,
                        task_path=task.path,
                        variant_index=vi,
                    ))
        return out

    # ---- execution ----

    async def run(
        self,
        units: Iterable[RunUnit] | None = None,
        *,
        max_attempts: int = 1,
    ) -> list[UnitResult]:
        """Run all units (or a filtered subset). Returns ``list[UnitResult]``.

        Failed units are requeued until they succeed or reach ``max_attempts``.
        Only the final result for each unit is returned; every attempt retains
        its own run directory on disk.
        """
        from .lifecycle import (
            get_shutdown_event,
            install_signal_handlers,
            run_one_unit,
        )

        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        install_signal_handlers()
        shutdown_event = get_shutdown_event()
        shutdown_event.clear()
        unit_list = list(units) if units is not None else self.enumerate_units()
        if not unit_list:
            logger.warning("Runner.run: no units to execute")
            return []

        self._output_root.mkdir(parents=True, exist_ok=True)

        n = self._spec.concurrency
        sem = asyncio.Semaphore(n)
        logger.info("runner: %d units, concurrency=%d", len(unit_list), n)

        async def _drive(u: RunUnit) -> UnitResult:
            for attempt in range(1, max_attempts + 1):
                logger.info(
                    "[%s] queued (attempt %d/%d)", u.slug, attempt, max_attempts,
                )
                result = await run_one_unit(
                    unit=u,
                    router=self._router,
                    output_root=self._output_root,
                    artifacts=self._spec.artifacts,
                    sem=sem,
                    cleanup_mode=self._spec.cleanup_mode,
                    prompt_suffix=self._spec.prompt_suffix,
                    wall_time_s=self._spec.wall_time_s,
                )
                logger.info(
                    "[%s] done: status=%s score=%s duration=%.1fs (attempt %d/%d)",
                    u.slug,
                    result.status,
                    result.score,
                    result.duration_s or 0,
                    attempt,
                    max_attempts,
                )
                if result.status != "failed" or attempt == max_attempts:
                    return result
                logger.warning(
                    "[%s] failed on attempt %d/%d; requeued",
                    u.slug,
                    attempt,
                    max_attempts,
                )

            raise AssertionError("unreachable")

        tasks = [
            asyncio.create_task(_drive(u), name=f"ale-unit:{u.slug}")
            for u in unit_list
        ]
        results = await _gather_until_shutdown(
            tasks,
            shutdown_event=shutdown_event,
        )
        return list(results)
