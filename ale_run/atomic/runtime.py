"""Isolated one-shot runtime shared by independent atomic capabilities."""

from __future__ import annotations

import subprocess
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..base_interface import Provider
from ..environments.env import ALEEnv
from ..orchestration.config_loader import load_experiment
from ..orchestration.factory import EnvironmentRouter
from ..orchestration.lifecycle import _build_env_spec, _stage_task_data
from ..tasks.driver import TaskDriver
from ..tasks.loader import TaskLoader
from .contracts import AtomicInfrastructureError, EvaluateRequest, SolveRequest


@dataclass
class AtomicRuntime:
    """A provisioned task environment scoped to one atomic operation."""

    env: ALEEnv
    provider: Provider
    task_meta: dict[str, Any]
    task_data: Any
    task_dir: Path
    runtime_spec: Any
    task_driver: TaskDriver | None = None

    @classmethod
    @asynccontextmanager
    async def open(
        cls,
        *,
        request: SolveRequest | EvaluateRequest,
        run_setup: bool = True,
        provider: Provider | None = None,
    ) -> AsyncIterator[AtomicRuntime]:
        """Provision, validate, stage, and deterministically tear down one VM."""
        task_dir = request.task_repo.resolve() / "tasks" / request.task_path
        expected_commit = (
            request.task_commit if isinstance(request, SolveRequest) else request.evaluator_version
        )
        actual_commit = _repository_head(request.task_repo)
        if actual_commit != expected_commit:
            raise AtomicInfrastructureError(
                "task_checkout",
                f"task checkout mismatch: expected {expected_commit}, got {actual_commit!r}",
            )

        runtime_spec = load_experiment(request.runtime_spec_path)
        task_meta = TaskLoader(str(task_dir)).load(request.variant_index)
        env_spec = _build_env_spec(task_meta)
        runtime_provider = provider
        if runtime_provider is None:
            runtime_provider = EnvironmentRouter(runtime_spec.environment).provider_for(
                env_spec.snapshot
            )
        env = ALEEnv(provider=runtime_provider, spec=env_spec)

        try:
            await env.reset_async()
            actual_image_id = env.sandbox.metadata.get("image_id")
            if actual_image_id != request.image_id:
                raise AtomicInfrastructureError(
                    "image_identity",
                    f"image_id mismatch: expected {request.image_id}, got {actual_image_id!r}",
                )

            await _stage_task_data(
                env=env,
                provider=runtime_provider,
                artifacts=runtime_spec.artifacts,
                task_meta=task_meta,
            )
            task_driver: TaskDriver | None = None
            if run_setup:
                task_driver = TaskDriver(
                    task_path=str(task_dir),
                    session=env.session,
                    variant=request.variant_index,
                    os_type=env.sandbox.os,
                    session_rebuilder=env.reset_session,
                )
                await task_driver.setup()

            yield cls(
                env=env,
                provider=runtime_provider,
                task_meta=task_meta,
                task_data=task_meta.get("task_data"),
                task_dir=task_dir,
                runtime_spec=runtime_spec,
                task_driver=task_driver,
            )
        finally:
            await env.close_async(mode="delete")


def _repository_head(task_repo: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(task_repo), "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise AtomicInfrastructureError("task_checkout", f"cannot resolve task checkout: {detail}")
    return result.stdout.strip()
