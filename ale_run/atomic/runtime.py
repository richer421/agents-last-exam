"""Isolated one-shot runtime shared by independent atomic capabilities."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..base_interface import Provider
from ..environments.env import ALEEnv
from ..orchestration.config_loader import load_experiment
from ..orchestration.factory import EnvironmentRouter
from ..orchestration.lifecycle import _build_env_spec, _task_data_source
from ..tasks.driver import TaskDriver
from ..tasks.loader import TaskLoader
from .contracts import (
    AtomicInfrastructureError,
    EvaluateRequest,
    EvaluatorRegistryRecord,
    SolveRequest,
)
from .trusted_staging import (
    prepare_atomic_input,
    prepare_atomic_reference,
    sanitize_solve_environment,
    stage_atomic_input,
    stage_atomic_reference,
)

_MAX_TASK_CARD_BYTES = 1024 * 1024


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
        evaluator_registry_record: EvaluatorRegistryRecord | None = None,
    ) -> AsyncIterator[AtomicRuntime]:
        """Provision, validate, stage, and deterministically tear down one VM."""
        task_dir = request.task_repo.resolve() / "tasks" / request.task_path
        runtime_spec = load_experiment(request.runtime_spec_path)
        task_meta = TaskLoader(str(task_dir)).load(request.variant_index)
        env_spec = _build_env_spec(task_meta)
        task_data = task_meta.get("task_data")
        declared_input_paths = _declared_input_files(task_dir)
        declared_reference_paths = _declared_reference_files(task_dir)
        if declared_input_paths and (task_data is None or not task_data.requires_task_data):
            raise AtomicInfrastructureError(
                "input",
                "task declares inputFiles but no task-data staging is configured",
            )
        if evaluator_registry_record is not None and (
            task_data is None or not task_data.requires_task_data
        ):
            raise AtomicInfrastructureError(
                "reference",
                "evaluator registry requires reference staging but no task-data staging is configured",
            )
        source = _task_data_source(runtime_spec.artifacts)

        async with AsyncExitStack() as stack:
            prepared_input = None
            if task_data is not None:
                try:
                    prepared_input = await stack.enter_async_context(
                        prepare_atomic_input(
                            source=source,
                            task_data=task_data,
                            declared_input_paths=declared_input_paths,
                        )
                    )
                except AtomicInfrastructureError as exc:
                    if exc.category == "input":
                        raise
                    raise AtomicInfrastructureError(
                        "input",
                        f"trusted host input preparation failed: {exc}"[:2000],
                    ) from exc
                except Exception as exc:
                    raise AtomicInfrastructureError(
                        "input",
                        f"trusted host input preparation failed: {type(exc).__name__}: {exc}"[
                            :2000
                        ],
                    ) from exc
            prepared_reference = None
            if evaluator_registry_record is not None:
                assert task_data is not None
                try:
                    prepared_reference = await stack.enter_async_context(
                        prepare_atomic_reference(
                            record=evaluator_registry_record,
                            source=source,
                            task_data=task_data,
                            declared_reference_paths=declared_reference_paths,
                        )
                    )
                except AtomicInfrastructureError as exc:
                    if exc.category == "reference":
                        raise
                    raise AtomicInfrastructureError(
                        "reference",
                        f"trusted host reference preparation failed: {exc}"[:2000],
                    ) from exc
                except Exception as exc:
                    raise AtomicInfrastructureError(
                        "reference",
                        f"trusted host reference preparation failed: {type(exc).__name__}: {exc}"[
                            :2000
                        ],
                    ) from exc

            runtime_provider = provider
            if runtime_provider is None:
                environment = (
                    sanitize_solve_environment(runtime_spec.environment)
                    if isinstance(request, SolveRequest)
                    else runtime_spec.environment
                )
                runtime_provider = EnvironmentRouter(environment).provider_for(env_spec.snapshot)
            env = ALEEnv(provider=runtime_provider, spec=env_spec)

            try:
                await env.reset_async()
                actual_image_id = env.sandbox.metadata.get("image_id")
                if actual_image_id != request.image_id:
                    raise AtomicInfrastructureError(
                        "image_identity",
                        f"image_id mismatch: expected {request.image_id}, got {actual_image_id!r}",
                    )

                if task_data is not None:
                    await stage_atomic_input(
                        env.sandbox,
                        task_data,
                        source=source,
                        declared_input_paths=declared_input_paths,
                        prepared=prepared_input,
                    )
                if evaluator_registry_record is not None:
                    assert task_data is not None
                    assert prepared_reference is not None
                    await stage_atomic_reference(
                        env.sandbox,
                        task_data,
                        source=source,
                        declared_reference_paths=declared_reference_paths,
                        prepared=prepared_reference,
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
                    task_data=task_data,
                    task_dir=task_dir,
                    runtime_spec=runtime_spec,
                    task_driver=task_driver,
                )
            finally:
                await env.close_async(mode="delete")


def _declared_input_files(task_dir: Path) -> tuple[str, ...]:
    return _declared_task_files(task_dir, field="inputFiles", category="input")


def _declared_reference_files(task_dir: Path) -> tuple[str, ...]:
    return _declared_task_files(
        task_dir,
        field="referenceFiles",
        category="reference",
    )


def _declared_task_files(
    task_dir: Path,
    *,
    field: str,
    category: str,
) -> tuple[str, ...]:
    task_card = task_dir / "task_card.json"
    if task_card.is_symlink():
        raise AtomicInfrastructureError(category, "task_card.json is a symlink")
    try:
        with task_card.open("rb") as stream:
            raw = stream.read(_MAX_TASK_CARD_BYTES + 1)
    except OSError as exc:
        raise AtomicInfrastructureError(
            category,
            f"cannot read task_card.json {field}: {exc}",
        ) from exc
    if len(raw) > _MAX_TASK_CARD_BYTES:
        raise AtomicInfrastructureError(
            category,
            "task_card.json exceeds the 1 MiB limit",
        )
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AtomicInfrastructureError(
            category,
            f"invalid task_card.json: {exc}",
        ) from exc
    entries = payload.get(field, []) if isinstance(payload, dict) else []
    if not isinstance(entries, list):
        raise AtomicInfrastructureError(
            category,
            f"task_card.json {field} must be a list",
        )
    declared: list[str] = []
    for entry in entries:
        path = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(path, str) or not path:
            raise AtomicInfrastructureError(
                category,
                f"each {field} entry must have a non-empty string path",
            )
        if path in declared:
            raise AtomicInfrastructureError(
                category,
                f"duplicate declared {field} path: {path}",
            )
        declared.append(path)
    return tuple(declared)
