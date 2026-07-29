"""Independent one-shot solve capability."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ..orchestration.config_loader import load_experiment
from ..orchestration.factory import build_config, resolve_agent
from ..orchestration.lifecycle import _DEFAULT_TIMEOUT_S, _build_executor
from .contracts import AtomicInfrastructureError, SolveRequest, SolveResult
from .oss_submission import (
    publish_submission,
    read_existing_submission_manifest,
)
from .runtime import AtomicRuntime


async def solve(request: SolveRequest) -> SolveResult:
    """Run one configured agent and atomically publish its declared outputs."""
    started_at = datetime.now(UTC)
    run_id = uuid4().hex
    agent_spec, deployer_cls, config = _resolve_requested_agent(request)
    config_digest = _agent_config_digest(config)
    existing = await read_existing_submission_manifest(request)
    if existing is not None:
        _validate_idempotent_manifest(
            existing,
            request,
            config_digest=config_digest,
        )
        return SolveResult(
            status="submitted",
            submission_id=request.submission_id,
            manifest=existing,
        )

    with tempfile.TemporaryDirectory(prefix="ale-atomic-solve-") as work_dir:
        async with AtomicRuntime.open(request=request) as runtime:
            executor_type = (
                agent_spec.executor or getattr(deployer_cls, "default_executor", "") or "sandbox"
            )
            executor = _build_executor(
                executor_type=executor_type,
                env=runtime.env,
                config=config,
                agent_name=getattr(config, "name", "agent"),
                run_id=run_id,
                host_artifacts_dir=Path(work_dir),
            )
            run_result = await executor.run_deployer(
                deployer_cls=deployer_cls,
                prompt=runtime.task_meta["description"],
                timeout_s=float(_DEFAULT_TIMEOUT_S),
            )
            if run_result.status != "completed":
                return SolveResult(
                    status="failed",
                    submission_id=request.submission_id,
                    error=run_result.error or run_result.status,
                )

            manifest = await publish_submission(
                runtime.env.sandbox,
                runtime.task_data,
                request,
                provenance={
                    "ale_run_id": run_id,
                    "model_id": str(config.model),
                    "config_digest": config_digest,
                    "started_at": started_at,
                    "completed_at": datetime.now(UTC),
                },
            )
            return SolveResult(
                status="submitted",
                submission_id=request.submission_id,
                manifest=manifest,
            )


def _resolve_requested_agent(request: SolveRequest):
    runtime_spec = load_experiment(request.runtime_spec_path)
    matching_agents = [agent for agent in runtime_spec.agents if agent.id == request.agent_id]
    if len(matching_agents) != 1:
        raise ValueError(
            f"solve request requires exactly one matching agent_id {request.agent_id!r}; "
            f"found {len(matching_agents)}"
        )
    agent_spec = matching_agents[0]
    deployer_cls, config_cls = resolve_agent(agent_spec)
    return agent_spec, deployer_cls, build_config(config_cls, agent_spec.config)


def _agent_config_digest(config: object) -> str:
    config_payload = (
        dataclasses.asdict(config) if dataclasses.is_dataclass(config) else vars(config)
    )
    return hashlib.sha256(
        json.dumps(
            config_payload,
            default=str,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _validate_idempotent_manifest(
    manifest,
    request: SolveRequest,
    *,
    config_digest: str,
) -> None:
    identity = {
        "status": "submitted",
        "submission_id": request.submission_id,
        "task_path": request.task_path,
        "variant_index": request.variant_index,
        "task_commit": request.task_commit,
        "image_id": request.image_id,
        "agent_id": request.agent_id,
        "config_digest": config_digest,
    }
    mismatches = [
        field for field, expected in identity.items() if getattr(manifest, field) != expected
    ]
    if mismatches:
        raise AtomicInfrastructureError(
            "idempotency_conflict",
            f"submission manifest already occupies the key with conflicting "
            f"identity: {', '.join(mismatches)}",
        )
