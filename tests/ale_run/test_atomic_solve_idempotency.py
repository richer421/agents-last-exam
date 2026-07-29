from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest

from ale_run.atomic.contracts import (
    ArtifactEntry,
    AtomicInfrastructureError,
    SubmissionManifest,
)
from ale_run.orchestration.experiment_spec import AgentSpec
from tests.ale_run.test_atomic_solve import (
    _make_solve_request,
    _TestAgentConfig,
    _TestDeployer,
)


def _config_digest(config: _TestAgentConfig) -> str:
    return hashlib.sha256(
        json.dumps(
            dataclasses.asdict(config),
            default=str,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _committed_manifest(request, config_digest: str) -> SubmissionManifest:
    return SubmissionManifest(
        status="submitted",
        submission_id=request.submission_id,
        task_path=request.task_path,
        variant_index=request.variant_index,
        task_commit=request.task_commit,
        image_id=request.image_id,
        ale_run_id="lost-response-run",
        agent_id=request.agent_id,
        model_id="test-model",
        config_digest=config_digest,
        started_at=datetime(2026, 7, 29, tzinfo=UTC),
        completed_at=datetime(2026, 7, 29, 0, 1, tzinfo=UTC),
        artifacts=(
            ArtifactEntry(
                path="output/answer.txt",
                size_bytes=6,
                sha256="b" * 64,
                media_type="text/plain",
            ),
        ),
    )


@pytest.mark.asyncio
async def test_lost_response_retry_returns_manifest_without_second_vm_or_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _make_solve_request(tmp_path)
    solve_module = import_module("ale_run.atomic.solve")
    config = _TestAgentConfig()
    manifest = _committed_manifest(request, _config_digest(config))
    selected = AgentSpec(id=request.agent_id, class_="test", config={})
    events: list[str] = []

    monkeypatch.setattr(
        solve_module,
        "load_experiment",
        lambda _path: SimpleNamespace(agents=[selected]),
        raising=False,
    )
    monkeypatch.setattr(
        solve_module,
        "resolve_agent",
        lambda _spec: (_TestDeployer, _TestAgentConfig),
    )
    monkeypatch.setattr(solve_module, "build_config", lambda *_args: config)

    async def read_manifest(actual_request):
        events.append("manifest preflight")
        assert actual_request is request
        return manifest

    monkeypatch.setattr(
        solve_module,
        "read_existing_submission_manifest",
        read_manifest,
        raising=False,
    )
    monkeypatch.setattr(
        solve_module.AtomicRuntime,
        "open",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("VM acquired")),
    )
    monkeypatch.setattr(
        solve_module,
        "publish_submission",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("publisher called")),
    )

    result = await solve_module.solve(request)

    assert result.status == "submitted"
    assert result.manifest == manifest
    assert events == ["manifest preflight"]


@pytest.mark.asyncio
async def test_lost_response_retry_bypasses_new_dirty_repository_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _make_solve_request(tmp_path)
    solve_module = import_module("ale_run.atomic.solve")
    config = _TestAgentConfig()
    manifest = _committed_manifest(request, _config_digest(config))
    selected = AgentSpec(id=request.agent_id, class_="test", config={})
    (request.task_repo / "tasks" / request.task_path / "untracked.txt").write_text(
        "retry-local state",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        solve_module,
        "load_experiment",
        lambda _path: SimpleNamespace(agents=[selected]),
    )
    monkeypatch.setattr(
        solve_module,
        "resolve_agent",
        lambda _spec: (_TestDeployer, _TestAgentConfig),
    )
    monkeypatch.setattr(solve_module, "build_config", lambda *_args: config)

    async def read_manifest(_request):
        return manifest

    monkeypatch.setattr(solve_module, "read_existing_submission_manifest", read_manifest)
    monkeypatch.setattr(
        solve_module.AtomicRuntime,
        "open",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("provider acquired")),
    )

    result = await solve_module.solve(request)

    assert result.status == "submitted"
    assert result.manifest == manifest


@pytest.mark.asyncio
async def test_occupied_submission_key_with_different_agent_config_fails_before_vm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _make_solve_request(tmp_path)
    solve_module = import_module("ale_run.atomic.solve")
    config = _TestAgentConfig()
    selected = AgentSpec(id=request.agent_id, class_="test", config={})
    conflicting = _committed_manifest(request, "f" * 64)

    monkeypatch.setattr(
        solve_module,
        "load_experiment",
        lambda _path: SimpleNamespace(agents=[selected]),
        raising=False,
    )
    monkeypatch.setattr(
        solve_module,
        "resolve_agent",
        lambda _spec: (_TestDeployer, _TestAgentConfig),
    )
    monkeypatch.setattr(solve_module, "build_config", lambda *_args: config)

    async def read_manifest(_request):
        return conflicting

    monkeypatch.setattr(
        solve_module,
        "read_existing_submission_manifest",
        read_manifest,
        raising=False,
    )
    monkeypatch.setattr(
        solve_module.AtomicRuntime,
        "open",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("VM acquired")),
    )

    with pytest.raises(AtomicInfrastructureError) as caught:
        await solve_module.solve(request)

    assert caught.value.category == "idempotency_conflict"
