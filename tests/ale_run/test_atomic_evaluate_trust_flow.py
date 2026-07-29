from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from importlib import import_module
from types import SimpleNamespace
from uuid import uuid4

import pytest

from ale_run.atomic.contracts import (
    ArtifactEntry,
    AtomicInfrastructureError,
    EvaluationResult,
    HarborProvenance,
    SubmissionManifest,
)
from ale_run.executors.sandbox_evaluator import SandboxEvaluationResult
from tests.ale_run.test_atomic_evaluator_registry import _registry_request


def _manifest(request) -> SubmissionManifest:
    return SubmissionManifest(
        submission_id=request.submission_id,
        task_path=request.task_path,
        variant_index=request.variant_index,
        task_commit=request.task_commit,
        image_id=request.image_id,
        ale_run_id="solve-run",
        agent_id="agent",
        model_id="model",
        config_digest="d" * 64,
        started_at=datetime(2026, 7, 29, tzinfo=UTC),
        completed_at=datetime(2026, 7, 29, 0, 1, tzinfo=UTC),
        artifacts=(
            ArtifactEntry(
                path="output/answer.json",
                size_bytes=2,
                sha256="e" * 64,
                media_type="application/json",
            ),
        ),
    )


@pytest.mark.asyncio
async def test_evaluate_gates_and_materializes_before_runtime_then_emits_v1_result(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, record_payload, registry_root = _registry_request(tmp_path)
    monkeypatch.setenv("ALE_EVALUATOR_REGISTRY_ROOT", str(registry_root))
    manifest = _manifest(request)
    evaluate_module = import_module("ale_run.atomic.evaluate")
    events: list[object] = []
    sandbox = SimpleNamespace(
        is_linux=True,
        work_dir_base="/work",
    )
    raw_reward = {
        "score": 0.75,
        "outcome": "valid",
        "report": {"hard_gate_passed": True},
    }

    async def read_manifest(actual_request):
        assert actual_request is request
        events.append("manifest")
        return manifest

    async def read_existing(actual_request, record):
        assert actual_request is request
        assert record.rubric_hash == record_payload["rubric_hash"]
        events.append("existing")

    @asynccontextmanager
    async def open_runtime(
        *,
        request: object,
        run_setup: bool,
        evaluator_registry_record: object,
    ):
        assert request.task_repo != request.task_repo.parent / "task-repo"
        assert not (request.task_repo / ".git").exists()
        assert run_setup is False
        assert evaluator_registry_record.rubric_hash == record_payload["rubric_hash"]
        events.append("runtime")
        yield SimpleNamespace(
            env=SimpleNamespace(sandbox=sandbox),
            task_data=SimpleNamespace(requires_task_data=True),
            task_dir=request.task_repo / "tasks" / request.task_path,
        )
        events.append("cleanup")

    async def stage_submission(_sandbox, _task_data, actual_request):
        assert actual_request.task_repo != request.task_repo
        events.append("submission")
        return manifest

    class Executor:
        def __init__(self, **_kwargs):
            pass

        async def stage_runtime(self):
            events.append("stage runtime")

    async def run_evaluator(**kwargs):
        assert kwargs["task_path"] != request.task_repo / "tasks" / request.task_path
        events.append("evaluator")
        return SandboxEvaluationResult(result=raw_reward, log="")

    async def publish(_sandbox, actual_request, result):
        assert actual_request is request
        events.append("publish")
        assert result.harbor is not None
        assert result.harbor.reward == raw_reward
        return "oss://canonical/result.json"

    async def record_diagnostic(*_args, **_kwargs):
        return None

    monkeypatch.setattr(evaluate_module, "_read_submission_manifest", read_manifest)
    monkeypatch.setattr(
        evaluate_module,
        "_read_existing_evaluation_result",
        read_existing,
    )
    monkeypatch.setattr(
        evaluate_module.AtomicRuntime,
        "open",
        staticmethod(open_runtime),
    )
    monkeypatch.setattr(evaluate_module, "stage_submission", stage_submission)
    monkeypatch.setattr(evaluate_module, "SandboxExecutor", Executor)
    monkeypatch.setattr(evaluate_module, "evaluate_in_sandbox", run_evaluator)
    monkeypatch.setattr(evaluate_module, "publish_evaluation_result", publish)
    monkeypatch.setattr(
        evaluate_module,
        "_record_attempt_diagnostic",
        record_diagnostic,
    )
    monkeypatch.setattr(evaluate_module, "_evaluator_environment", dict)
    monkeypatch.setattr(
        evaluate_module,
        "uuid4",
        lambda: SimpleNamespace(hex="attempt-trust-flow"),
    )

    result = await evaluate_module.evaluate(request)

    assert result == EvaluationResult(
        status="scored",
        submission_id=request.submission_id,
        task_path=request.task_path,
        variant_index=request.variant_index,
        task_commit=request.task_commit,
        image_id=request.image_id,
        evaluator_id=request.evaluator_id,
        evaluator_version=request.evaluator_version,
        outcome="valid",
        score=0.75,
        rubric_hash=record_payload["rubric_hash"],
        harbor=HarborProvenance(
            reward=raw_reward,
            details_path="evidence/reward-details.json",
        ),
    )
    assert events == [
        "manifest",
        "existing",
        "runtime",
        "submission",
        "stage runtime",
        "evaluator",
        "publish",
        "cleanup",
    ]


@pytest.mark.asyncio
async def test_evaluate_registry_failure_returns_bounded_infra_without_runtime(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, _record, registry_root = _registry_request(tmp_path)
    next(registry_root.iterdir()).unlink()
    monkeypatch.setenv("ALE_EVALUATOR_REGISTRY_ROOT", str(registry_root))
    evaluate_module = import_module("ale_run.atomic.evaluate")

    async def forbidden(*_args, **_kwargs):
        raise AssertionError("work after registry gate was invoked")

    async def record_diagnostic(*_args, **_kwargs):
        return None

    monkeypatch.setattr(evaluate_module, "_read_submission_manifest", forbidden)
    monkeypatch.setattr(
        evaluate_module.AtomicRuntime,
        "open",
        staticmethod(forbidden),
    )
    monkeypatch.setattr(
        evaluate_module,
        "_record_attempt_diagnostic",
        record_diagnostic,
    )
    monkeypatch.setattr(
        evaluate_module,
        "uuid4",
        lambda: SimpleNamespace(hex="attempt-registry-failure"),
    )

    result = await evaluate_module.evaluate(request)

    assert result.status == "infra_failed"
    assert result.submission_id == request.submission_id
    assert result.error_category == "evaluator_registry"
    assert result.attempt_id == "attempt-registry-failure"
    assert "cannot read evaluator registry record" in result.error_detail
    serialized = result.model_dump(mode="json")
    assert "outcome" not in serialized
    assert "score" not in serialized


@pytest.mark.asyncio
async def test_evaluate_rejects_self_signed_record_outside_registry_root_before_runtime(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, _record, _outside_root = _registry_request(tmp_path)
    configured_root = tmp_path / "configured-registry"
    configured_root.mkdir()
    monkeypatch.setenv("ALE_EVALUATOR_REGISTRY_ROOT", str(configured_root))
    evaluate_module = import_module("ale_run.atomic.evaluate")
    runtime_opened = False

    @asynccontextmanager
    async def forbidden_runtime(**_kwargs):
        nonlocal runtime_opened
        runtime_opened = True
        raise AssertionError("runtime acquired for unauthorized evaluator")
        yield

    async def record_diagnostic(*_args, **_kwargs):
        return None

    monkeypatch.setattr(
        evaluate_module.AtomicRuntime,
        "open",
        staticmethod(forbidden_runtime),
    )
    monkeypatch.setattr(
        evaluate_module,
        "_record_attempt_diagnostic",
        record_diagnostic,
    )

    result = await evaluate_module.evaluate(request)

    assert result.status == "infra_failed"
    assert result.error_category == "evaluator_registry"
    assert runtime_opened is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("submission_id", uuid4()),
        ("task_path", "different-task"),
        ("variant_index", 1),
        ("task_commit", "c" * 40),
        ("image_id", "m-different"),
        ("evaluator_id", "different-evaluator"),
        ("evaluator_version", "d" * 40),
        ("rubric_hash", "e" * 64),
    ],
)
async def test_existing_canonical_result_must_match_full_trusted_identity(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    request, record_payload, registry_root = _registry_request(tmp_path)
    monkeypatch.setenv("ALE_EVALUATOR_REGISTRY_ROOT", str(registry_root))
    evaluate_module = import_module("ale_run.atomic.evaluate")
    record = evaluate_module.validate_evaluator_registry(request)
    valid = EvaluationResult(
        status="scored",
        submission_id=request.submission_id,
        task_path=request.task_path,
        variant_index=request.variant_index,
        task_commit=request.task_commit,
        image_id=request.image_id,
        evaluator_id=request.evaluator_id,
        evaluator_version=request.evaluator_version,
        outcome="valid",
        score=0.75,
        rubric_hash=record_payload["rubric_hash"],
        harbor=HarborProvenance(
            reward={"score": 0.75},
            details_path="evidence/reward-details.json",
        ),
    ).model_copy(update={field: value})
    raw = json.dumps(
        valid.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    async def read_object(*_args, **_kwargs):
        return raw

    monkeypatch.setattr(evaluate_module, "_read_oss_object", read_object)

    with pytest.raises(AtomicInfrastructureError) as caught:
        await evaluate_module._read_existing_evaluation_result(request, record)

    assert caught.value.category == "idempotency_conflict"
    assert field in caught.value.message
