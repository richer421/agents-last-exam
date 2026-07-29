from __future__ import annotations

import asyncio
import subprocess
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from ale_run.atomic.contracts import (
    ArtifactEntry,
    AtomicInfrastructureError,
    EvaluateRequest,
    EvaluationResult,
    SubmissionManifest,
)
from ale_run.atomic.runtime import AtomicRuntime
from ale_run.base_interface import TaskDataSpec
from ale_run.executors.sandbox_evaluator import SandboxEvaluationResult
from tests.ale_run.test_atomic_solve import _FakeProvider, _make_solve_request


def _make_evaluate_request(tmp_path: Path) -> EvaluateRequest:
    solve_request = _make_solve_request(tmp_path)
    return EvaluateRequest(
        submission_id=solve_request.submission_id,
        runtime_spec_path=solve_request.runtime_spec_path,
        task_repo=solve_request.task_repo,
        task_path=solve_request.task_path,
        variant_index=solve_request.variant_index,
        task_commit=solve_request.task_commit,
        image_id=solve_request.image_id,
        submission_root=solve_request.submission_root,
        evaluator_id="strict-evaluator",
        evaluator_version=solve_request.task_commit,
    )


def _manifest(request: EvaluateRequest) -> SubmissionManifest:
    return SubmissionManifest(
        submission_id=request.submission_id,
        task_path=request.task_path,
        variant_index=request.variant_index,
        task_commit=request.task_commit,
        image_id=request.image_id,
        ale_run_id="solve-run",
        agent_id="solve-agent",
        model_id="solve-model",
        config_digest="a" * 64,
        started_at=datetime(2026, 7, 29, tzinfo=UTC),
        completed_at=datetime(2026, 7, 29, 0, 1, tzinfo=UTC),
        artifacts=(
            ArtifactEntry(
                path="output/final.txt",
                size_bytes=6,
                sha256="b" * 64,
            ),
        ),
    )


def _runtime(request: EvaluateRequest) -> SimpleNamespace:
    sandbox = SimpleNamespace(
        id="sandbox-evaluate",
        is_linux=True,
        os="linux",
        metadata={"image_id": request.image_id},
        work_dir_base="/home/user/.ale",
        task_data_root="/data",
        cua_server_port=5000,
    )
    return SimpleNamespace(
        env=SimpleNamespace(sandbox=sandbox),
        provider=SimpleNamespace(),
        task_meta={},
        task_data=TaskDataSpec(
            requires_task_data=True,
            domain_name="demo",
            task_name="toy",
            variant_name="base",
        ),
        task_dir=request.task_repo / "tasks" / request.task_path,
        runtime_spec=SimpleNamespace(
            artifacts=SimpleNamespace(task_data_source="baked_in_sandbox")
        ),
        task_driver=None,
    )


def _patch_happy_path(
    monkeypatch: pytest.MonkeyPatch,
    request: EvaluateRequest,
    *,
    sandbox_result: dict[str, object] | None = None,
) -> tuple[object, list[str], list[tuple[str, object]]]:
    evaluate_module = import_module("ale_run.atomic.evaluate")
    events: list[str] = []
    diagnostics: list[tuple[str, object]] = []
    manifest = _manifest(request)
    runtime = _runtime(request)

    async def read_manifest(actual_request: EvaluateRequest) -> SubmissionManifest:
        events.append("read manifest")
        assert actual_request is request
        return manifest

    async def read_result(actual_request: EvaluateRequest) -> None:
        assert actual_request is request

    @asynccontextmanager
    async def open_runtime(*, request: EvaluateRequest, run_setup: bool):
        events.extend(["open fresh runtime", "stage input"])
        assert request is request_under_test
        assert run_setup is False
        assert runtime.env.sandbox.metadata["image_id"] == request.image_id
        try:
            yield runtime
        finally:
            events.append("cleanup")

    async def stage_reference(_sandbox, _task_data, *, source: str):
        events.append("stage reference")
        assert source == "baked_in_sandbox"
        return {"staged": ["reference"], "source": source}

    async def stage_submission(_sandbox, _task_data, actual_request):
        events.append("stage submission")
        assert actual_request is request
        return manifest

    class Executor:
        def __init__(self, *, config, work_dir, sandbox, env):
            assert config is None
            assert sandbox is runtime.env.sandbox
            assert env == {}
            assert str(request.submission_id) not in work_dir

        async def stage_runtime(self):
            events.append("stage runtime")

        async def run_deployer(self, **_kwargs):
            raise AssertionError("evaluate must never run a deployer")

    async def run_evaluator(**kwargs):
        events.append("run evaluator")
        assert kwargs["sandbox"] is runtime.env.sandbox
        assert kwargs["task_path"] == runtime.task_dir
        assert kwargs["variant"] == request.variant_index
        assert kwargs["evaluator_env"] == {"OPENAI_API_KEY": "judge-key"}
        return SandboxEvaluationResult(
            result=sandbox_result
            or {
                "score": 0.75,
                "outcome": "valid",
                "report": {"hard_gate_passed": True},
            },
            log="bounded evaluator log",
        )

    async def publish(_sandbox, actual_request, result):
        events.append("publish result")
        assert actual_request is request
        return "oss://canonical/result.json"

    async def record_diagnostic(actual_request, attempt_id, error):
        diagnostics.append((attempt_id, error))
        assert actual_request is request

    request_under_test = request
    monkeypatch.setattr(evaluate_module, "_read_submission_manifest", read_manifest)
    monkeypatch.setattr(evaluate_module, "_read_existing_evaluation_result", read_result)
    monkeypatch.setattr(evaluate_module.AtomicRuntime, "open", open_runtime)
    monkeypatch.setattr(
        evaluate_module.task_data_pkg,
        "select",
        lambda source: SimpleNamespace(stage_reference=stage_reference),
    )
    monkeypatch.setattr(evaluate_module, "stage_submission", stage_submission)
    monkeypatch.setattr(evaluate_module, "SandboxExecutor", Executor)
    monkeypatch.setattr(evaluate_module, "evaluate_in_sandbox", run_evaluator)
    monkeypatch.setattr(evaluate_module, "publish_evaluation_result", publish)
    monkeypatch.setattr(evaluate_module, "_record_attempt_diagnostic", record_diagnostic)
    monkeypatch.setattr(
        evaluate_module,
        "_evaluator_environment",
        lambda: {"OPENAI_API_KEY": "judge-key"},
    )
    monkeypatch.setattr(
        "ale_run.orchestration.factory.resolve_agent",
        lambda _spec: (_ for _ in ()).throw(AssertionError("agent resolved")),
    )
    return evaluate_module, events, diagnostics


@pytest.mark.asyncio
async def test_evaluate_runs_precise_independent_order_and_returns_normal_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _make_evaluate_request(tmp_path)
    evaluate_module, events, diagnostics = _patch_happy_path(monkeypatch, request)

    result = await evaluate_module.evaluate(request)

    assert result == EvaluationResult(status="scored", outcome="valid", score=0.75)
    assert events == [
        "read manifest",
        "open fresh runtime",
        "stage input",
        "stage reference",
        "stage submission",
        "stage runtime",
        "run evaluator",
        "publish result",
        "cleanup",
    ]
    assert diagnostics == []


@pytest.mark.asyncio
async def test_evaluate_maps_only_a_legal_hard_gate_to_invalid_output_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _make_evaluate_request(tmp_path)
    evaluate_module, events, diagnostics = _patch_happy_path(
        monkeypatch,
        request,
        sandbox_result={
            "score": 0.0,
            "outcome": "valid",
            "report": {"hard_gate_passed": False},
        },
    )

    result = await evaluate_module.evaluate(request)

    assert result == EvaluationResult(
        status="scored",
        outcome="invalid_output",
        score=0.0,
    )
    assert "publish result" in events
    assert diagnostics == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("submission_id", uuid4()),
        ("task_path", "different-task"),
        ("variant_index", 1),
        ("task_commit", "c" * 40),
        ("image_id", "m-different"),
        ("agent_id", ""),
    ],
)
async def test_evaluate_rejects_manifest_request_or_agent_provenance_before_vm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    request = _make_evaluate_request(tmp_path)
    evaluate_module = import_module("ale_run.atomic.evaluate")
    diagnostics: list[AtomicInfrastructureError] = []
    mismatched = _manifest(request).model_copy(update={field: value})

    async def read_manifest(_request):
        return mismatched

    async def no_result(_request):
        return None

    async def record(_request, _attempt_id, error):
        diagnostics.append(error)

    monkeypatch.setattr(evaluate_module, "_read_submission_manifest", read_manifest)
    monkeypatch.setattr(evaluate_module, "_read_existing_evaluation_result", no_result)
    monkeypatch.setattr(evaluate_module, "_record_attempt_diagnostic", record)
    monkeypatch.setattr(
        evaluate_module.AtomicRuntime,
        "open",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("VM provisioned")),
    )

    result = await evaluate_module.evaluate(request)

    assert result == EvaluationResult(status="infra_failed")
    assert len(diagnostics) == 1
    assert diagnostics[0].category == "submission_integrity"


@pytest.mark.asyncio
@pytest.mark.parametrize("backend_failure", ["skipped", "exception"])
async def test_declared_reference_is_strict_and_missing_data_is_infrastructure_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend_failure: str,
) -> None:
    request = _make_evaluate_request(tmp_path)
    task_card = request.task_repo / "tasks" / request.task_path / "task_card.json"
    task_card.write_text(
        '{"vm":{"snapshot":"test-image"},"referenceFiles":[{"path":"reference/answer.json"}]}',
        encoding="utf-8",
    )
    evaluate_module, events, diagnostics = _patch_happy_path(monkeypatch, request)

    async def fail_reference(_sandbox, _task_data, *, source: str):
        events.append("stage reference")
        if backend_failure == "exception":
            raise RuntimeError("reference transport failed")
        return {"skipped": True, "reason": "missing reference"}

    monkeypatch.setattr(
        evaluate_module.task_data_pkg,
        "select",
        lambda source: SimpleNamespace(stage_reference=fail_reference),
    )

    result = await evaluate_module.evaluate(request)

    assert result == EvaluationResult(status="infra_failed")
    assert "stage submission" not in events
    assert "run evaluator" not in events
    assert events[-1] == "cleanup"
    assert len(diagnostics) == 1
    assert diagnostics[0][1].category == "reference"


@pytest.mark.asyncio
async def test_artifact_hash_mismatch_never_runs_or_publishes_evaluator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _make_evaluate_request(tmp_path)
    evaluate_module, events, diagnostics = _patch_happy_path(monkeypatch, request)

    async def reject_hash(*_args):
        events.append("stage submission")
        raise AtomicInfrastructureError("submission_integrity", "artifact sha256 mismatch")

    monkeypatch.setattr(evaluate_module, "stage_submission", reject_hash)

    result = await evaluate_module.evaluate(request)

    assert result == EvaluationResult(status="infra_failed")
    assert "run evaluator" not in events
    assert "publish result" not in events
    assert events[-1] == "cleanup"
    assert diagnostics[0][1].category == "submission_integrity"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        TimeoutError("evaluator exceeded 10 seconds"),
        RuntimeError("sandbox evaluator worker crashed"),
    ],
)
async def test_evaluator_timeout_or_runtime_failure_is_unscored_infrastructure_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
) -> None:
    request = _make_evaluate_request(tmp_path)
    evaluate_module, events, diagnostics = _patch_happy_path(monkeypatch, request)

    async def fail_evaluator(**_kwargs):
        events.append("run evaluator")
        raise failure

    monkeypatch.setattr(evaluate_module, "evaluate_in_sandbox", fail_evaluator)

    result = await evaluate_module.evaluate(request)

    assert result == EvaluationResult(status="infra_failed", outcome=None, score=None)
    assert "publish result" not in events
    assert events[-1] == "cleanup"
    assert len(diagnostics) == 1
    assert diagnostics[0][1].category == "evaluator"


@pytest.mark.asyncio
async def test_evaluator_error_result_is_not_converted_to_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _make_evaluate_request(tmp_path)
    evaluate_module, events, diagnostics = _patch_happy_path(
        monkeypatch,
        request,
        sandbox_result={"error": "judge API unavailable"},
    )

    result = await evaluate_module.evaluate(request)

    assert result == EvaluationResult(status="infra_failed")
    assert "publish result" not in events
    assert diagnostics[0][1].category == "evaluator"


@pytest.mark.asyncio
async def test_existing_valid_result_returns_without_vm_provisioning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _make_evaluate_request(tmp_path)
    evaluate_module = import_module("ale_run.atomic.evaluate")
    existing = EvaluationResult(status="scored", outcome="valid", score=0.625)
    events: list[str] = []

    async def read_manifest(_request):
        events.append("read manifest")
        return _manifest(request)

    async def read_result(_request):
        events.append("read result")
        return existing

    monkeypatch.setattr(evaluate_module, "_read_submission_manifest", read_manifest)
    monkeypatch.setattr(evaluate_module, "_read_existing_evaluation_result", read_result)
    monkeypatch.setattr(
        evaluate_module.AtomicRuntime,
        "open",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("VM provisioned")),
    )
    monkeypatch.setattr(
        evaluate_module,
        "_record_attempt_diagnostic",
        lambda *_args: (_ for _ in ()).throw(AssertionError("diagnostic written")),
    )

    result = await evaluate_module.evaluate(request)

    assert result is existing
    assert events == ["read manifest", "read result"]


@pytest.mark.asyncio
async def test_diagnostic_write_failure_cannot_disguise_infrastructure_failure_as_score(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _make_evaluate_request(tmp_path)
    evaluate_module, events, _diagnostics = _patch_happy_path(monkeypatch, request)

    async def fail_evaluator(**_kwargs):
        events.append("run evaluator")
        raise RuntimeError("evaluator transport failed")

    async def fail_diagnostic(*_args):
        raise RuntimeError("diagnostic storage unavailable")

    monkeypatch.setattr(evaluate_module, "evaluate_in_sandbox", fail_evaluator)
    monkeypatch.setattr(evaluate_module, "_record_attempt_diagnostic", fail_diagnostic)

    result = await evaluate_module.evaluate(request)

    assert result == EvaluationResult(status="infra_failed", outcome=None, score=None)
    assert "publish result" not in events


def test_evaluator_environment_is_a_strict_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluate_module = import_module("ale_run.atomic.evaluate")
    monkeypatch.setenv("OPENAI_API_KEY", "judge-key")
    monkeypatch.setenv("LLM_JUDGE_MODEL", "judge-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "agent-key")
    monkeypatch.setenv("ALE_REFERENCE_ARCHIVE_PASSWORD", "reference-key")
    monkeypatch.setenv("UNRELATED_SECRET", "do-not-forward")

    env = evaluate_module._evaluator_environment()

    assert env["OPENAI_API_KEY"] == "judge-key"
    assert env["LLM_JUDGE_MODEL"] == "judge-model"
    assert "ANTHROPIC_API_KEY" not in env
    assert "ALE_REFERENCE_ARCHIVE_PASSWORD" not in env
    assert "UNRELATED_SECRET" not in env


def test_empty_host_ossutil_output_has_a_stable_diagnostic() -> None:
    evaluate_module = import_module("ale_run.atomic.evaluate")

    assert evaluate_module._host_command_diagnostic((7, b"", b"")) == "ossutil exited 7"


def test_host_ossutil_status_code_404_is_treated_as_a_missing_object() -> None:
    evaluate_module = import_module("ale_run.atomic.evaluate")

    assert evaluate_module._is_missing_object("ServerError: StatusCode=404, RequestId=example")


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata", [{}, {"image_id": "m-other"}])
async def test_open_fails_closed_for_missing_or_mismatched_image_identity(
    tmp_path: Path,
    metadata: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _make_evaluate_request(tmp_path)
    monkeypatch.setattr(
        "ale_run.atomic.runtime.load_experiment",
        lambda _path: SimpleNamespace(environment=SimpleNamespace(), artifacts=None),
    )
    provider = _FakeProvider(metadata=metadata)

    with pytest.raises(AtomicInfrastructureError, match="image_id mismatch") as caught:
        async with AtomicRuntime.open(
            request=request,
            run_setup=False,
            provider=provider,
        ):
            pass

    assert caught.value.category == "image_identity"
    assert provider.release_calls == ["delete"]


@pytest.mark.asyncio
async def test_evaluate_runtime_validates_evaluator_checkout_not_solve_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _make_evaluate_request(tmp_path)
    task_main = request.task_repo / "tasks" / request.task_path / "main.py"
    task_main.write_text(
        task_main.read_text(encoding="utf-8") + "\nEVALUATOR_VERSION = 2\n",
        encoding="utf-8",
    )
    await asyncio.to_thread(
        subprocess.run,
        ["git", "-C", str(request.task_repo), "add", "tasks/toy/main.py"],
        check=True,
    )
    await asyncio.to_thread(
        subprocess.run,
        [
            "git",
            "-C",
            str(request.task_repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "evaluator",
        ],
        check=True,
    )
    evaluator_version = (
        await asyncio.to_thread(
            subprocess.check_output,
            ["git", "-C", str(request.task_repo), "rev-parse", "HEAD"],
            text=True,
        )
    ).strip()
    request = request.model_copy(update={"evaluator_version": evaluator_version})
    monkeypatch.setattr(
        "ale_run.atomic.runtime.load_experiment",
        lambda _path: SimpleNamespace(environment=SimpleNamespace(), artifacts=None),
    )
    provider = _FakeProvider(metadata={"image_id": request.image_id})

    async with AtomicRuntime.open(
        request=request,
        run_setup=False,
        provider=provider,
    ) as runtime:
        assert runtime.task_driver is None

    assert request.task_commit != request.evaluator_version
    assert provider.release_calls == ["delete"]


@pytest.mark.asyncio
async def test_evaluate_open_skips_setup_and_never_resolves_an_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _make_evaluate_request(tmp_path)
    task_main = request.task_repo / "tasks" / request.task_path / "main.py"
    task_main.write_text(
        task_main.read_text(encoding="utf-8")
        + "\ndef setup(_config, _session):\n"
        + "    (__import__('pathlib').Path(__file__).with_name('setup-ran')).write_text('yes')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "ale_run.atomic.runtime.load_experiment",
        lambda _path: SimpleNamespace(environment=SimpleNamespace(), artifacts=None),
    )
    monkeypatch.setattr(
        "ale_run.orchestration.factory.resolve_agent",
        lambda _spec: (_ for _ in ()).throw(AssertionError("agent resolved")),
    )
    provider = _FakeProvider(metadata={"image_id": "m-expected"})

    async with AtomicRuntime.open(request=request, run_setup=False, provider=provider) as runtime:
        assert runtime.task_driver is None

    assert not (request.task_repo / "tasks" / request.task_path / "setup-ran").exists()
    assert provider.release_calls == ["delete"]
