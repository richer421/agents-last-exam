import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pytest

from ale_run.atomic.author_agent import AuthoredEvaluator, AuthorInputBundle
from ale_run.atomic.author_evaluator import (
    AuthorEvaluatorDependencies,
    _default_input_loader,
    author_evaluator,
)
from ale_run.atomic.author_registry import FilesystemAuthorRegistry
from ale_run.atomic.contracts import (
    AtomicInfrastructureError,
    AuthorEvaluatorRegistryIdentity,
    AuthorEvaluatorRequest,
    EvaluatorRegistryRecord,
    RubricPlan,
    RubricPlanEntry,
)
from ale_run.atomic.github_publish import GitHubPublication


def _request() -> AuthorEvaluatorRequest:
    return AuthorEvaluatorRequest(
        authoring_id=UUID("00000000-0000-0000-0000-000000000001"),
        task_repository_url="https://github.com/openai/ale-tasks",
        task_path="demo/example",
        variant_index=0,
        task_commit="a" * 40,
        image_id="m-authoring-image",
        rubric_uri="oss://ale-rubrics/example/rubrics.json",
        rubric_hash="b" * 64,
        reference_manifest_uri="oss://ale-reference/example/manifest.json",
        reference_manifest_hash="c" * 64,
        evaluator_id="rubric",
        evaluator_sdk_version="1.2.3",
        timeout_seconds=10,
    )


def _identity() -> AuthorEvaluatorRegistryIdentity:
    request = _request()
    return AuthorEvaluatorRegistryIdentity(
        task_commit=request.task_commit,
        rubric_hash=request.rubric_hash,
        reference_manifest_hash=request.reference_manifest_hash,
        evaluator_sdk_version=request.evaluator_sdk_version,
        image_id=request.image_id,
    )


def _record() -> EvaluatorRegistryRecord:
    request = _request()
    return EvaluatorRegistryRecord(
        status="ready",
        task_path=request.task_path,
        variant_index=request.variant_index,
        task_commit=request.task_commit,
        evaluator_id=request.evaluator_id,
        evaluator_version="d" * 40,
        rubric_hash=request.rubric_hash,
        reference_manifest_uri=request.reference_manifest_uri,
        reference_manifest_hash=request.reference_manifest_hash,
        evaluator_sdk_version=request.evaluator_sdk_version,
        harbor_version="2.0.0",
        rewardkit_version="1.0.0",
        image_id=request.image_id,
        pull_request_url="https://github.com/openai/ale-tasks/pull/123",
        ci_run_id="987654",
        ready_at=datetime(2026, 8, 2, tzinfo=UTC),
    )


def _plan() -> RubricPlan:
    return RubricPlan(
        rubric_hash="b" * 64,
        items=(
            RubricPlanEntry(
                rubric_id="quality",
                implementation_mode="llm_judge",
                weight=1,
                score_min=0,
                score_max=1,
                required=True,
                rationale="Semantic quality.",
            ),
        ),
    )


@pytest.mark.asyncio
async def test_default_input_loader_accepts_large_photoshop_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = b"x" * (50 * 1024 * 1024 + 1)
    manifest = json.dumps(
        {
            "schema_version": 1,
            "files": [
                {
                    "path": "final_result.psd",
                    "size_bytes": len(artifact),
                    "sha256": hashlib.sha256(artifact).hexdigest(),
                }
            ],
        },
        separators=(",", ":"),
    ).encode()
    request = _request().model_copy(
        update={"reference_manifest_hash": hashlib.sha256(manifest).hexdigest()}
    )
    task_directory = tmp_path / "tasks" / "demo" / "example"
    task_directory.mkdir(parents=True)
    (task_directory / "task_card.json").write_text("{}\n", encoding="utf-8")
    workspace = SimpleNamespace(task_directory=task_directory)

    async def read_object(uri, *, limit, **_kwargs):
        if uri == request.rubric_uri:
            return b'{"rubrics":[]}\n'
        if uri == request.reference_manifest_uri:
            return manifest
        assert uri.endswith("/final_result.psd")
        if limit < len(artifact):
            raise AtomicInfrastructureError("author_input", "object exceeds limit")
        return artifact

    monkeypatch.setattr(
        "ale_run.atomic.author_evaluator.read_host_oss_object",
        read_object,
    )

    bundle = await _default_input_loader(request, workspace)

    assert len(bundle.reference_artifacts["final_result.psd"]) == len(artifact)
    sdk_contract = json.loads(bundle.evaluator_sdk_contract)
    assert sdk_contract["rubric_plan_schema"] == RubricPlan.model_json_schema()
    assert sdk_contract["rubric_plan_schema"]["additionalProperties"] is False


class FakeRegistry:
    def __init__(self, existing=None, publish_error: Exception | None = None) -> None:
        self.existing = existing
        self.publish_error = publish_error
        self.get_calls = []
        self.publish_calls = []
        self._claims = {}

    @asynccontextmanager
    async def claim(self, identity):
        lock = self._claims.setdefault(identity.model_dump_json(), asyncio.Lock())
        async with lock:
            yield FakeRegistryClaim(self, identity)

    def get_ready(self, identity):
        self.get_calls.append(identity)
        return self.existing

    def publish_ready(self, identity, record):
        self.publish_calls.append((identity, record))
        if self.publish_error:
            raise self.publish_error
        self.existing = self.existing or record
        return self.existing


class FakeRegistryClaim:
    def __init__(self, registry, identity) -> None:
        self.registry = registry
        self.identity = identity

    def get_ready(self):
        return self.registry.get_ready(self.identity)

    def publish_ready(self, record):
        return self.registry.publish_ready(self.identity, record)


class FakeAgent:
    def __init__(self, error: Exception | None = None, delay: float = 0) -> None:
        self.error = error
        self.delay = delay
        self.calls = []

    async def author(self, request, workspace, inputs):
        self.calls.append((request, workspace, inputs))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise self.error
        return AuthoredEvaluator(
            rubric_plan=_plan(),
            files=("rubric-plan.json", "test.sh"),
        )


class FakePublisher:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = []

    async def publish(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return GitHubPublication(
            pull_request_url="https://github.com/openai/ale-tasks/pull/123",
            ci_run_id="987654",
            evaluator_version="d" * 40,
        )


def _dependencies(
    *,
    registry=None,
    agent=None,
    publisher=None,
    test_error: Exception | None = None,
    calls: list[str] | None = None,
) -> AuthorEvaluatorDependencies:
    call_log = calls if calls is not None else []
    workspace = SimpleNamespace(root=Path("/tmp/workspace"), task_directory=Path("/tmp/task"))

    @asynccontextmanager
    async def workspace_factory(_request):
        call_log.append("workspace_enter")
        try:
            yield workspace
        finally:
            call_log.append("workspace_exit")

    async def input_loader(_request, loaded_workspace):
        assert loaded_workspace is workspace
        call_log.append("inputs")
        return AuthorInputBundle({}, b"rubric", b"manifest", {}, b"sdk", b"image")

    async def test_runner(loaded_workspace, timeout_seconds):
        assert loaded_workspace is workspace
        assert timeout_seconds > 0
        call_log.append("tests")
        if test_error:
            raise test_error

    return AuthorEvaluatorDependencies(
        registry=registry or FakeRegistry(),
        workspace_factory=workspace_factory,
        input_loader=input_loader,
        author_agent=agent or FakeAgent(),
        publisher=publisher or FakePublisher(),
        local_test_runner=test_runner,
        harbor_version="2.0.0",
        rewardkit_version="1.0.0",
        now=lambda: datetime(2026, 8, 2, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_author_evaluator_composes_author_tests_publish_and_registry() -> None:
    calls: list[str] = []
    registry = FakeRegistry()
    agent = FakeAgent()
    publisher = FakePublisher()
    dependencies = _dependencies(
        registry=registry,
        agent=agent,
        publisher=publisher,
        calls=calls,
    )

    result = await author_evaluator(_request(), dependencies=dependencies)

    assert result.status == "ready"
    assert result.evaluator_version == "d" * 40
    assert result.pull_request_url.endswith("/pull/123")
    assert result.ci_run_id == "987654"
    assert calls == ["workspace_enter", "inputs", "tests", "workspace_exit"]
    assert len(agent.calls) == 1
    assert len(publisher.calls) == 1
    assert registry.get_calls == [_identity()]
    assert registry.publish_calls[0][0] == _identity()
    assert registry.publish_calls[0][1] == _record()


@pytest.mark.asyncio
async def test_author_evaluator_reuses_ready_registry_without_other_dependencies() -> None:
    registry = FakeRegistry(existing=_record())
    calls: list[str] = []
    agent = FakeAgent()
    publisher = FakePublisher()

    result = await author_evaluator(
        _request(),
        dependencies=_dependencies(
            registry=registry,
            agent=agent,
            publisher=publisher,
            calls=calls,
        ),
    )

    assert result.status == "ready"
    assert result.evaluator_version == "d" * 40
    assert calls == []
    assert agent.calls == []
    assert publisher.calls == []
    assert registry.publish_calls == []


@pytest.mark.asyncio
async def test_author_evaluator_serializes_same_identity_authoring() -> None:
    registry = FakeRegistry()
    agent = FakeAgent(delay=0.01)
    publisher = FakePublisher()
    dependencies = _dependencies(registry=registry, agent=agent, publisher=publisher)
    second_request = _request().model_copy(
        update={"authoring_id": UUID("00000000-0000-0000-0000-000000000002")}
    )

    first, second = await asyncio.gather(
        author_evaluator(_request(), dependencies=dependencies),
        author_evaluator(second_request, dependencies=dependencies),
    )

    assert len(agent.calls) == 1
    assert len(publisher.calls) == 1
    assert [first.authoring_id, second.authoring_id] == [
        _request().authoring_id,
        second_request.authoring_id,
    ]
    assert first.status == second.status == "ready"


@pytest.mark.asyncio
async def test_author_evaluator_serializes_with_filesystem_registry(
    tmp_path: Path,
) -> None:
    registry = FilesystemAuthorRegistry(tmp_path / "author-registry")
    agent = FakeAgent(delay=0.01)
    publisher = FakePublisher()
    dependencies = _dependencies(registry=registry, agent=agent, publisher=publisher)
    second_request = _request().model_copy(
        update={"authoring_id": UUID("00000000-0000-0000-0000-000000000002")}
    )

    first, second = await asyncio.gather(
        author_evaluator(_request(), dependencies=dependencies),
        author_evaluator(second_request, dependencies=dependencies),
    )

    assert len(agent.calls) == 1
    assert len(publisher.calls) == 1
    assert [first.authoring_id, second.authoring_id] == [
        _request().authoring_id,
        second_request.authoring_id,
    ]
    assert first.status == second.status == "ready"


@pytest.mark.parametrize("failure_stage", ["author", "tests", "publish", "registry"])
@pytest.mark.asyncio
async def test_author_evaluator_returns_structured_failure_without_ready_registration(
    failure_stage: str,
) -> None:
    failure = AtomicInfrastructureError(failure_stage, "operation failed")
    registry = FakeRegistry(publish_error=failure if failure_stage == "registry" else None)
    agent = FakeAgent(error=failure if failure_stage == "author" else None)
    publisher = FakePublisher(error=failure if failure_stage == "publish" else None)
    dependencies = _dependencies(
        registry=registry,
        agent=agent,
        publisher=publisher,
        test_error=failure if failure_stage == "tests" else None,
    )

    result = await author_evaluator(_request(), dependencies=dependencies)

    assert result.status == "authoring_failed"
    assert result.error_category == failure_stage
    assert result.error_detail == "operation failed"
    assert result.evaluator_version is None
    if failure_stage != "registry":
        assert registry.publish_calls == []


@pytest.mark.asyncio
async def test_author_evaluator_enforces_whole_capability_timeout() -> None:
    request = _request().model_copy(update={"timeout_seconds": 1})

    result = await author_evaluator(
        request,
        dependencies=_dependencies(agent=FakeAgent(delay=60)),
    )

    assert result.status == "authoring_failed"
    assert result.error_category == "timeout"


@pytest.mark.asyncio
async def test_author_evaluator_redacts_secret_values_from_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUTHOR_API_KEY", "very-secret-value")
    failure = AtomicInfrastructureError(
        "author_agent",
        "command rejected very-secret-value",
    )

    result = await author_evaluator(
        _request(),
        dependencies=_dependencies(agent=FakeAgent(error=failure)),
    )

    assert result.status == "authoring_failed"
    assert result.error_detail == "command rejected [redacted]"
    assert "very-secret-value" not in result.model_dump_json()


@pytest.mark.asyncio
async def test_author_evaluator_rejects_mismatched_ready_record() -> None:
    registry = FakeRegistry(existing=_record().model_copy(update={"evaluator_id": "other"}))

    result = await author_evaluator(
        _request(),
        dependencies=_dependencies(registry=registry),
    )

    assert result.status == "authoring_failed"
    assert result.error_category == "author_registry"
