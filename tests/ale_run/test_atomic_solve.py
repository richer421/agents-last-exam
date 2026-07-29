from __future__ import annotations

import subprocess
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from ale_run.atomic.contracts import (
    ArtifactEntry,
    AtomicInfrastructureError,
    SolveRequest,
    SubmissionManifest,
)
from ale_run.atomic.runtime import AtomicRuntime
from ale_run.base_interface import AgentRunResult, TaskDataSpec
from ale_run.orchestration import lifecycle
from ale_run.orchestration.experiment_spec import AgentSpec, EnvironmentSpec


class _FakeProvider:
    def __init__(self, *, metadata: dict[str, str]) -> None:
        self.sandbox = SimpleNamespace(id="sandbox-1", metadata=metadata, os="linux")
        self.acquire_calls: list[object] = []
        self.release_calls: list[str] = []

    async def acquire(self, spec: object) -> SimpleNamespace:
        self.acquire_calls.append(spec)
        return self.sandbox

    def open_session(self, _sandbox: object) -> SimpleNamespace:
        return SimpleNamespace(computer=SimpleNamespace(interface=SimpleNamespace()))

    async def release(self, _sandbox: object, *, mode: str) -> None:
        self.release_calls.append(mode)


@pytest.fixture
def solve_request(tmp_path: Path) -> SolveRequest:
    return _make_solve_request(tmp_path)


def _make_solve_request(tmp_path: Path) -> SolveRequest:
    repo = tmp_path / "task-repo"
    task_dir = repo / "tasks" / "toy"
    task_dir.mkdir(parents=True)
    (task_dir / "main.py").write_text(
        "class Config:\n"
        "    task_description = 'Solve the toy task.'\n"
        "    metadata = {}\n"
        "config = Config()\n",
        encoding="utf-8",
    )
    (task_dir / "task_card.json").write_text('{"vm":{"snapshot":"test-image"}}\n', encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "tasks/toy"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "task",
        ],
        check=True,
    )
    task_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    return SolveRequest(
        submission_id=uuid4(),
        runtime_spec_path=tmp_path / "runtime.yaml",
        task_repo=repo,
        task_path="toy",
        variant_index=0,
        agent_id="agent-under-test",
        task_commit=task_commit,
        image_id="m-expected",
        submission_root="oss://submissions",
    )


def _commit_runtime_spec(request: SolveRequest) -> SolveRequest:
    runtime_spec_path = request.task_repo / "runtime.yaml"
    runtime_spec_path.write_text("agents: []\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(request.task_repo), "add", "runtime.yaml"],
        check=True,
    )
    subprocess.run(
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
            "runtime spec",
        ],
        check=True,
    )
    task_commit = subprocess.check_output(
        ["git", "-C", str(request.task_repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    return request.model_copy(
        update={
            "runtime_spec_path": runtime_spec_path,
            "task_commit": task_commit,
        }
    )


@pytest.mark.asyncio
async def test_open_acquires_one_vm_and_releases_it_when_the_body_raises(
    solve_request: SolveRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_spec = SimpleNamespace(environment=SimpleNamespace(), artifacts=None)
    monkeypatch.setattr("ale_run.atomic.runtime.load_experiment", lambda _path: runtime_spec)
    provider = _FakeProvider(metadata={"image_id": "m-expected"})

    with pytest.raises(RuntimeError, match="body failed"):
        async with AtomicRuntime.open(request=solve_request, provider=provider) as runtime:
            assert runtime.provider is provider
            assert runtime.task_dir == solve_request.task_repo / "tasks" / "toy"
            assert runtime.runtime_spec is runtime_spec
            raise RuntimeError("body failed")

    assert len(provider.acquire_calls) == 1
    assert provider.release_calls == ["delete"]


@dataclass
class _TestAgentConfig:
    model: str = "test-model"
    name: str = "test-agent"


class _TestDeployer:
    default_executor = "local"
    supported_executors = frozenset({"local"})


def _submission_manifest(request: SolveRequest) -> SubmissionManifest:
    return SubmissionManifest(
        submission_id=request.submission_id,
        task_path=request.task_path,
        variant_index=request.variant_index,
        task_commit=request.task_commit,
        image_id=request.image_id,
        ale_run_id="solve-run",
        agent_id=request.agent_id,
        model_id="test-model",
        config_digest="a" * 64,
        started_at=datetime(2026, 7, 29, tzinfo=UTC),
        completed_at=datetime(2026, 7, 29, 0, 1, tzinfo=UTC),
        artifacts=(
            ArtifactEntry(
                path="answer.txt",
                size_bytes=1,
                sha256="b" * 64,
                media_type="text/plain",
            ),
        ),
    )


def _runtime_for_solve(request: SolveRequest, agents: list[AgentSpec]) -> SimpleNamespace:
    return SimpleNamespace(
        runtime_spec=SimpleNamespace(agents=agents),
        env=SimpleNamespace(sandbox=SimpleNamespace(is_linux=True, work_dir_base="/ordinary/work")),
        task_meta={"description": "Solve the isolated task and write answer.txt."},
        task_data=TaskDataSpec(),
        task_driver=SimpleNamespace(
            evaluate=lambda: (_ for _ in ()).throw(AssertionError("evaluate called"))
        ),
    )


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    runtime: SimpleNamespace,
    events: list[str],
    solve_module: object,
) -> None:
    async def no_existing_manifest(_request):
        return None

    @asynccontextmanager
    async def open_runtime(*, request: SolveRequest):
        events.extend(["open runtime", "task setup"])
        try:
            yield runtime
        finally:
            events.append("cleanup")

    monkeypatch.setattr(solve_module.AtomicRuntime, "open", staticmethod(open_runtime))
    monkeypatch.setattr(
        solve_module,
        "load_experiment",
        lambda _path: runtime.runtime_spec,
    )
    monkeypatch.setattr(
        solve_module,
        "read_existing_submission_manifest",
        no_existing_manifest,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["tracked", "untracked"])
async def test_solve_rejects_repository_changes_before_runtime_provider_entry(
    solve_request: SolveRequest,
    change: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solve_module = import_module("ale_run.atomic.solve")
    selected = AgentSpec(id=solve_request.agent_id, class_="test", config={})

    if change == "tracked":
        changed_path = solve_request.task_repo / "tasks" / "toy" / "main.py"
        with changed_path.open("a", encoding="utf-8") as stream:
            stream.write("MUTABLE = True\n")
    else:
        changed_path = solve_request.task_repo / "tasks" / "toy" / "untracked.py"
        changed_path.write_text("MUTABLE = True\n", encoding="utf-8")

    class ProviderSentinel(_FakeProvider):
        async def acquire(self, spec: object) -> SimpleNamespace:
            raise AssertionError("provider acquired")

    provider = ProviderSentinel(metadata={"image_id": solve_request.image_id})

    class Router:
        def __init__(self, _environment):
            pass

        def provider_for(self, _snapshot):
            return provider

    async def no_existing_manifest(_request):
        return None

    monkeypatch.setattr(
        solve_module,
        "read_existing_submission_manifest",
        no_existing_manifest,
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
    monkeypatch.setattr(solve_module, "build_config", lambda *_args: _TestAgentConfig())
    monkeypatch.setattr(
        "ale_run.atomic.runtime.load_experiment",
        lambda _path: SimpleNamespace(environment=EnvironmentSpec(), artifacts=None),
    )
    monkeypatch.setattr("ale_run.atomic.runtime.EnvironmentRouter", Router)

    with pytest.raises(AtomicInfrastructureError) as caught:
        await solve_module.solve(solve_request)

    assert caught.value.category == "task_checkout"
    assert "tracked or untracked changes" in caught.value.message


@pytest.mark.asyncio
async def test_solve_runtime_uses_exact_committed_checkout_after_host_bytes_change(
    solve_request: SolveRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solve_request = _commit_runtime_spec(solve_request)
    solve_module = import_module("ale_run.atomic.solve")
    selected = AgentSpec(id=solve_request.agent_id, class_="test", config={})
    runtime = _runtime_for_solve(solve_request, [selected])
    host_main = solve_request.task_repo / "tasks" / "toy" / "main.py"
    observed_repositories: list[Path] = []
    loaded_runtime_specs: list[Path] = []

    async def no_existing_manifest(_request):
        return None

    @asynccontextmanager
    async def open_runtime(*, request: SolveRequest):
        host_main.write_text("MUTABLE = True\n", encoding="utf-8")
        observed_repositories.append(request.task_repo)
        assert request.task_repo != solve_request.task_repo
        assert not (request.task_repo / ".git").exists()
        assert (
            (request.task_repo / "tasks" / "toy" / "main.py")
            .read_text(encoding="utf-8")
            .startswith("class Config:")
        )
        yield runtime

    class Executor:
        async def run_deployer(self, **_kwargs):
            return AgentRunResult(status="failed", error="stop after checkout assertion")

    monkeypatch.setattr(
        solve_module,
        "read_existing_submission_manifest",
        no_existing_manifest,
    )
    monkeypatch.setattr(
        solve_module,
        "load_experiment",
        lambda path: loaded_runtime_specs.append(Path(path)) or SimpleNamespace(agents=[selected]),
    )
    monkeypatch.setattr(
        solve_module,
        "resolve_agent",
        lambda _spec: (_TestDeployer, _TestAgentConfig),
    )
    monkeypatch.setattr(solve_module, "build_config", lambda *_args: _TestAgentConfig())
    monkeypatch.setattr(solve_module.AtomicRuntime, "open", staticmethod(open_runtime))
    monkeypatch.setattr(solve_module, "_build_executor", lambda **_kwargs: Executor())

    result = await solve_module.solve(solve_request)

    assert result.status == "failed"
    assert len(observed_repositories) == 1
    assert loaded_runtime_specs == [observed_repositories[0] / "runtime.yaml"]


@pytest.mark.asyncio
async def test_solve_runs_one_selected_agent_then_publishes_without_exposing_submission_identity(
    solve_request: SolveRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solve_module = import_module("ale_run.atomic.solve")

    events: list[str] = []
    selected = AgentSpec(id=solve_request.agent_id, class_="test", config={})
    runtime = _runtime_for_solve(solve_request, [selected, AgentSpec(id="other", class_="test")])
    _patch_runtime(monkeypatch, runtime, events, solve_module)
    config = _TestAgentConfig()
    monkeypatch.setattr(_TestDeployer, "default_executor", "sandbox")
    monkeypatch.setattr(solve_module, "uuid4", lambda: SimpleNamespace(hex="opaque-run-id"))
    monkeypatch.setenv("OPENAI_API_KEY", "allowed-key")
    monkeypatch.setenv("SUBMISSION_ID", str(solve_request.submission_id))
    monkeypatch.setenv("EVALUATOR_IDENTITY", "strict-evaluator")

    class Executor:
        def __init__(self, *, config, work_dir, sandbox, env):
            assert config is solve_config
            assert work_dir == "/ordinary/work/test-agent/opaque-run-id"
            assert sandbox is runtime.env.sandbox
            assert env["OPENAI_API_KEY"] == "allowed-key"
            assert str(solve_request.submission_id) not in str(env)
            assert "strict-evaluator" not in str(env)
            self.work_dir = work_dir

        async def run_deployer(self, *, deployer_cls, prompt, timeout_s):
            events.append("run deployer")
            assert deployer_cls is _TestDeployer
            assert prompt == runtime.task_meta["description"]
            assert str(solve_request.submission_id) not in prompt
            assert "strict-evaluator" not in prompt
            assert timeout_s == 18_000.0
            return AgentRunResult(status="completed")

    def resolve_agent(spec):
        assert spec is selected
        return _TestDeployer, _TestAgentConfig

    def build_config(config_cls, raw):
        assert config_cls is _TestAgentConfig
        assert raw is selected.config
        return solve_config

    async def publish(sandbox, task_data, request, *, provenance):
        events.append("publish submission")
        assert sandbox is runtime.env.sandbox
        assert task_data is runtime.task_data
        assert request.task_repo != solve_request.task_repo
        assert not (request.task_repo / ".git").exists()
        assert request.model_copy(update={"task_repo": solve_request.task_repo}) == solve_request
        assert set(provenance) == {
            "ale_run_id",
            "model_id",
            "config_digest",
            "started_at",
            "completed_at",
        }
        return _submission_manifest(request)

    solve_config = config
    monkeypatch.setattr(solve_module, "resolve_agent", resolve_agent)
    monkeypatch.setattr(solve_module, "build_config", build_config)
    monkeypatch.setattr(lifecycle, "SandboxExecutor", Executor)
    monkeypatch.setattr(solve_module, "publish_submission", publish)

    result = await solve_module.solve(solve_request)

    assert events == ["open runtime", "task setup", "run deployer", "publish submission", "cleanup"]
    assert result.status == "submitted"
    assert result.submission_id == solve_request.submission_id
    assert result.manifest == _submission_manifest(solve_request)


@pytest.mark.asyncio
async def test_solve_fails_closed_when_publisher_returns_a_different_submission_manifest(
    solve_request: SolveRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solve_module = import_module("ale_run.atomic.solve")

    events: list[str] = []
    runtime = _runtime_for_solve(
        solve_request,
        [AgentSpec(id=solve_request.agent_id, class_="test", config={})],
    )
    _patch_runtime(monkeypatch, runtime, events, solve_module)

    class Executor:
        async def run_deployer(self, **_kwargs):
            events.append("run deployer")
            return AgentRunResult(status="completed")

    async def publish(*_args, **_kwargs):
        events.append("publish submission")
        return _submission_manifest(solve_request).model_copy(update={"submission_id": uuid4()})

    monkeypatch.setattr(
        solve_module, "resolve_agent", lambda _spec: (_TestDeployer, _TestAgentConfig)
    )
    monkeypatch.setattr(solve_module, "build_config", lambda *_args: _TestAgentConfig())
    monkeypatch.setattr(solve_module, "_build_executor", lambda **_kwargs: Executor())
    monkeypatch.setattr(solve_module, "publish_submission", publish)

    with pytest.raises(ValidationError, match="submission_id"):
        await solve_module.solve(solve_request)

    assert events == ["open runtime", "task setup", "run deployer", "publish submission", "cleanup"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "timeout"])
async def test_solve_does_not_publish_unsuccessful_agent_runs(
    solve_request: SolveRequest,
    status: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solve_module = import_module("ale_run.atomic.solve")

    events: list[str] = []
    runtime = _runtime_for_solve(
        solve_request,
        [AgentSpec(id=solve_request.agent_id, class_="test", config={})],
    )
    _patch_runtime(monkeypatch, runtime, events, solve_module)

    class Executor:
        async def run_deployer(self, **_kwargs):
            events.append("run deployer")
            return AgentRunResult(status=status, error=f"{status} run")

    monkeypatch.setattr(
        solve_module, "resolve_agent", lambda spec: (_TestDeployer, _TestAgentConfig)
    )
    monkeypatch.setattr(solve_module, "build_config", lambda *_args: _TestAgentConfig())
    monkeypatch.setattr(solve_module, "_build_executor", lambda **_kwargs: Executor())
    monkeypatch.setattr(
        solve_module,
        "publish_submission",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("publish called")),
    )

    result = await solve_module.solve(solve_request)

    assert events == ["open runtime", "task setup", "run deployer", "cleanup"]
    assert result.status == "failed"
    assert result.submission_id == solve_request.submission_id
    assert result.manifest is None
    assert result.error == f"{status} run"


@pytest.mark.asyncio
async def test_solve_propagates_submission_publication_failures_after_cleanup(
    solve_request: SolveRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solve_module = import_module("ale_run.atomic.solve")

    events: list[str] = []
    runtime = _runtime_for_solve(
        solve_request,
        [AgentSpec(id=solve_request.agent_id, class_="test", config={})],
    )
    _patch_runtime(monkeypatch, runtime, events, solve_module)

    class Executor:
        async def run_deployer(self, **_kwargs):
            events.append("run deployer")
            return AgentRunResult(status="completed")

    async def publish(*_args, **_kwargs):
        events.append("publish submission")
        raise RuntimeError("immutable publish failed")

    monkeypatch.setattr(
        solve_module, "resolve_agent", lambda spec: (_TestDeployer, _TestAgentConfig)
    )
    monkeypatch.setattr(solve_module, "build_config", lambda *_args: _TestAgentConfig())
    monkeypatch.setattr(solve_module, "_build_executor", lambda **_kwargs: Executor())
    monkeypatch.setattr(solve_module, "publish_submission", publish)

    with pytest.raises(RuntimeError, match="immutable publish failed"):
        await solve_module.solve(solve_request)

    assert events == ["open runtime", "task setup", "run deployer", "publish submission", "cleanup"]


@pytest.mark.asyncio
async def test_solve_requires_exactly_one_matching_agent(
    solve_request: SolveRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solve_module = import_module("ale_run.atomic.solve")

    events: list[str] = []
    runtime = _runtime_for_solve(
        solve_request,
        [
            AgentSpec(id=solve_request.agent_id, class_="test"),
            AgentSpec(id=solve_request.agent_id, class_="other-test"),
        ],
    )
    _patch_runtime(monkeypatch, runtime, events, solve_module)

    with pytest.raises(ValueError, match="exactly one"):
        await solve_module.solve(solve_request)

    assert events == []
