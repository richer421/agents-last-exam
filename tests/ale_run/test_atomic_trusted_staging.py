from __future__ import annotations

import asyncio
import shutil
import subprocess
import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from ale_run.atomic.contracts import (
    AtomicInfrastructureError,
    EvaluateRequest,
    EvaluatorRegistryRecord,
)
from ale_run.atomic.runtime import AtomicRuntime
from ale_run.atomic.trusted_staging import (
    prepare_atomic_input,
    sanitize_solve_environment,
    stage_atomic_input,
)
from ale_run.base_interface import TaskDataSpec
from ale_run.orchestration.experiment_spec import (
    EnvironmentSpec,
    ProviderSpec,
)
from tests.ale_run.test_atomic_solve import _FakeProvider, _make_solve_request


class _LocalSandbox:
    def __init__(self, root: Path) -> None:
        self.is_linux = True
        self.python = sys.executable
        self.task_data_root = str(root / "task-data")
        self.commands: list[str] = []

    async def mkdir(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

    async def upload_local_file(self, local_path: str, remote_path: str) -> None:
        Path(remote_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, remote_path)

    async def write_file(self, path: str, content: str | bytes) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            destination.write_bytes(content)
        else:
            destination.write_text(content, encoding="utf-8")

    async def run_command(
        self,
        command: str,
        *,
        timeout: float = 60,
    ) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        return await asyncio.to_thread(
            subprocess.run,
            command,
            capture_output=True,
            check=False,
            shell=True,
            text=True,
            timeout=timeout,
        )


def _task_data() -> TaskDataSpec:
    return TaskDataSpec(
        requires_task_data=True,
        domain_name="demo",
        task_name="toy",
        variant_name="base",
    )


def test_solve_environment_removes_storage_identity_without_mutating_legacy_config() -> None:
    environment = EnvironmentSpec(
        provider_specs={
            "aliyun": ProviderSpec(
                kind="aliyun",
                config={
                    "region": "cn-hangzhou",
                    "ram_role_name": "oss-capable-role",
                    "output_to_bucket": True,
                },
            )
        },
        snapshot_kind={"test-image": "aliyun"},
    )

    sanitized = sanitize_solve_environment(environment)

    assert sanitized is not environment
    assert sanitized.provider_specs["aliyun"].config["ram_role_name"] == ""
    assert sanitized.provider_specs["aliyun"].config["output_to_bucket"] is False
    assert environment.provider_specs["aliyun"].config["ram_role_name"] == "oss-capable-role"
    assert environment.provider_specs["aliyun"].config["output_to_bucket"] is True


@pytest.mark.asyncio
async def test_atomic_runtime_removes_ram_role_before_provider_acquisition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _make_solve_request(tmp_path)
    environment = EnvironmentSpec(
        provider_specs={
            "aliyun": ProviderSpec(
                kind="aliyun",
                config={
                    "region": "cn-hangzhou",
                    "ram_role_name": "oss-capable-role",
                    "output_to_bucket": True,
                },
            )
        },
        snapshot_kind={"test-image": "aliyun"},
    )
    runtime_spec = type(
        "RuntimeSpec",
        (),
        {"environment": environment, "artifacts": None},
    )()
    provider = _FakeProvider(metadata={"image_id": request.image_id})
    observed: list[dict[str, object]] = []

    class Router:
        def __init__(self, actual_environment):
            observed.append(dict(actual_environment.provider_specs["aliyun"].config))

        def provider_for(self, _snapshot):
            return provider

    monkeypatch.setattr(
        "ale_run.atomic.runtime.load_experiment",
        lambda _path: runtime_spec,
    )
    monkeypatch.setattr("ale_run.atomic.runtime.EnvironmentRouter", Router)

    async with AtomicRuntime.open(request=request):
        pass

    assert observed == [
        {
            "region": "cn-hangzhou",
            "ram_role_name": "",
            "output_to_bucket": False,
        }
    ]
    assert environment.provider_specs["aliyun"].config["ram_role_name"] == "oss-capable-role"


@pytest.mark.asyncio
async def test_atomic_runtime_prepares_declared_input_before_provider_acquisition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _make_solve_request(tmp_path)
    task_card = request.task_repo / "tasks" / request.task_path / "task_card.json"
    task_card.write_text(
        '{"vm":{"snapshot":"test-image"},"inputFiles":[{"path":"input/required.txt"}]}\n',
        encoding="utf-8",
    )
    task_data = _task_data()
    runtime_spec = SimpleNamespace(
        environment=SimpleNamespace(),
        artifacts=SimpleNamespace(task_data_source="oss://private-bucket/tasks"),
    )
    events: list[object] = []

    class Provider(_FakeProvider):
        async def acquire(self, spec: object) -> SimpleNamespace:
            events.append("acquire")
            return await super().acquire(spec)

        async def release(self, sandbox: object, *, mode: str) -> None:
            events.append("release")
            await super().release(sandbox, mode=mode)

    provider = Provider(metadata={"image_id": request.image_id})
    prepared = object()

    @asynccontextmanager
    async def prepare_input(**kwargs):
        events.append(
            (
                "prepare",
                kwargs["source"],
                kwargs["declared_input_paths"],
            )
        )
        yield prepared
        events.append("prepared cleanup")

    async def stage_input(_sandbox, actual_task_data, **kwargs):
        assert actual_task_data is task_data
        assert kwargs == {
            "source": "oss://private-bucket/tasks",
            "declared_input_paths": ("input/required.txt",),
            "prepared": prepared,
        }
        events.append("stage")

    monkeypatch.setattr(
        "ale_run.atomic.runtime.load_experiment",
        lambda _path: runtime_spec,
    )
    monkeypatch.setattr(
        "ale_run.atomic.runtime.TaskLoader",
        lambda _path: SimpleNamespace(
            load=lambda _variant: {
                "image_category": "test-image",
                "task_data": task_data,
            }
        ),
    )
    monkeypatch.setattr(
        "ale_run.atomic.runtime.prepare_atomic_input",
        prepare_input,
        raising=False,
    )
    monkeypatch.setattr(
        "ale_run.atomic.runtime.stage_atomic_input",
        stage_input,
        raising=False,
    )

    async with AtomicRuntime.open(
        request=request,
        provider=provider,
        run_setup=False,
    ):
        events.append("body")

    assert events == [
        ("prepare", "oss://private-bucket/tasks", ("input/required.txt",)),
        "acquire",
        "stage",
        "body",
        "release",
        "prepared cleanup",
    ]


@pytest.mark.asyncio
async def test_evaluate_runtime_reference_prepare_failure_prevents_provider_acquisition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solve_request = _make_solve_request(tmp_path)
    task_card = solve_request.task_repo / "tasks" / solve_request.task_path / "task_card.json"
    task_card.write_text(
        '{"vm":{"snapshot":"test-image"},"referenceFiles":[{"path":"reference/answer.json"}]}\n',
        encoding="utf-8",
    )
    request = EvaluateRequest(
        submission_id=uuid4(),
        runtime_spec_path=solve_request.runtime_spec_path,
        task_repo=solve_request.task_repo,
        task_path=solve_request.task_path,
        variant_index=solve_request.variant_index,
        task_commit=solve_request.task_commit,
        image_id=solve_request.image_id,
        submission_root=solve_request.submission_root,
        evaluator_id="rubric",
        evaluator_version="b" * 40,
        evaluator_registry_record_path=tmp_path / "registry.json",
        evaluator_registry_record_sha256="c" * 64,
    )
    record = EvaluatorRegistryRecord(
        status="ready",
        task_path=request.task_path,
        variant_index=request.variant_index,
        task_commit=request.task_commit,
        evaluator_id=request.evaluator_id,
        evaluator_version=request.evaluator_version,
        rubric_hash="d" * 64,
        reference_manifest_uri="oss://trusted/reference/manifest.json",
        reference_manifest_hash="e" * 64,
        evaluator_sdk_version="1.2.3",
        harbor_version="0.20.0",
        rewardkit_version="0.4.0",
        image_id=request.image_id,
        pull_request_url="https://github.com/example/tasks/pull/42",
        ci_run_id="123456",
        ready_at=datetime(2026, 7, 29, tzinfo=UTC),
    )
    task_data = _task_data()
    runtime_spec = SimpleNamespace(
        environment=SimpleNamespace(),
        artifacts=SimpleNamespace(task_data_source="oss://private-bucket/tasks"),
    )
    provider = _FakeProvider(metadata={"image_id": request.image_id})

    @asynccontextmanager
    async def prepare_input(**_kwargs):
        yield None

    @asynccontextmanager
    async def prepare_reference(**kwargs):
        assert kwargs["record"] is record
        assert kwargs["declared_reference_paths"] == ("reference/answer.json",)
        raise AtomicInfrastructureError(
            "submission_storage",
            "manifest stat unavailable",
        )
        yield  # pragma: no cover

    monkeypatch.setattr(
        "ale_run.atomic.runtime.load_experiment",
        lambda _path: runtime_spec,
    )
    monkeypatch.setattr(
        "ale_run.atomic.runtime.TaskLoader",
        lambda _path: SimpleNamespace(
            load=lambda _variant: {
                "image_category": "test-image",
                "task_data": task_data,
            }
        ),
    )
    monkeypatch.setattr(
        "ale_run.atomic.runtime.prepare_atomic_input",
        prepare_input,
    )
    monkeypatch.setattr(
        "ale_run.atomic.runtime.prepare_atomic_reference",
        prepare_reference,
        raising=False,
    )

    with pytest.raises(AtomicInfrastructureError) as caught:
        async with AtomicRuntime.open(
            request=request,
            provider=provider,
            run_setup=False,
            evaluator_registry_record=record,
        ):
            raise AssertionError("runtime body entered")

    assert caught.value.category == "reference"
    assert "manifest stat unavailable" in caught.value.message
    assert provider.acquire_calls == []


@pytest.mark.asyncio
async def test_oss_input_is_downloaded_by_host_and_vm_never_receives_oss_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host_calls: list[tuple[str, ...]] = []

    async def host_ossutil(*arguments: str):
        host_calls.append(arguments)
        if arguments[0] == "ls":
            return 0, b"Object Number is: 1\n", b""
        destination = Path(arguments[-1])
        destination.mkdir(parents=True, exist_ok=True)
        if arguments[-2].endswith("/input/"):
            (destination / "required.txt").write_text("trusted input", encoding="utf-8")
        else:
            (destination / "tool.sh").write_text("#!/bin/sh\n", encoding="utf-8")
        return 0, b"", b""

    monkeypatch.setattr(
        "ale_run.atomic.trusted_staging.run_host_ossutil",
        host_ossutil,
    )
    sandbox = _LocalSandbox(tmp_path / "vm")

    async with prepare_atomic_input(
        source="oss://private-bucket/tasks",
        task_data=_task_data(),
        declared_input_paths=("input/required.txt",),
    ) as prepared:
        await stage_atomic_input(
            sandbox,
            _task_data(),
            source="oss://private-bucket/tasks",
            declared_input_paths=("input/required.txt",),
            prepared=prepared,
        )

    base = Path(sandbox.task_data_root) / "demo" / "toy" / "base"
    assert (base / "input" / "required.txt").read_text(encoding="utf-8") == "trusted input"
    assert (base / "software" / "tool.sh").is_file()
    assert any(call[0] == "sync" for call in host_calls)
    vm_commands = "\n".join(sandbox.commands).lower()
    assert "ossutil" not in vm_commands
    assert "100.100.100.200" not in vm_commands
    assert "private-bucket" not in vm_commands


@pytest.mark.asyncio
async def test_atomic_input_fails_closed_for_unsupported_remote_backend() -> None:
    with pytest.raises(AtomicInfrastructureError) as caught:
        async with prepare_atomic_input(
            source="s3://bucket/tasks",
            task_data=_task_data(),
            declared_input_paths=("input/required.txt",),
        ):
            pass

    assert caught.value.category == "input"
    assert "unsupported" in caught.value.message


@pytest.mark.asyncio
async def test_baked_input_uses_existing_backend_without_host_oss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    sandbox = _LocalSandbox(tmp_path / "vm")
    required = Path(sandbox.task_data_root) / "demo" / "toy" / "base" / "input"
    required.mkdir(parents=True)
    (required / "required.txt").write_text("baked", encoding="utf-8")

    async def stage_input(_sandbox, _task_data, *, source):
        events.append(source)
        return {"staged": ["input"], "source": source}

    monkeypatch.setattr(
        "ale_run.atomic.trusted_staging.task_data_pkg.select",
        lambda _source: type("Backend", (), {"stage_input": staticmethod(stage_input)}),
    )

    async with prepare_atomic_input(
        source="baked_in_sandbox",
        task_data=_task_data(),
        declared_input_paths=("input/required.txt",),
    ) as prepared:
        assert prepared is None
        await stage_atomic_input(
            sandbox,
            _task_data(),
            source="baked_in_sandbox",
            declared_input_paths=("input/required.txt",),
            prepared=prepared,
        )

    assert events == ["baked_in_sandbox"]
