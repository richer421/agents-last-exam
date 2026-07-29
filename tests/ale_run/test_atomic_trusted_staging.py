from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
import stat
import subprocess
import sys
import zipfile
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
    _STAGE_SCRIPT,
    PreparedInput,
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
        self.rm_calls: list[tuple[str, ...]] = []

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

    async def rm(self, paths) -> None:
        requested = tuple(paths)
        self.rm_calls.append(requested)
        for raw in requested:
            path = Path(raw)
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.exists():
                shutil.rmtree(path)

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


class _ScriptFaultSandbox(_LocalSandbox):
    def __init__(self, root: Path, faults: set[tuple[str, str]]) -> None:
        super().__init__(root)
        self.faults = faults

    async def write_file(self, path: str, content: str | bytes) -> None:
        if path.endswith(".py"):
            script = content.decode() if isinstance(content, bytes) else content
            fault_setup = (
                f"\ninjected_faults = {self.faults!r}\n"
                "def maybe_fail(operation, name):\n"
                "    if (operation, name) in injected_faults:\n"
                "        raise OSError(f'injected {operation} failure for {name}')\n"
            )
            script = script.replace(
                "staging_root = None\n",
                f"staging_root = None\n{fault_setup}",
                1,
            )
            script = script.replace(
                "                    live.replace(backup_root / name)",
                "                    maybe_fail('move-old', name)\n"
                "                    live.replace(backup_root / name)",
            )
            script = script.replace(
                "                    staged.replace(base / name)",
                "                    maybe_fail('install-new', name)\n"
                "                    staged.replace(base / name)",
            )
            script = script.replace(
                "            transaction_succeeded = True",
                "            maybe_fail('commit', '*')\n            transaction_succeeded = True",
            )
            script = script.replace(
                "                        remove_path(base / name)",
                "                        maybe_fail('remove-new', name)\n"
                "                        remove_path(base / name)",
            )
            script = script.replace(
                "                        previous.replace(base / name)",
                "                        maybe_fail('restore-old', name)\n"
                "                        previous.replace(base / name)",
            )
            content = script.encode()
        await super().write_file(path, content)


class _OrchestrationFailingSandbox(_LocalSandbox):
    def __init__(
        self,
        root: Path,
        failure: str,
        *,
        cleanup_fails: bool = False,
    ) -> None:
        super().__init__(root)
        self.failure = failure
        self.cleanup_fails = cleanup_fails

    async def upload_local_file(self, local_path: str, remote_path: str) -> None:
        await super().upload_local_file(local_path, remote_path)
        if self.failure == "upload":
            raise RuntimeError("injected upload failure")

    async def write_file(self, path: str, content: str | bytes) -> None:
        await super().write_file(path, content)
        if self.failure == "script-write" and path.endswith(".py"):
            raise RuntimeError("injected script-write failure")
        if self.failure == "config-write" and path.endswith(".json"):
            raise RuntimeError("injected config-write failure")

    async def run_command(
        self,
        command: str,
        *,
        timeout: float = 60,
    ) -> subprocess.CompletedProcess[str]:
        if self.failure == "run":
            raise RuntimeError("injected run failure")
        return await super().run_command(command, timeout=timeout)

    async def rm(self, paths) -> None:
        requested = tuple(paths)
        self.rm_calls.append(requested)
        if self.cleanup_fails:
            raise OSError(f"injected cleanup failure for {requested[0]}")
        await super().rm(requested)


def _task_data() -> TaskDataSpec:
    return TaskDataSpec(
        requires_task_data=True,
        domain_name="demo",
        task_name="toy",
        variant_name="base",
    )


def _write_input_archive(
    root: Path,
    members: dict[str, bytes],
    *,
    corrupt_member: str | None = None,
    remove_type_metadata: bool = False,
) -> PreparedInput:
    archive = root / f"input-{uuid4().hex}.zip"
    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_STORED) as bundle:
        for name, content in members.items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = (stat.S_IFREG | 0o600) << 16
            bundle.writestr(info, content)
    if corrupt_member is not None:
        with zipfile.ZipFile(archive) as bundle:
            info = bundle.getinfo(corrupt_member)
            data_offset = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)
        contents = bytearray(archive.read_bytes())
        contents[data_offset] ^= 0xFF
        archive.write_bytes(contents)
    if remove_type_metadata:
        contents = bytearray(archive.read_bytes())
        central_header = contents.index(b"PK\x01\x02")
        contents[central_header + 38 : central_header + 42] = b"\0\0\0\0"
        archive.write_bytes(contents)
    return PreparedInput(
        archive_path=archive,
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    )


def _input_base(sandbox: _LocalSandbox) -> Path:
    return Path(sandbox.task_data_root) / "demo" / "toy" / "base"


def _assert_no_input_staging_artifacts(base: Path) -> None:
    assert list(base.glob(".ale-input-*")) == []


def _populate_old_live_roots(base: Path) -> None:
    (base / "input").mkdir(parents=True)
    (base / "software").mkdir()
    (base / "input" / "required.txt").write_text("old input", encoding="utf-8")
    (base / "software" / "tool.sh").write_text("old software", encoding="utf-8")


def _assert_old_live_roots(base: Path) -> None:
    assert (base / "input" / "required.txt").read_text(encoding="utf-8") == "old input"
    assert (base / "software" / "tool.sh").read_text(encoding="utf-8") == "old software"


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
async def test_atomic_runtime_accepts_materialized_solve_checkout_without_git_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _make_solve_request(tmp_path)
    checkout = tmp_path / "materialized-checkout"
    shutil.copytree(request.task_repo / "tasks", checkout / "tasks")
    runtime_request = request.model_copy(update={"task_repo": checkout})
    runtime_spec = SimpleNamespace(environment=SimpleNamespace(), artifacts=None)
    provider = _FakeProvider(metadata={"image_id": request.image_id})

    monkeypatch.setattr(
        "ale_run.atomic.runtime.load_experiment",
        lambda _path: runtime_spec,
    )

    async with AtomicRuntime.open(request=runtime_request, provider=provider) as runtime:
        assert runtime.task_dir == checkout / "tasks" / request.task_path

    assert len(provider.acquire_calls) == 1


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
async def test_remote_input_cleanly_replaces_both_live_roots(tmp_path: Path) -> None:
    sandbox = _LocalSandbox(tmp_path / "vm")
    base = _input_base(sandbox)
    (base / "input").mkdir(parents=True)
    (base / "software").mkdir()
    (base / "reference").mkdir()
    (base / "output").mkdir()
    (base / "input" / "required.txt").write_text("old required", encoding="utf-8")
    (base / "input" / "stale.txt").write_text("stale input", encoding="utf-8")
    (base / "software" / "stale.sh").write_text("stale software", encoding="utf-8")
    (base / "reference" / "answer.txt").write_text("answer", encoding="utf-8")
    (base / "output" / "result.txt").write_text("result", encoding="utf-8")
    (base / "control.json").write_text("control", encoding="utf-8")
    prepared = _write_input_archive(
        tmp_path,
        {
            "input/required.txt": b"new required",
            "input/new.txt": b"new input",
            "software/tool.sh": b"new software",
        },
    )

    await stage_atomic_input(
        sandbox,
        _task_data(),
        source="oss://private-bucket/tasks",
        declared_input_paths=("input/required.txt",),
        prepared=prepared,
    )

    assert (base / "input" / "required.txt").read_bytes() == b"new required"
    assert (base / "input" / "new.txt").read_bytes() == b"new input"
    assert not (base / "input" / "stale.txt").exists()
    assert (base / "software" / "tool.sh").read_bytes() == b"new software"
    assert not (base / "software" / "stale.sh").exists()
    assert (base / "reference" / "answer.txt").read_text(encoding="utf-8") == "answer"
    assert (base / "output" / "result.txt").read_text(encoding="utf-8") == "result"
    assert (base / "control.json").read_text(encoding="utf-8") == "control"
    _assert_no_input_staging_artifacts(base)


@pytest.mark.asyncio
async def test_remote_input_removes_stale_software_when_archive_omits_it(
    tmp_path: Path,
) -> None:
    sandbox = _LocalSandbox(tmp_path / "vm")
    base = _input_base(sandbox)
    (base / "software").mkdir(parents=True)
    (base / "software" / "old-tool.sh").write_text("old", encoding="utf-8")
    prepared = _write_input_archive(tmp_path, {"input/required.txt": b"trusted"})

    await stage_atomic_input(
        sandbox,
        _task_data(),
        source="oss://private-bucket/tasks",
        declared_input_paths=("input/required.txt",),
        prepared=prepared,
    )

    assert not (base / "software").exists()
    assert (base / "input" / "required.txt").read_bytes() == b"trusted"
    _assert_no_input_staging_artifacts(base)


@pytest.mark.asyncio
async def test_remote_input_validates_declared_paths_only_in_staged_tree(
    tmp_path: Path,
) -> None:
    sandbox = _LocalSandbox(tmp_path / "vm")
    base = _input_base(sandbox)
    (base / "input").mkdir(parents=True)
    (base / "input" / "required.txt").write_text("old required", encoding="utf-8")
    prepared = _write_input_archive(tmp_path, {"input/other.txt": b"other"})

    with pytest.raises(AtomicInfrastructureError, match="declared input is missing"):
        await stage_atomic_input(
            sandbox,
            _task_data(),
            source="oss://private-bucket/tasks",
            declared_input_paths=("input/required.txt",),
            prepared=prepared,
        )

    assert (base / "input" / "required.txt").read_text(encoding="utf-8") == "old required"
    assert not (base / "input" / "other.txt").exists()
    _assert_no_input_staging_artifacts(base)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("members", "corrupt_member", "remove_type_metadata"),
    [
        ({"../escape.txt": b"unsafe"}, None, False),
        ({"input/required.txt": b"corrupt me"}, "input/required.txt", False),
        ({"input/required.txt": b"untyped"}, None, True),
    ],
    ids=["unsafe-member", "extraction-crc-error", "unknown-member-type"],
)
async def test_remote_input_validation_or_extraction_failure_preserves_live_roots(
    tmp_path: Path,
    members: dict[str, bytes],
    corrupt_member: str | None,
    remove_type_metadata: bool,
) -> None:
    sandbox = _LocalSandbox(tmp_path / "vm")
    base = _input_base(sandbox)
    (base / "input").mkdir(parents=True)
    (base / "software").mkdir()
    (base / "input" / "required.txt").write_text("old input", encoding="utf-8")
    (base / "software" / "tool.sh").write_text("old software", encoding="utf-8")
    prepared = _write_input_archive(
        tmp_path,
        members,
        corrupt_member=corrupt_member,
        remove_type_metadata=remove_type_metadata,
    )

    with pytest.raises(AtomicInfrastructureError):
        await stage_atomic_input(
            sandbox,
            _task_data(),
            source="oss://private-bucket/tasks",
            declared_input_paths=("input/required.txt",),
            prepared=prepared,
        )

    assert (base / "input" / "required.txt").read_text(encoding="utf-8") == "old input"
    assert (base / "software" / "tool.sh").read_text(encoding="utf-8") == "old software"
    assert not (tmp_path / "escape.txt").exists()
    _assert_no_input_staging_artifacts(base)


@pytest.mark.asyncio
async def test_remote_input_replaces_symlink_destinations_without_following_them(
    tmp_path: Path,
) -> None:
    sandbox = _LocalSandbox(tmp_path / "vm")
    base = _input_base(sandbox)
    base.mkdir(parents=True)
    external_input = tmp_path / "external-input"
    external_software = tmp_path / "external-software"
    external_input.mkdir()
    external_software.mkdir()
    (external_input / "old.txt").write_text("external input", encoding="utf-8")
    (external_software / "old.sh").write_text("external software", encoding="utf-8")
    (base / "input").symlink_to(external_input, target_is_directory=True)
    (base / "software").symlink_to(external_software, target_is_directory=True)
    prepared = _write_input_archive(
        tmp_path,
        {
            "input/required.txt": b"new input",
            "software/tool.sh": b"new software",
        },
    )

    await stage_atomic_input(
        sandbox,
        _task_data(),
        source="oss://private-bucket/tasks",
        declared_input_paths=("input/required.txt",),
        prepared=prepared,
    )

    assert not (base / "input").is_symlink()
    assert not (base / "software").is_symlink()
    assert (base / "input" / "required.txt").read_bytes() == b"new input"
    assert (base / "software" / "tool.sh").read_bytes() == b"new software"
    assert (external_input / "old.txt").read_text(encoding="utf-8") == "external input"
    assert (external_software / "old.sh").read_text(encoding="utf-8") == "external software"
    _assert_no_input_staging_artifacts(base)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "name"),
    [
        ("move-old", "input"),
        ("move-old", "software"),
        ("install-new", "input"),
        ("install-new", "software"),
        ("commit", "*"),
    ],
)
async def test_remote_input_replacement_failure_restores_both_roots(
    tmp_path: Path,
    operation: str,
    name: str,
) -> None:
    sandbox = _ScriptFaultSandbox(tmp_path / "vm", {(operation, name)})
    base = _input_base(sandbox)
    _populate_old_live_roots(base)
    prepared = _write_input_archive(
        tmp_path,
        {
            "input/required.txt": b"new input",
            "software/tool.sh": b"new software",
        },
    )

    with pytest.raises(AtomicInfrastructureError, match="injected"):
        await stage_atomic_input(
            sandbox,
            _task_data(),
            source="oss://private-bucket/tasks",
            declared_input_paths=("input/required.txt",),
            prepared=prepared,
        )

    _assert_old_live_roots(base)
    _assert_no_input_staging_artifacts(base)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rollback_operation", "name"),
    [
        ("remove-new", "input"),
        ("remove-new", "software"),
        ("restore-old", "input"),
        ("restore-old", "software"),
    ],
)
async def test_remote_input_incomplete_rollback_preserves_and_reports_backup(
    tmp_path: Path,
    rollback_operation: str,
    name: str,
) -> None:
    sandbox = _ScriptFaultSandbox(
        tmp_path / "vm",
        {("commit", "*"), (rollback_operation, name)},
    )
    base = _input_base(sandbox)
    _populate_old_live_roots(base)
    prepared = _write_input_archive(
        tmp_path,
        {
            "input/required.txt": b"new input",
            "software/tool.sh": b"new software",
        },
    )

    with pytest.raises(AtomicInfrastructureError) as caught:
        await stage_atomic_input(
            sandbox,
            _task_data(),
            source="oss://private-bucket/tasks",
            declared_input_paths=("input/required.txt",),
            prepared=prepared,
        )

    match = re.search(r"rollback backup preserved at ([^;]+)", str(caught.value))
    assert match is not None
    backup_root = Path(match.group(1))
    assert backup_root.is_dir()
    assert (backup_root / ".previous" / name).exists()
    assert "injected commit failure" in str(caught.value)
    assert f"injected {rollback_operation} failure for {name}" in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "members",
    [{}, {"software/tool.sh": b"software only"}],
    ids=["empty-archive", "software-only"],
)
async def test_remote_archive_without_input_preserves_live_roots(
    tmp_path: Path,
    members: dict[str, bytes],
) -> None:
    sandbox = _LocalSandbox(tmp_path / "vm")
    base = _input_base(sandbox)
    _populate_old_live_roots(base)
    prepared = _write_input_archive(tmp_path, members)

    with pytest.raises(AtomicInfrastructureError, match="contains no input"):
        await stage_atomic_input(
            sandbox,
            _task_data(),
            source="oss://private-bucket/tasks",
            declared_input_paths=(),
            prepared=prepared,
        )

    _assert_old_live_roots(base)
    _assert_no_input_staging_artifacts(base)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    ["upload", "script-write", "config-write", "run"],
)
async def test_host_removes_staging_artifacts_after_orchestration_failure(
    tmp_path: Path,
    failure: str,
) -> None:
    sandbox = _OrchestrationFailingSandbox(tmp_path / "vm", failure)
    base = _input_base(sandbox)
    prepared = _write_input_archive(tmp_path, {"input/required.txt": b"trusted"})

    with pytest.raises(RuntimeError, match=f"injected {failure} failure"):
        await stage_atomic_input(
            sandbox,
            _task_data(),
            source="oss://private-bucket/tasks",
            declared_input_paths=("input/required.txt",),
            prepared=prepared,
        )

    _assert_no_input_staging_artifacts(base)
    assert sandbox.rm_calls


@pytest.mark.asyncio
async def test_host_cleanup_errors_do_not_mask_orchestration_failure(
    tmp_path: Path,
) -> None:
    sandbox = _OrchestrationFailingSandbox(
        tmp_path / "vm",
        "run",
        cleanup_fails=True,
    )
    prepared = _write_input_archive(tmp_path, {"input/required.txt": b"trusted"})

    with pytest.raises(RuntimeError, match="injected run failure") as caught:
        await stage_atomic_input(
            sandbox,
            _task_data(),
            source="oss://private-bucket/tasks",
            declared_input_paths=("input/required.txt",),
            prepared=prepared,
        )

    assert any("staging cleanup failed" in note for note in caught.value.__notes__)
    assert len(sandbox.rm_calls) >= 4


def test_vm_config_parse_failure_removes_known_archive_and_control_files(
    tmp_path: Path,
) -> None:
    base = tmp_path / "task-data"
    base.mkdir()
    script = base / ".ale-input-invalid.py"
    config = base / ".ale-input-invalid.json"
    archive = base / ".ale-input-invalid.zip"
    script.write_text(_STAGE_SCRIPT, encoding="utf-8")
    config.write_text("{", encoding="utf-8")
    archive.write_bytes(b"uploaded archive")

    result = subprocess.run(
        [sys.executable, str(script), str(config)],
        capture_output=True,
        check=False,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 2
    assert payload["ok"] is False
    assert "Expecting property name" in payload["error"]
    assert not archive.exists()
    assert not config.exists()
    assert not script.exists()


def test_vm_cleanup_attempts_all_artifacts_without_masking_primary_failure(
    tmp_path: Path,
) -> None:
    base = tmp_path / "task-data"
    base.mkdir()
    script = base / ".ale-input-cleanup.py"
    config = base / ".ale-input-cleanup.json"
    archive = base / ".ale-input-cleanup.zip"
    prepared = _write_input_archive(tmp_path, {"software/tool.sh": b"software only"})
    shutil.copyfile(prepared.archive_path, archive)
    injected_script = _STAGE_SCRIPT.replace(
        "archive.unlink(missing_ok=True)",
        "raise OSError('injected archive cleanup failure')",
    )
    script.write_text(injected_script, encoding="utf-8")
    config.write_text(
        json.dumps(
            {
                "archive_path": str(archive),
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "base": str(base),
                "declared_input_paths": [],
                "max_files": 100,
                "max_file_bytes": 1024,
                "max_source_bytes": 4096,
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(script), str(config)],
        capture_output=True,
        check=False,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 2
    assert payload["ok"] is False
    assert "contains no input" in payload["error"]
    assert "injected archive cleanup failure" in payload["error"]
    assert archive.exists()
    assert not config.exists()
    assert not script.exists()
    assert not list(base.glob("*.stage"))


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
