from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
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
    _download_prefix,
    _OssObject,
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


def _oss_listing(rows: list[tuple[int, str]]) -> bytes:
    lines = [
        "LastModifiedTime              Size(B)  StorageClass   ETAG                              ObjectName"
    ]
    lines.extend(
        f"2026-07-29 12:00:00 +0800 CST  {size}  Standard  0123456789ABCDEF0123456789ABCDEF  {url}"
        for size, url in rows
    )
    lines.append(f"Object Number is: {len(rows)}")
    return ("\n".join(lines) + "\n").encode()


class _CleanupFailingTemporaryDirectory:
    def __init__(self, real_factory, *args, **kwargs) -> None:
        self._inner = real_factory(*args, **kwargs)
        self.name = self._inner.name
        self.cleanup_calls = 0

    def __enter__(self) -> str:
        return self._inner.__enter__()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.cleanup()

    def cleanup(self) -> None:
        self.cleanup_calls += 1
        self._inner.cleanup()
        raise PermissionError("injected temporary cleanup failure")


class _LocalSandbox:
    def __init__(self, root: Path) -> None:
        self.is_linux = True
        self.python = sys.executable
        self.task_data_root = str(root / "task-data")
        self.commands: list[str] = []
        self.cleanup_commands: list[str] = []

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
        cleanup_outcome: str | None = None,
    ) -> None:
        super().__init__(root)
        self.failure = failure
        self.cleanup_outcome = cleanup_outcome

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
        if "ale-input-cleanup" in command:
            self.cleanup_commands.append(command)
            if self.cleanup_outcome == "nonzero":
                return subprocess.CompletedProcess(
                    command,
                    returncode=17,
                    stdout="",
                    stderr="injected cleanup nonzero",
                )
            if self.cleanup_outcome == "transport":
                raise OSError("injected cleanup transport failure")
            return await super().run_command(command, timeout=timeout)
        if self.failure == "run":
            raise RuntimeError("injected run failure")
        return await super().run_command(command, timeout=timeout)


class _VerifierPayloadSandbox(_LocalSandbox):
    def __init__(
        self,
        root: Path,
        *,
        stdout: str = "",
        returncode: int = 0,
        run_error: Exception | None = None,
        is_linux: bool = True,
    ) -> None:
        super().__init__(root)
        self.stdout = stdout
        self.returncode = returncode
        self.run_error = run_error
        self.is_linux = is_linux
        self.config_path: Path | None = None
        self.staging_path: Path | None = None

    async def write_file(self, path: str, content: str | bytes) -> None:
        await super().write_file(path, content)
        if path.endswith(".json"):
            self.config_path = Path(path)

    async def run_command(
        self,
        command: str,
        *,
        timeout: float = 60,
    ) -> subprocess.CompletedProcess[str]:
        if "ale-input-cleanup" in command:
            self.cleanup_commands.append(command)
            return subprocess.CompletedProcess(
                command,
                returncode=0,
                stdout="",
                stderr="",
            )
        assert self.config_path is not None
        self.staging_path = self.config_path.with_suffix(".stage")
        (self.staging_path / ".previous" / "input").mkdir(parents=True)
        (self.staging_path / ".previous" / "input" / "required.txt").write_text(
            "rollback copy",
            encoding="utf-8",
        )
        if self.run_error is not None:
            raise self.run_error
        return subprocess.CompletedProcess(
            command,
            returncode=self.returncode,
            stdout=self.stdout,
            stderr="",
        )


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
    input_prefix = "oss://private-bucket/tasks/demo/toy/base/input/"
    software_prefix = "oss://private-bucket/tasks/demo/toy/base/software/"
    objects = {
        f"{input_prefix}required.txt": b"trusted input",
        f"{software_prefix}tool.sh": b"#!/bin/sh\n",
    }

    async def host_ossutil(*arguments: str):
        host_calls.append(arguments)
        if arguments[0] == "ls":
            prefix = arguments[3]
            rows = [
                (len(content), url) for url, content in objects.items() if url.startswith(prefix)
            ]
            return 0, _oss_listing(rows), b""
        assert arguments[0:3] == ("cp", "--payer", "requester")
        Path(arguments[4]).write_bytes(objects[arguments[3]])
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
    assert host_calls[:2] == [
        ("ls", "--payer", "requester", input_prefix, "--limited-num", "16"),
        ("ls", "--payer", "requester", software_prefix, "--limited-num", "16"),
    ]
    assert [call[:4] for call in host_calls[2:]] == [
        ("cp", "--payer", "requester", f"{input_prefix}required.txt"),
        ("cp", "--payer", "requester", f"{software_prefix}tool.sh"),
    ]
    assert not any(call[0] == "sync" for call in host_calls)
    vm_commands = "\n".join(sandbox.commands).lower()
    assert "ossutil" not in vm_commands
    assert "100.100.100.200" not in vm_commands
    assert "private-bucket" not in vm_commands


@pytest.mark.asyncio
async def test_oss_metadata_listing_paginates_before_exact_object_downloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    input_prefix = "oss://private-bucket/tasks/demo/toy/base/input/"
    software_prefix = "oss://private-bucket/tasks/demo/toy/base/software/"
    objects = {f"{input_prefix}{index:02d}.txt": b"x" for index in range(17)}

    async def host_ossutil(*arguments: str):
        calls.append(arguments)
        if arguments[0] == "ls":
            prefix = arguments[3]
            marker = arguments[7] if len(arguments) == 8 else None
            urls = sorted(url for url in objects if url.startswith(prefix))
            if marker is not None:
                urls = [url for url in urls if url.removeprefix("oss://private-bucket/") > marker]
            page = [(1, url) for url in urls[:16]]
            return 0, _oss_listing(page) if page else b"Object Number is: 0\n", b""
        Path(arguments[4]).write_bytes(objects[arguments[3]])
        return 0, b"", b""

    monkeypatch.setattr(
        "ale_run.atomic.trusted_staging.run_host_ossutil",
        host_ossutil,
    )

    async with prepare_atomic_input(
        source="oss://private-bucket/tasks",
        task_data=_task_data(),
        declared_input_paths=("input/00.txt",),
    ):
        pass

    expected_marker = "tasks/demo/toy/base/input/15.txt"
    assert calls[:3] == [
        ("ls", "--payer", "requester", input_prefix, "--limited-num", "16"),
        (
            "ls",
            "--payer",
            "requester",
            input_prefix,
            "--limited-num",
            "16",
            "--marker",
            expected_marker,
        ),
        ("ls", "--payer", "requester", software_prefix, "--limited-num", "16"),
    ]
    assert [call[3] for call in calls if call[0] == "cp"] == sorted(objects)
    assert not any(call[0] == "sync" for call in calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_prefix", ["input", "software"])
async def test_oss_metadata_list_failure_is_never_treated_as_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_prefix: str,
) -> None:
    calls: list[tuple[str, ...]] = []

    async def host_ossutil(*arguments: str):
        calls.append(arguments)
        assert arguments[0] == "ls"
        if f"/{failing_prefix}/" in arguments[3]:
            return 9, b"", b"AccessDenied"
        url = f"{arguments[3]}required.txt"
        return 0, _oss_listing([(1, url)]), b""

    monkeypatch.setattr("ale_run.atomic.trusted_staging.run_host_ossutil", host_ossutil)

    with pytest.raises(AtomicInfrastructureError, match="cannot list OSS prefix") as caught:
        async with prepare_atomic_input(
            source="oss://private-bucket/tasks",
            task_data=_task_data(),
            declared_input_paths=("input/required.txt",),
        ):
            pass

    assert caught.value.category == "input"
    assert not any(call[0] == "cp" for call in calls)


@pytest.mark.asyncio
async def test_required_oss_input_prefix_cannot_be_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def host_ossutil(*arguments: str):
        return 0, b"Object Number is: 0\n0.0123(s) elapsed\n", b""

    monkeypatch.setattr("ale_run.atomic.trusted_staging.run_host_ossutil", host_ossutil)

    with pytest.raises(
        AtomicInfrastructureError, match="required OSS input prefix is empty"
    ) as caught:
        async with prepare_atomic_input(
            source="oss://private-bucket/tasks",
            task_data=_task_data(),
            declared_input_paths=("input/required.txt",),
        ):
            pass

    assert caught.value.category == "input"


@pytest.mark.asyncio
async def test_oss_metadata_accepts_documented_timezone_spaces_and_unicode_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_prefix = "oss://private-bucket/tasks/demo/toy/base/input/"
    input_url = f"{input_prefix}dir/space \u4e2d\u6587.txt"
    calls: list[tuple[str, ...]] = []

    async def host_ossutil(*arguments: str):
        calls.append(arguments)
        if arguments[0] == "ls":
            if "/software/" in arguments[3]:
                return 0, b"Object Number is: 0\n", b""
            output = _oss_listing([(1, input_url)]).replace(b"+0800 CST", b"+0000 UTC")
            return 0, output, b""
        Path(arguments[4]).write_bytes(b"x")
        return 0, b"", b""

    monkeypatch.setattr("ale_run.atomic.trusted_staging.run_host_ossutil", host_ossutil)

    async with prepare_atomic_input(
        source="oss://private-bucket/tasks",
        task_data=_task_data(),
        declared_input_paths=("input/dir/space \u4e2d\u6587.txt",),
    ):
        pass

    assert [call[3] for call in calls if call[0] == "cp"] == [input_url]


@pytest.mark.asyncio
async def test_oss_metadata_accepts_blank_line_before_elapsed_footer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_prefix = "oss://private-bucket/tasks/demo/toy/base/input/"
    input_url = f"{input_prefix}required.txt"
    calls: list[tuple[str, ...]] = []

    async def host_ossutil(*arguments: str):
        calls.append(arguments)
        if arguments[0] == "ls":
            if "/software/" in arguments[3]:
                return 0, b"Object Number is: 0\n\n0.001(s) elapsed\n", b""
            return 0, _oss_listing([(1, input_url)]) + b"\n0.123(s) elapsed\n", b""
        Path(arguments[4]).write_bytes(b"x")
        return 0, b"", b""

    monkeypatch.setattr("ale_run.atomic.trusted_staging.run_host_ossutil", host_ossutil)

    async with prepare_atomic_input(
        source="oss://private-bucket/tasks",
        task_data=_task_data(),
        declared_input_paths=("input/required.txt",),
    ):
        pass

    assert [call[3] for call in calls if call[0] == "cp"] == [input_url]


@pytest.mark.asyncio
async def test_oss_metadata_rejects_a_backward_pagination_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_prefix = "oss://private-bucket/tasks/demo/toy/base/input/"
    calls: list[tuple[str, ...]] = []
    first_page = [(1, f"{input_prefix}{chr(ord('b') + index)}") for index in range(16)]

    async def host_ossutil(*arguments: str):
        calls.append(arguments)
        if len(calls) == 1:
            return 0, _oss_listing(first_page), b""
        return 0, _oss_listing([(1, f"{input_prefix}a")]), b""

    monkeypatch.setattr("ale_run.atomic.trusted_staging.run_host_ossutil", host_ossutil)

    with pytest.raises(AtomicInfrastructureError, match="marker did not advance"):
        async with prepare_atomic_input(
            source="oss://private-bucket/tasks",
            task_data=_task_data(),
            declared_input_paths=("input/b",),
        ):
            pass

    assert calls[1][-2:] == ("--marker", "tasks/demo/toy/base/input/q")
    assert not any(call[0] == "cp" for call in calls)


@pytest.mark.asyncio
async def test_oss_metadata_listing_output_is_bounded_even_with_a_mocked_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def host_ossutil(*arguments: str):
        return 0, b"x" * (64 * 1024 + 1), b""

    monkeypatch.setattr("ale_run.atomic.trusted_staging.run_host_ossutil", host_ossutil)

    with pytest.raises(AtomicInfrastructureError, match="exceeded 64 KiB"):
        async with prepare_atomic_input(
            source="oss://private-bucket/tasks",
            task_data=_task_data(),
            declared_input_paths=("input/a",),
        ):
            pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "input_listing",
    [
        b"unexpected output\nObject Number is: 1\n",
        _oss_listing([(1, "oss://private-bucket/tasks/demo/toy/base/input/a.txt")]).replace(
            b"Object Number is: 1", b"Object Number is: 2"
        ),
        _oss_listing(
            [
                (1, "oss://private-bucket/tasks/demo/toy/base/input/a.txt"),
                (1, "oss://private-bucket/tasks/demo/toy/base/input/a.txt"),
            ]
        ),
        _oss_listing([(0, "oss://private-bucket/tasks/demo/toy/base/input/")]),
        _oss_listing([(1, "oss://private-bucket/tasks/demo/toy/base/input/../escape")]),
        _oss_listing([(1, "oss://private-bucket/tasks/demo/toy/base/input/a//b")]),
        _oss_listing([(1, "oss://private-bucket/tasks/demo/toy/base/input/a\\b")]),
    ],
    ids=[
        "malformed-row",
        "summary-mismatch",
        "duplicate-key",
        "directory-marker",
        "parent-traversal",
        "noncanonical-path",
        "backslash",
    ],
)
async def test_oss_metadata_listing_rejects_ambiguous_or_unsafe_objects_before_cp(
    monkeypatch: pytest.MonkeyPatch,
    input_listing: bytes,
) -> None:
    calls: list[tuple[str, ...]] = []

    async def host_ossutil(*arguments: str):
        calls.append(arguments)
        return 0, input_listing, b""

    monkeypatch.setattr("ale_run.atomic.trusted_staging.run_host_ossutil", host_ossutil)

    with pytest.raises(AtomicInfrastructureError, match="OSS"):
        async with prepare_atomic_input(
            source="oss://private-bucket/tasks",
            task_data=_task_data(),
            declared_input_paths=("input/a.txt",),
        ):
            pass

    assert not any(call[0] == "cp" for call in calls)


@pytest.mark.asyncio
async def test_oss_local_namespace_collisions_are_rejected_before_any_cp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    input_url = "oss://private-bucket/tasks/demo/toy/base/input/required.txt"

    async def host_ossutil(*arguments: str):
        calls.append(arguments)
        if arguments[0] == "ls":
            rows = (
                [(1, input_url)]
                if "/input/" in arguments[3]
                else [(1, f"{arguments[3]}{name}") for name in ["a", "a/b"]]
            )
            return 0, _oss_listing(rows), b""
        Path(arguments[4]).write_bytes(b"x")
        return 0, b"", b""

    monkeypatch.setattr("ale_run.atomic.trusted_staging.run_host_ossutil", host_ossutil)

    with pytest.raises(AtomicInfrastructureError, match="namespace collision") as caught:
        async with prepare_atomic_input(
            source="oss://private-bucket/tasks",
            task_data=_task_data(),
            declared_input_paths=("input/required.txt",),
        ):
            pass

    assert caught.value.category == "input"
    assert not any(call[0] == "cp" for call in calls)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "relative_paths",
    [["a", "a/b"], ["a/b", "a"]],
    ids=["ancestor-first", "descendant-first"],
)
async def test_oss_download_namespace_collision_is_order_independent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_paths: list[str],
) -> None:
    calls: list[tuple[str, ...]] = []

    async def host_ossutil(*arguments: str):
        calls.append(arguments)
        return 0, b"", b""

    monkeypatch.setattr("ale_run.atomic.trusted_staging.run_host_ossutil", host_ossutil)
    objects = [
        _OssObject(
            url=f"oss://bucket/input/{relative}",
            key=f"input/{relative}",
            relative_path=Path(relative),
            size_bytes=1,
        )
        for relative in relative_paths
    ]

    with pytest.raises(AtomicInfrastructureError, match="namespace collision") as caught:
        await _download_prefix(objects, tmp_path / "payload", actual_total=0)

    assert caught.value.category == "input"
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "component",
    ["a" * 256, "\u754c" * 86],
    ids=["ascii-256-bytes", "unicode-258-bytes"],
)
async def test_oss_overlong_local_components_are_rejected_before_any_cp(
    monkeypatch: pytest.MonkeyPatch,
    component: str,
) -> None:
    calls: list[tuple[str, ...]] = []
    input_url = "oss://private-bucket/tasks/demo/toy/base/input/required.txt"

    async def host_ossutil(*arguments: str):
        calls.append(arguments)
        if arguments[0] == "ls":
            rows = (
                [(1, input_url)]
                if "/input/" in arguments[3]
                else [(1, f"{arguments[3]}{component}")]
            )
            return 0, _oss_listing(rows), b""
        Path(arguments[4]).write_bytes(b"x")
        return 0, b"", b""

    monkeypatch.setattr("ale_run.atomic.trusted_staging.run_host_ossutil", host_ossutil)

    with pytest.raises(AtomicInfrastructureError, match="255 bytes") as caught:
        async with prepare_atomic_input(
            source="oss://private-bucket/tasks",
            task_data=_task_data(),
            declared_input_paths=("input/required.txt",),
        ):
            pass

    assert caught.value.category == "input"
    assert not any(call[0] == "cp" for call in calls)


@pytest.mark.asyncio
async def test_oss_metadata_rejects_header_with_zero_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    header_with_zero = _oss_listing([])

    async def host_ossutil(*arguments: str):
        calls.append(arguments)
        return 0, header_with_zero, b""

    monkeypatch.setattr("ale_run.atomic.trusted_staging.run_host_ossutil", host_ossutil)

    with pytest.raises(AtomicInfrastructureError, match="zero-object") as caught:
        async with prepare_atomic_input(
            source="oss://private-bucket/tasks",
            task_data=_task_data(),
            declared_input_paths=("input/required.txt",),
        ):
            pass

    assert caught.value.category == "input"
    assert not any(call[0] == "cp" for call in calls)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("input_rows", "software_rows", "constant", "limit", "message"),
    [
        (["a", "b"], ["c"], "_MAX_STAGED_FILES", 2, "file-count"),
        (["a"], [], "_MAX_STAGED_FILE_BYTES", 0, "512 MiB"),
        (["a"], ["b"], "_MAX_STAGED_SOURCE_BYTES", 1, "total-size"),
    ],
)
async def test_oss_metadata_limits_are_preflighted_across_input_and_software_before_cp(
    monkeypatch: pytest.MonkeyPatch,
    input_rows: list[str],
    software_rows: list[str],
    constant: str,
    limit: int,
    message: str,
) -> None:
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(f"ale_run.atomic.trusted_staging.{constant}", limit)

    async def host_ossutil(*arguments: str):
        calls.append(arguments)
        names = input_rows if "/input/" in arguments[3] else software_rows
        rows = [(1, f"{arguments[3]}{name}") for name in names]
        return 0, _oss_listing(rows) if rows else b"Object Number is: 0\n", b""

    monkeypatch.setattr("ale_run.atomic.trusted_staging.run_host_ossutil", host_ossutil)

    with pytest.raises(AtomicInfrastructureError, match=message):
        async with prepare_atomic_input(
            source="oss://private-bucket/tasks",
            task_data=_task_data(),
            declared_input_paths=("input/a",),
        ):
            pass

    assert [call[0] for call in calls] == ["ls", "ls"]


@pytest.mark.asyncio
async def test_oss_metadata_exact_count_file_and_total_limits_are_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr("ale_run.atomic.trusted_staging._MAX_STAGED_FILES", 2)
    monkeypatch.setattr("ale_run.atomic.trusted_staging._MAX_STAGED_FILE_BYTES", 5)
    monkeypatch.setattr("ale_run.atomic.trusted_staging._MAX_STAGED_SOURCE_BYTES", 10)

    async def host_ossutil(*arguments: str):
        calls.append(arguments)
        if arguments[0] == "ls":
            suffix = "required.txt" if "/input/" in arguments[3] else "tool.bin"
            return 0, _oss_listing([(5, f"{arguments[3]}{suffix}")]), b""
        Path(arguments[4]).write_bytes(b"12345")
        return 0, b"", b""

    monkeypatch.setattr("ale_run.atomic.trusted_staging.run_host_ossutil", host_ossutil)

    async with prepare_atomic_input(
        source="oss://private-bucket/tasks",
        task_data=_task_data(),
        declared_input_paths=("input/required.txt",),
    ):
        pass

    assert [call[0] for call in calls].count("cp") == 2
    assert not any(call[0] == "sync" for call in calls)


@pytest.mark.asyncio
async def test_oss_metadata_full_terminal_page_is_followed_by_empty_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    input_prefix = "oss://private-bucket/tasks/demo/toy/base/input/"
    rows = [(1, f"{input_prefix}{index:02d}.txt") for index in range(16)]

    async def host_ossutil(*arguments: str):
        calls.append(arguments)
        if arguments[0] == "ls":
            if "/software/" in arguments[3] or "--marker" in arguments:
                return 0, b"Object Number is: 0\n", b""
            return 0, _oss_listing(rows), b""
        Path(arguments[4]).write_bytes(b"x")
        return 0, b"", b""

    monkeypatch.setattr("ale_run.atomic.trusted_staging.run_host_ossutil", host_ossutil)

    async with prepare_atomic_input(
        source="oss://private-bucket/tasks",
        task_data=_task_data(),
        declared_input_paths=("input/00.txt",),
    ):
        pass

    ls_calls = [call for call in calls if call[0] == "ls"]
    assert len(ls_calls) == 3
    assert ls_calls[1][-2:] == ("--marker", "tasks/demo/toy/base/input/15.txt")
    assert [call[0] for call in calls].count("cp") == 16
    assert not any(call[0] == "sync" for call in calls)


@pytest.mark.asyncio
async def test_oss_metadata_rejects_input2_as_outside_input_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    async def host_ossutil(*arguments: str):
        calls.append(arguments)
        return 0, _oss_listing([(1, arguments[3].replace("/input/", "/input2/") + "a")]), b""

    monkeypatch.setattr("ale_run.atomic.trusted_staging.run_host_ossutil", host_ossutil)

    with pytest.raises(AtomicInfrastructureError, match="outside the requested prefix"):
        async with prepare_atomic_input(
            source="oss://private-bucket/tasks",
            task_data=_task_data(),
            declared_input_paths=("input/a",),
        ):
            pass

    assert not any(call[0] == "cp" for call in calls)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "raised"),
    [
        (
            "list-submission-storage",
            AtomicInfrastructureError("submission_storage", "runner failed"),
        ),
        ("list-permission", PermissionError("listing denied")),
        ("copy-submission-storage", AtomicInfrastructureError("submission_storage", "copy failed")),
        ("copy-oserror", OSError("copy transport failed")),
    ],
)
async def test_host_oss_runner_exceptions_are_normalized_to_input(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    raised: Exception,
) -> None:
    calls: list[tuple[str, ...]] = []
    input_url = "oss://private-bucket/tasks/demo/toy/base/input/required.txt"

    async def host_ossutil(*arguments: str):
        calls.append(arguments)
        if operation.startswith("list") or arguments[0] == "cp":
            raise raised
        rows = [(1, input_url)] if "/input/" in arguments[3] else []
        return 0, _oss_listing(rows) if rows else b"Object Number is: 0\n", b""

    monkeypatch.setattr("ale_run.atomic.trusted_staging.run_host_ossutil", host_ossutil)

    with pytest.raises(AtomicInfrastructureError) as caught:
        async with prepare_atomic_input(
            source="oss://private-bucket/tasks",
            task_data=_task_data(),
            declared_input_paths=("input/required.txt",),
        ):
            pass

    assert caught.value.category == "input"
    assert caught.value.__cause__ is raised
    assert len(caught.value.message) <= 2000
    assert not any(call[0] == "sync" for call in calls)


@pytest.mark.asyncio
async def test_host_tree_setup_oserror_is_normalized_to_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_chmod = os.chmod

    def denied_chmod(path, mode, *, dir_fd=None, follow_symlinks=True):
        if str(path).startswith(tempfile.gettempdir()) and mode == 0o700:
            raise PermissionError("tree chmod denied")
        return original_chmod(path, mode, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

    monkeypatch.setattr("ale_run.atomic.trusted_staging.os.chmod", denied_chmod)

    with pytest.raises(AtomicInfrastructureError) as caught:
        async with prepare_atomic_input(
            source="oss://private-bucket/tasks",
            task_data=_task_data(),
            declared_input_paths=("input/required.txt",),
        ):
            pass

    assert caught.value.category == "input"
    assert isinstance(caught.value.__cause__, PermissionError)


@pytest.mark.asyncio
async def test_host_temporary_cleanup_failure_is_normalized_to_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_factory = tempfile.TemporaryDirectory
    created: list[_CleanupFailingTemporaryDirectory] = []

    def failing_factory(*args, **kwargs):
        temporary = _CleanupFailingTemporaryDirectory(original_factory, *args, **kwargs)
        created.append(temporary)
        return temporary

    async def host_ossutil(*arguments: str):
        if arguments[0] == "ls":
            if "/software/" in arguments[3]:
                return 0, b"Object Number is: 0\n", b""
            return 0, _oss_listing([(1, f"{arguments[3]}required.txt")]), b""
        Path(arguments[4]).write_bytes(b"x")
        return 0, b"", b""

    monkeypatch.setattr(
        "ale_run.atomic.trusted_staging.tempfile.TemporaryDirectory", failing_factory
    )
    monkeypatch.setattr("ale_run.atomic.trusted_staging.run_host_ossutil", host_ossutil)

    with pytest.raises(AtomicInfrastructureError) as caught:
        async with prepare_atomic_input(
            source="oss://private-bucket/tasks",
            task_data=_task_data(),
            declared_input_paths=("input/required.txt",),
        ):
            pass

    assert caught.value.category == "input"
    assert "temporary staging cleanup failed" in caught.value.message
    assert len(caught.value.message) <= 2000
    assert isinstance(caught.value.__cause__, PermissionError)
    assert len(created) == 1
    assert created[0].cleanup_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_phase", ["staging", "consumer"])
async def test_host_temporary_cleanup_failure_does_not_mask_primary_error(
    monkeypatch: pytest.MonkeyPatch,
    failure_phase: str,
) -> None:
    original_factory = tempfile.TemporaryDirectory
    created: list[_CleanupFailingTemporaryDirectory] = []
    primary = AtomicInfrastructureError(
        "input" if failure_phase == "staging" else "consumer",
        f"primary {failure_phase} failure",
    )

    def failing_factory(*args, **kwargs):
        temporary = _CleanupFailingTemporaryDirectory(original_factory, *args, **kwargs)
        created.append(temporary)
        return temporary

    async def host_ossutil(*arguments: str):
        if failure_phase == "staging":
            raise primary
        if arguments[0] == "ls":
            if "/software/" in arguments[3]:
                return 0, b"Object Number is: 0\n", b""
            return 0, _oss_listing([(1, f"{arguments[3]}required.txt")]), b""
        Path(arguments[4]).write_bytes(b"x")
        return 0, b"", b""

    monkeypatch.setattr(
        "ale_run.atomic.trusted_staging.tempfile.TemporaryDirectory", failing_factory
    )
    monkeypatch.setattr("ale_run.atomic.trusted_staging.run_host_ossutil", host_ossutil)

    with pytest.raises(AtomicInfrastructureError) as caught:
        async with prepare_atomic_input(
            source="oss://private-bucket/tasks",
            task_data=_task_data(),
            declared_input_paths=("input/required.txt",),
        ):
            if failure_phase == "consumer":
                raise primary

    if failure_phase == "consumer":
        assert caught.value is primary
    else:
        assert caught.value.category == "input"
        assert caught.value.__cause__ is primary
        assert "cannot list OSS prefix" in caught.value.message
    assert caught.value.category == primary.category
    assert "temporary staging cleanup failed" in "\n".join(caught.value.__notes__)
    assert len(created) == 1
    assert created[0].cleanup_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("actual_sizes", "source_limit", "message"),
    [
        ([2], 10, "size changed"),
        ([1, 10], 10, "total-size"),
    ],
    ids=["per-object-drift", "cumulative-actual-overrun"],
)
async def test_oss_download_rechecks_local_sizes_and_cumulative_actual_bytes(
    monkeypatch: pytest.MonkeyPatch,
    actual_sizes: list[int],
    source_limit: int,
    message: str,
) -> None:
    calls: list[tuple[str, ...]] = []
    input_prefix = "oss://private-bucket/tasks/demo/toy/base/input/"
    metadata_rows = [(1, f"{input_prefix}{index}.txt") for index in range(len(actual_sizes))]
    monkeypatch.setattr("ale_run.atomic.trusted_staging._MAX_STAGED_SOURCE_BYTES", source_limit)

    async def host_ossutil(*arguments: str):
        calls.append(arguments)
        if arguments[0] == "ls":
            rows = metadata_rows if "/input/" in arguments[3] else []
            return 0, _oss_listing(rows) if rows else b"Object Number is: 0\n", b""
        index = int(arguments[3].rsplit("/", 1)[1].removesuffix(".txt"))
        Path(arguments[4]).write_bytes(b"x" * actual_sizes[index])
        return 0, b"", b""

    monkeypatch.setattr("ale_run.atomic.trusted_staging.run_host_ossutil", host_ossutil)

    with pytest.raises(AtomicInfrastructureError, match=message):
        async with prepare_atomic_input(
            source="oss://private-bucket/tasks",
            task_data=_task_data(),
            declared_input_paths=("input/0.txt",),
        ):
            pass

    assert not any(call[0] == "sync" for call in calls)


@pytest.mark.asyncio
async def test_empty_optional_software_prefix_is_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    input_url = "oss://private-bucket/tasks/demo/toy/base/input/required.txt"

    async def host_ossutil(*arguments: str):
        calls.append(arguments)
        if arguments[0] == "ls":
            if "/input/" in arguments[3]:
                return 0, _oss_listing([(1, input_url)]), b""
            return 0, b"Object Number is: 0\n", b""
        Path(arguments[4]).write_bytes(b"x")
        return 0, b"", b""

    monkeypatch.setattr("ale_run.atomic.trusted_staging.run_host_ossutil", host_ossutil)

    async with prepare_atomic_input(
        source="oss://private-bucket/tasks",
        task_data=_task_data(),
        declared_input_paths=("input/required.txt",),
    ):
        pass

    assert [call[3] for call in calls if call[0] == "cp"] == [input_url]


@pytest.mark.asyncio
async def test_oss_download_rejects_a_preexisting_symlink_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "payload"
    destination.mkdir(mode=0o700)
    outside = tmp_path / "outside"
    outside.mkdir()
    (destination / "dir").symlink_to(outside, target_is_directory=True)
    calls: list[tuple[str, ...]] = []

    async def host_ossutil(*arguments: str):
        calls.append(arguments)
        return 0, b"", b""

    monkeypatch.setattr("ale_run.atomic.trusted_staging.run_host_ossutil", host_ossutil)

    with pytest.raises(AtomicInfrastructureError, match="unsafe host OSS staging parent"):
        await _download_prefix(
            [
                _OssObject(
                    url="oss://bucket/input/dir/file",
                    key="input/dir/file",
                    relative_path=Path("dir/file"),
                    size_bytes=1,
                )
            ],
            destination,
            actual_total=0,
        )

    assert calls == []


@pytest.mark.asyncio
async def test_oss_download_rechecks_parent_containment_after_cp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "payload"
    outside = tmp_path / "outside"
    outside.mkdir()

    async def host_ossutil(*arguments: str):
        target = Path(arguments[4])
        target.parent.rmdir()
        target.parent.symlink_to(outside, target_is_directory=True)
        target.write_bytes(b"x")
        return 0, b"", b""

    monkeypatch.setattr("ale_run.atomic.trusted_staging.run_host_ossutil", host_ossutil)

    with pytest.raises(AtomicInfrastructureError, match="unsafe host OSS staging parent"):
        await _download_prefix(
            [
                _OssObject(
                    url="oss://bucket/input/dir/file",
                    key="input/dir/file",
                    relative_path=Path("dir/file"),
                    size_bytes=1,
                )
            ],
            destination,
            actual_total=0,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("replacement", ["destination", "parent", "target"])
async def test_oss_download_rejects_path_replacement_after_cp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
) -> None:
    destination = tmp_path / "payload"
    outside = tmp_path / "outside"
    outside.mkdir()

    async def host_ossutil(*arguments: str):
        target = Path(arguments[4])
        if replacement == "destination":
            shutil.rmtree(target.parent.parent)
            target.parent.parent.mkdir()
            target.parent.mkdir()
            target.write_bytes(b"x")
        elif replacement == "parent":
            target.parent.rmdir()
            target.parent.mkdir()
            target.write_bytes(b"x")
        else:
            target.symlink_to(outside / "replacement")
        return 0, b"", b""

    monkeypatch.setattr("ale_run.atomic.trusted_staging.run_host_ossutil", host_ossutil)

    with pytest.raises(AtomicInfrastructureError) as caught:
        await _download_prefix(
            [
                _OssObject(
                    url="oss://bucket/input/dir/file",
                    key="input/dir/file",
                    relative_path=Path("dir/file"),
                    size_bytes=1,
                )
            ],
            destination,
            actual_total=0,
        )

    assert caught.value.category == "input"


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["pre-copy", "post-copy"])
async def test_host_filesystem_oserror_is_normalized_to_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    destination = tmp_path / "payload"

    async def host_ossutil(*arguments: str):
        Path(arguments[4]).write_bytes(b"x")
        return 0, b"", b""

    monkeypatch.setattr("ale_run.atomic.trusted_staging.run_host_ossutil", host_ossutil)
    if phase == "pre-copy":
        original_mkdir = Path.mkdir

        def denied_mkdir(self, *args, **kwargs):
            if self == destination:
                raise PermissionError("destination mkdir denied")
            return original_mkdir(self, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", denied_mkdir)
    else:
        original_chmod = os.chmod

        def denied_chmod(path, mode, *, dir_fd=None, follow_symlinks=True):
            if Path(path).name == "file":
                raise OSError("download chmod failed")
            return original_chmod(path, mode, dir_fd=dir_fd, follow_symlinks=follow_symlinks)

        monkeypatch.setattr("ale_run.atomic.trusted_staging.os.chmod", denied_chmod)

    with pytest.raises(AtomicInfrastructureError) as caught:
        await _download_prefix(
            [
                _OssObject(
                    url="oss://bucket/input/dir/file",
                    key="input/dir/file",
                    relative_path=Path("dir/file"),
                    size_bytes=1,
                )
            ],
            destination,
            actual_total=0,
        )

    assert caught.value.category == "input"
    assert isinstance(caught.value.__cause__, OSError)


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
    assert sandbox.cleanup_commands


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_outcome", ["nonzero", "transport"])
async def test_host_cleanup_errors_do_not_mask_orchestration_failure(
    tmp_path: Path,
    cleanup_outcome: str,
) -> None:
    sandbox = _OrchestrationFailingSandbox(
        tmp_path / "vm",
        "run",
        cleanup_outcome=cleanup_outcome,
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
    assert len(sandbox.cleanup_commands) >= 6


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_outcome", ["nonzero", "transport"])
async def test_host_cleanup_failure_fails_otherwise_successful_staging(
    tmp_path: Path,
    cleanup_outcome: str,
) -> None:
    sandbox = _OrchestrationFailingSandbox(
        tmp_path / "vm",
        "none",
        cleanup_outcome=cleanup_outcome,
    )
    prepared = _write_input_archive(tmp_path, {"input/required.txt": b"trusted"})

    with pytest.raises(AtomicInfrastructureError, match="staging cleanup failed"):
        await stage_atomic_input(
            sandbox,
            _task_data(),
            source="oss://private-bucket/tasks",
            declared_input_paths=("input/required.txt",),
            prepared=prepared,
        )

    assert len(sandbox.cleanup_commands) >= 7


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stdout", "returncode", "run_error"),
    [
        (
            '{"ok":false,"error":"verifier failed","preserve_staging":false}',
            2,
            None,
        ),
        ('{"status":"done"}', 0, None),
        ("not-json", 2, None),
        ("", 0, RuntimeError("injected run after backup creation")),
    ],
    ids=["failed-payload", "arbitrary-dict", "invalid-json", "run-exception"],
)
async def test_untrusted_verifier_result_preserves_and_reports_staging_path(
    tmp_path: Path,
    stdout: str,
    returncode: int,
    run_error: Exception | None,
) -> None:
    sandbox = _VerifierPayloadSandbox(
        tmp_path / "vm",
        stdout=stdout,
        returncode=returncode,
        run_error=run_error,
    )
    prepared = _write_input_archive(tmp_path, {"input/required.txt": b"trusted"})

    with pytest.raises((AtomicInfrastructureError, RuntimeError)) as caught:
        await stage_atomic_input(
            sandbox,
            _task_data(),
            source="oss://private-bucket/tasks",
            declared_input_paths=("input/required.txt",),
            prepared=prepared,
        )

    assert sandbox.staging_path is not None
    assert sandbox.staging_path.is_dir()
    cleanup_commands = "\n".join(sandbox.cleanup_commands)
    assert str(sandbox.staging_path) not in cleanup_commands
    diagnostic = "\n".join([str(caught.value), *getattr(caught.value, "__notes__", [])])
    assert str(sandbox.staging_path) in diagnostic


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"ok": 1},
        {"ok": True, "error": "conflicting success"},
        {"ok": True, "preserve_staging": False},
        {"ok": True, "preserve_staging": True},
    ],
)
async def test_untrusted_payload_cannot_spoof_terminal_success(
    tmp_path: Path,
    payload: dict[str, object],
) -> None:
    sandbox = _VerifierPayloadSandbox(
        tmp_path / "vm",
        stdout=json.dumps(payload),
    )
    prepared = _write_input_archive(tmp_path, {"input/required.txt": b"trusted"})

    with pytest.raises(AtomicInfrastructureError) as caught:
        await stage_atomic_input(
            sandbox,
            _task_data(),
            source="oss://private-bucket/tasks",
            declared_input_paths=("input/required.txt",),
            prepared=prepared,
        )

    assert sandbox.staging_path is not None
    assert sandbox.staging_path.is_dir()
    assert str(sandbox.staging_path) not in "\n".join(sandbox.cleanup_commands)
    diagnostic = "\n".join([str(caught.value), *getattr(caught.value, "__notes__", [])])
    assert str(sandbox.staging_path) in diagnostic


@pytest.mark.asyncio
async def test_exact_terminal_success_authorizes_host_staging_cleanup(
    tmp_path: Path,
) -> None:
    sandbox = _VerifierPayloadSandbox(tmp_path / "vm", stdout='{"ok":true}')
    prepared = _write_input_archive(tmp_path, {"input/required.txt": b"trusted"})

    await stage_atomic_input(
        sandbox,
        _task_data(),
        source="oss://private-bucket/tasks",
        declared_input_paths=("input/required.txt",),
        prepared=prepared,
    )

    assert sandbox.staging_path is not None
    assert str(sandbox.staging_path) in "\n".join(sandbox.cleanup_commands)


@pytest.mark.asyncio
async def test_windows_checked_cleanup_is_link_aware(
    tmp_path: Path,
) -> None:
    sandbox = _VerifierPayloadSandbox(
        tmp_path / "vm",
        stdout='{"ok":true}',
        is_linux=False,
    )
    prepared = _write_input_archive(tmp_path, {"input/required.txt": b"trusted"})

    await stage_atomic_input(
        sandbox,
        _task_data(),
        source="oss://private-bucket/tasks",
        declared_input_paths=("input/required.txt",),
        prepared=prepared,
    )

    cleanup = "\n".join(sandbox.cleanup_commands)
    assert "powershell" in cleanup.lower()
    assert "LinkType" in cleanup
    assert "Remove-Item" in cleanup


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
