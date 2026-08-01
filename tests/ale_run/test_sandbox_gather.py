import asyncio
from types import SimpleNamespace
import subprocess

import pytest

from ale_run.base_interface import RangeResult, SandboxUnreachableError
from ale_run.executors.sandbox import (
    SandboxExecutor,
    _build_launcher,
    tail_hot_artifacts,
)
from ale_run.orchestration.termination import classify_error


@pytest.mark.asyncio
async def test_gather_reuses_complete_incremental_mirror(tmp_path) -> None:
    destination = tmp_path / "origin_log"
    destination.mkdir()
    mirrored = destination / "telemetry.jsonl"
    mirrored.write_bytes(b'{"done":true}\n')

    class Sandbox:
        is_linux = False

        async def list_dir(self, _src):
            return [
                {
                    "relpath": "telemetry.jsonl",
                    "is_dir": False,
                    "size": mirrored.stat().st_size,
                }
            ]

        async def download_to_local(self, *_args, **_kwargs):
            raise AssertionError("complete incremental mirror was downloaded again")

    executor = SandboxExecutor(
        config=SimpleNamespace(),
        work_dir=r"C:\work",
        sandbox=Sandbox(),
        env={},
    )
    executor.hot_artifacts = ("otel_requests.jsonl",)

    report = await executor.gather_dir(src=r"C:\work", dst=destination)

    assert report.error is None
    assert report.files == 1
    assert report.bytes == mirrored.stat().st_size


@pytest.mark.asyncio
async def test_gather_marks_reused_incomplete_hot_mirror_as_partial(tmp_path) -> None:
    destination = tmp_path / "origin_log"
    destination.mkdir()
    mirrored = destination / "otel_requests.jsonl"
    mirrored.write_bytes(b'partial-but-useful\n')

    class Sandbox:
        is_linux = False

        async def list_dir(self, _src):
            return [
                {
                    "relpath": "otel_requests.jsonl",
                    "is_dir": False,
                    "size": mirrored.stat().st_size + 50_000_000,
                }
            ]

        async def download_to_local(self, *_args, **_kwargs):
            raise AssertionError("hot-tail mirror must not be bulk-downloaded")

    executor = SandboxExecutor(
        config=SimpleNamespace(),
        work_dir=r"C:\work",
        sandbox=Sandbox(),
        env={},
    )
    executor.hot_artifacts = ("otel_requests.jsonl",)

    report = await executor.gather_dir(src=r"C:\work", dst=destination)

    assert report.error is None
    assert report.files == 1
    assert report.bytes == mirrored.stat().st_size
    assert report.warnings == [
        "partial hot artifact otel_requests.jsonl: "
        f"host={mirrored.stat().st_size} remote={mirrored.stat().st_size + 50_000_000}"
    ]


@pytest.mark.asyncio
async def test_gather_deadline_bounds_inflight_download(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("ale_run.executors.sandbox._GATHER_DEADLINE_S", 0.01)

    class Sandbox:
        is_linux = False

        async def list_dir(self, _src):
            return [{"relpath": "slow.log", "is_dir": False, "size": 100}]

        async def download_to_local(self, *_args, **_kwargs):
            await asyncio.sleep(10)
            return False

    executor = SandboxExecutor(
        config=SimpleNamespace(),
        work_dir=r"C:\work",
        sandbox=Sandbox(),
        env={},
    )

    report = await asyncio.wait_for(
        executor.gather_dir(src=r"C:\work", dst=tmp_path / "origin_log"),
        timeout=0.2,
    )

    assert report.error == "gather capped (deadline_0s) after 0 files"


@pytest.mark.asyncio
async def test_gather_deadline_bounds_directory_listing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("ale_run.executors.sandbox._GATHER_DEADLINE_S", 0.01)

    class Sandbox:
        is_linux = False

        async def list_dir(self, _src):
            await asyncio.sleep(10)
            return []

    executor = SandboxExecutor(
        config=SimpleNamespace(),
        work_dir=r"C:\work",
        sandbox=Sandbox(),
        env={},
    )

    report = await asyncio.wait_for(
        executor.gather_dir(src=r"C:\work", dst=tmp_path / "origin_log"),
        timeout=0.2,
    )

    assert report.error == "gather capped (deadline_0s) before directory listing"


@pytest.mark.asyncio
async def test_gather_never_pulls_media_from_agent_workdir(tmp_path) -> None:
    downloaded: list[str] = []

    class Sandbox:
        is_linux = False

        async def list_dir(self, _src):
            return [
                {"relpath": "transcript.jsonl", "is_dir": False, "size": 10},
                {"relpath": "final_result.png", "is_dir": False, "size": 3_000_000},
                {"relpath": "final_result.psd", "is_dir": False, "size": 50_000_000},
            ]

        async def download_to_local(self, remote, local, *, timeout):
            downloaded.append(remote)
            from pathlib import Path

            Path(local).write_bytes(b"x" * 10)
            return True

    executor = SandboxExecutor(
        config=SimpleNamespace(),
        work_dir=r"C:\work",
        sandbox=Sandbox(),
        env={},
    )

    report = await executor.gather_dir(src=r"C:\work", dst=tmp_path / "origin_log")

    assert report.files == 1
    assert downloaded == [r"C:\work\transcript.jsonl"]
    assert not (tmp_path / "origin_log" / "final_result.png").exists()


@pytest.mark.asyncio
async def test_pid_ack_uses_scalar_command_and_recovers_transport_failure(
    monkeypatch,
) -> None:
    class Sandbox:
        is_linux = False

        def __init__(self) -> None:
            self.calls = 0

        async def run_command(self, command, *, timeout):
            assert "Get-Content" in command
            assert timeout <= 15
            self.calls += 1
            if self.calls == 1:
                return subprocess.CompletedProcess([], -1, "", "transport timeout")
            return subprocess.CompletedProcess([], 0, "__ALE_PID__=1552\n", "")

        async def read_text(self, _path):
            raise AssertionError("PID acknowledgement must not download a remote file")

    async def no_sleep(_seconds):
        return None

    sandbox = Sandbox()
    executor = SandboxExecutor(
        config=SimpleNamespace(),
        work_dir=r"C:\work",
        sandbox=sandbox,
        env={},
    )
    monkeypatch.setattr("ale_run.executors.sandbox.asyncio.sleep", no_sleep)

    assert await executor._read_pid(r"C:\work\_pid") == 1552
    assert sandbox.calls == 2


def test_windows_launcher_emits_pid_ack_for_new_and_existing_processes() -> None:
    sandbox = SimpleNamespace(is_linux=False)
    launcher = _build_launcher(
        sandbox=sandbox,
        python=r"C:\Python312\python.exe",
        ale_src_root=r"C:\ale-src",
        spec_path=r"C:\work\_spec.json",
        pid_file=r"C:\work\_pid",
        entry_log=r"C:\work\_entry.log",
    )

    assert launcher.count("__ALE_PID__=") >= 2
    assert "New-Item -ItemType Directory" in launcher
    assert "Remove-Item -LiteralPath $lockDir" in launcher


def test_windows_launcher_quotes_spaced_spec_path_as_one_argument() -> None:
    launcher = _build_launcher(
        sandbox=SimpleNamespace(is_linux=False),
        python=r"C:\Program Files\Python312\python.exe",
        ale_src_root=r"C:\ALE Source",
        spec_path=r"C:\Users\Test User\run\_spec.json",
        pid_file=r"C:\Users\Test User\run\_pid",
        entry_log=r"C:\Users\Test User\run\_entry.log",
    )

    assert "-ArgumentList '" in launcher
    assert '\"C:\\Users\\Test User\\run\\_spec.json\"' in launcher


@pytest.mark.asyncio
async def test_pid_probe_classifies_persistent_transport_as_infrastructure(
    monkeypatch,
) -> None:
    class Sandbox:
        is_linux = False

        async def run_command(self, _command, *, timeout):
            return subprocess.CompletedProcess([], -1, "", "transport timeout")

    executor = SandboxExecutor(
        config=SimpleNamespace(),
        work_dir=r"C:\work",
        sandbox=Sandbox(),
        env={},
    )
    monkeypatch.setattr("ale_run.executors.sandbox._PID_WAIT_S", 0.01)

    with pytest.raises(SandboxUnreachableError, match="PID acknowledgement"):
        await executor._read_pid(r"C:\work\_pid")


@pytest.mark.asyncio
async def test_pid_probe_does_not_swallow_programming_errors(monkeypatch) -> None:
    class Sandbox:
        is_linux = False

        async def run_command(self, _command, *, timeout):
            raise TypeError("broken sandbox implementation")

    executor = SandboxExecutor(
        config=SimpleNamespace(),
        work_dir=r"C:\work",
        sandbox=Sandbox(),
        env={},
    )
    monkeypatch.setattr("ale_run.executors.sandbox._PID_WAIT_S", 0.01)

    with pytest.raises(TypeError, match="broken sandbox implementation"):
        await executor._read_pid(r"C:\work\_pid")


def test_launch_ack_unavailable_is_classified_as_transport_error() -> None:
    error = RuntimeError(
        "infrastructure launch acknowledgement unavailable: "
        "PID acknowledgement transport failed"
    )

    assert classify_error(error) == "transport_error"


@pytest.mark.asyncio
async def test_tail_reconcile_bounds_an_inflight_range_probe(tmp_path, monkeypatch) -> None:
    class Executor:
        async def download_range(self, *, src, start, max_bytes, timeout_s=None):
            assert timeout_s is not None
            await asyncio.sleep(10)

    stop = asyncio.Event()
    stop.set()
    monkeypatch.setattr("ale_run.executors.sandbox._TAIL_RECONCILE_TIMEOUT_S", 0.01)

    error = await asyncio.wait_for(
        tail_hot_artifacts(
            executor=Executor(),
            targets=[("remote.jsonl", tmp_path / "local.jsonl")],
            stop_event=stop,
        ),
        timeout=0.2,
    )

    assert "reconcile timeout" in error


@pytest.mark.asyncio
async def test_tail_reconcile_reports_stable_backlog_as_partial(tmp_path, monkeypatch) -> None:
    remote_size = 3 * 16 * 1024 * 1024

    class Executor:
        async def download_range(self, *, src, start, max_bytes, timeout_s=None):
            return RangeResult(
                success=True,
                new_data=b"x\n" * (max_bytes // 2),
                new_size=remote_size,
            )

    stop = asyncio.Event()
    stop.set()
    monkeypatch.setattr("ale_run.executors.sandbox._TAIL_RECONCILE_RETRIES", 1)
    monkeypatch.setattr("ale_run.executors.sandbox._TAIL_RECONCILE_DELAY_S", 0)

    error = await tail_hot_artifacts(
        executor=Executor(),
        targets=[("remote.jsonl", tmp_path / "local.jsonl")],
        stop_event=stop,
    )

    assert "reconcile incomplete" in error
    assert "33554432/50331648 bytes" in error


@pytest.mark.asyncio
async def test_tail_reconcile_reports_uncommitted_trailing_fragment(tmp_path, monkeypatch) -> None:
    class Executor:
        async def download_range(self, *, src, start, max_bytes, timeout_s=None):
            return RangeResult(success=True, new_data=b"partial", new_size=7)

    stop = asyncio.Event()
    stop.set()
    monkeypatch.setattr("ale_run.executors.sandbox._TAIL_RECONCILE_RETRIES", 1)
    monkeypatch.setattr("ale_run.executors.sandbox._TAIL_RECONCILE_DELAY_S", 0)

    error = await tail_hot_artifacts(
        executor=Executor(),
        targets=[("remote.jsonl", tmp_path / "local.jsonl")],
        stop_event=stop,
    )

    assert "reconcile incomplete" in error
    assert "0/7 bytes" in error
