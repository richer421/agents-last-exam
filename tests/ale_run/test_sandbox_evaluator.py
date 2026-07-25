from __future__ import annotations

import json
import tarfile
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from ale_run.executors.sandbox_evaluator import (
    SandboxEvaluationError,
    SandboxEvaluationResult,
    _build_task_archive,
    evaluate_in_sandbox,
)
from ale_run.executors._sandbox_eval_entry import _start_remote_session, _worker_argv
from ale_run.orchestration.lifecycle import _evaluate_task


class _FakeSandbox:
    is_linux = False
    python = r"C:\Program Files\Python312\python.exe"
    work_dir_base = r"C:\Users\User\.ale"
    task_data_root = r"E:\agenthle"
    cua_server_port = 5000

    def __init__(self) -> None:
        self.writes: list[tuple[str, bytes]] = []
        self.commands: list[tuple[str, float | None]] = []
        self._polls = 0
        self.result_payload = {
            "ok": True,
            "result": {"score": 0.75, "raw_scores": [0.75]},
        }
        self.eval_log = b""

    async def mkdir(self, path: str) -> None:
        self.mkdir_path = path

    async def write_file(self, path: str, data: bytes) -> None:
        self.writes.append((path, data))

    async def run_command(self, command: str, timeout: float | None = None):
        self.commands.append((command, timeout))
        if "Start-Process" in command:
            return SimpleNamespace(
                returncode=0, stdout="__ALE_EVAL_PID__=4242\n", stderr=""
            )
        if "Get-Item" in command and ".Length" in command:
            return SimpleNamespace(returncode=0, stdout="128\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    async def exists(self, path: str) -> bool:
        self._polls += 1
        return self._polls >= 2

    async def read_file(self, path: str) -> bytes:
        if path.endswith("eval.log"):
            return self.eval_log
        assert path.endswith("result.json")
        return json.dumps(self.result_payload).encode()


def test_task_archive_is_deterministic_and_excludes_generated_data(tmp_path: Path) -> None:
    repo = tmp_path / "task-repo"
    (repo / "tasks" / "demo").mkdir(parents=True)
    (repo / "tasks" / "demo" / "main.py").write_text("VALUE = 1\n")
    (repo / "tasks" / "demo" / "auth.py").write_text("ENABLED = True\n")
    (repo / "tasks" / "demo" / "prompt.jinja").write_text("Hello {{ name }}\n")
    (repo / "tasks" / "demo" / "labels.csv").write_text("id,label\n1,demo\n")
    (repo / "pyproject.toml").write_text(
        "[project]\nname='demo'\nversion='0'\n"
        "[tool.ale.evaluator_archive]\n"
        "include=['tasks/demo/*.jinja', 'tasks/demo/*.csv']\n"
    )
    (repo / "artifacts").mkdir()
    (repo / "artifacts" / "huge.psd").write_bytes(b"x" * 1024)
    (repo / ".env").write_text("TOKEN=secret\n")
    (repo / "reference.png").write_bytes(b"not-code")
    (repo / "secrets.json").write_text('{"token":"secret"}\n')
    (repo / "reference.json").write_text('{"private":"data"}\n')
    (repo / "tasks" / "demo" / "__pycache__").mkdir()
    (repo / "tasks" / "demo" / "__pycache__" / "main.pyc").write_bytes(b"cache")

    first = _build_task_archive(repo)
    second = _build_task_archive(repo)

    assert first.payload == second.payload
    assert first.digest == second.digest
    assert first.files == 5
    assert b"huge.psd" not in first.payload
    with tarfile.open(fileobj=io.BytesIO(first.payload), mode="r:gz") as archive:
        assert "tasks/demo/prompt.jinja" in archive.getnames()
        assert "tasks/demo/labels.csv" in archive.getnames()
        assert "tasks/demo/auth.py" in archive.getnames()
        assert "secrets.json" not in archive.getnames()
        assert "reference.json" not in archive.getnames()


def test_task_archive_rejects_disguised_media_in_explicit_package_data(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "task-repo"
    (repo / "tasks" / "demo").mkdir(parents=True)
    (repo / "tasks" / "demo" / "main.py").write_text("VALUE = 1\n")
    (repo / "tasks" / "demo" / "labels.csv").write_bytes(
        b"\x89PNG\r\n\x1a\n" + b"not-really-csv"
    )
    (repo / "pyproject.toml").write_text(
        "[project]\nname='demo'\nversion='0'\n"
        "[tool.ale.evaluator_archive]\ninclude=['tasks/demo/*.csv']\n"
    )

    with pytest.raises(ValueError, match="media content"):
        _build_task_archive(repo)


def test_task_archive_rejects_additional_disguised_media_magic(tmp_path: Path) -> None:
    repo = tmp_path / "task-repo"
    (repo / "tasks" / "demo").mkdir(parents=True)
    (repo / "tasks" / "demo" / "fixture.csv").write_bytes(b"BM" + b"bitmap")
    (repo / "pyproject.toml").write_text(
        "[project]\nname='demo'\nversion='0'\n"
        "[tool.ale.evaluator_archive]\ninclude=['tasks/demo/*.csv']\n"
    )

    with pytest.raises(ValueError, match="media content"):
        _build_task_archive(repo)


def test_task_archive_rejects_recursive_extra_globs(tmp_path: Path) -> None:
    repo = tmp_path / "task-repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text(
        "[project]\nname='demo'\nversion='0'\n"
        "[tool.ale.evaluator_archive]\ninclude=['tasks/**']\n"
    )

    with pytest.raises(ValueError, match="unsafe evaluator archive include pattern"):
        _build_task_archive(repo)


@pytest.mark.asyncio
async def test_evaluate_in_sandbox_only_reads_small_result_protocol(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "task-repo"
    task = repo / "tasks" / "demo"
    task.mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0'\n")
    (task / "main.py").write_text("VALUE = 1\n")
    sandbox = _FakeSandbox()
    monkeypatch.setattr("ale_run.executors.sandbox_evaluator.asyncio.sleep", _no_sleep)

    result = await evaluate_in_sandbox(
        sandbox=sandbox,
        ale_src_root=r"C:\Users\User\.ale\src",
        task_path=task,
        variant=0,
        timeout_s=30,
    )

    assert result == SandboxEvaluationResult(
        result={"score": 0.75, "raw_scores": [0.75]},
        log="",
    )
    assert len(sandbox.writes) == 2
    assert any(path.endswith("tasks.tar.gz") for path, _ in sandbox.writes)
    assert any(path.endswith("spec.json") for path, _ in sandbox.writes)
    assert len(sandbox.commands) >= 3
    assert "Start-Process" in sandbox.commands[0][0]


async def _no_sleep(_: float) -> None:
    return None


@pytest.mark.asyncio
async def test_evaluator_timeout_kills_remote_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "task-repo"
    task = repo / "tasks" / "demo"
    task.mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0'\n")
    (task / "main.py").write_text("VALUE = 1\n")
    sandbox = _FakeSandbox()
    sandbox.exists = _always_missing
    monkeypatch.setattr("ale_run.executors.sandbox_evaluator.asyncio.sleep", _no_sleep)

    with pytest.raises(TimeoutError):
        await evaluate_in_sandbox(
            sandbox=sandbox,
            ale_src_root=r"C:\Users\User\.ale\src",
            task_path=task,
            variant=0,
            timeout_s=0,
        )

    assert any("taskkill /PID 4242 /T /F" in command for command, _ in sandbox.commands)


async def _always_missing(_: str) -> bool:
    return False


@pytest.mark.asyncio
async def test_remote_session_is_started_before_task_driver() -> None:
    class _Session:
        def __init__(self) -> None:
            self.started = False

        async def start(self, *, headless: bool) -> None:
            assert headless is True
            self.started = True

    session = _Session()
    await _start_remote_session(session)
    assert session.started is True


@pytest.mark.asyncio
async def test_failed_remote_evaluator_preserves_bounded_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "task-repo"
    task = repo / "tasks" / "demo"
    task.mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0'\n")
    (task / "main.py").write_text("VALUE = 1\n")
    sandbox = _FakeSandbox()
    sandbox.result_payload = {"ok": False, "error": "worker failed"}
    sandbox.eval_log = b"actual remote traceback"
    monkeypatch.setattr("ale_run.executors.sandbox_evaluator.asyncio.sleep", _no_sleep)

    with pytest.raises(SandboxEvaluationError) as caught:
        await evaluate_in_sandbox(
            sandbox=sandbox,
            ale_src_root=r"C:\Users\User\.ale\src",
            task_path=task,
            variant=0,
            timeout_s=30,
        )

    assert caught.value.evaluator_log == "actual remote traceback"
    assert "worker failed" in str(caught.value)


def test_worker_reexec_uses_importable_module_name() -> None:
    assert _worker_argv("python.exe", "spec.json") == [
        "python.exe",
        "-m",
        "ale_run.executors._sandbox_eval_entry",
        "spec.json",
    ]


@pytest.mark.asyncio
async def test_lifecycle_uses_near_data_evaluator_for_sandbox_executor(
    tmp_path: Path,
) -> None:
    class _RemoteExecutor:
        async def evaluate_task(self, *, task_path: Path, variant: int, timeout_s: float):
            self.called = (task_path, variant, timeout_s)
            return {"score": 0.9}

    class _HostDriver:
        async def evaluate(self):
            raise AssertionError("host evaluator must not read sandbox artifacts")

    executor = _RemoteExecutor()
    task_path = tmp_path / "tasks" / "demo"
    result = await _evaluate_task(
        task_driver=_HostDriver(),
        executor=executor,
        task_path=task_path,
        variant=2,
        timeout_s=45,
    )

    assert result == {"score": 0.9}
    assert executor.called == (task_path, 2, 45)
