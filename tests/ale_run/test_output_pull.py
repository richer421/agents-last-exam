from pathlib import Path

import pytest

from ale_run.base_interface import SandboxHandle, TaskDataSpec
from ale_run.environments.output_pull import pull_to_host, push_to_oss


@pytest.mark.asyncio
async def test_windows_output_pull_accepts_absolute_list_dir_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = (
        r"E:\agenthle\domain\task\variant\output"
    )
    sandbox = SandboxHandle(
        id="windows-output-test",
        endpoint="http://127.0.0.1:15000",
        os="windows",
        work_dir_base="",
        task_data_root=r"E:\agenthle",
        node="",
        python="",
        mcp_server_dir="",
        metadata={},
    )
    downloaded: list[str] = []

    async def fake_list_dir(path: str):
        assert path == output_root
        return [
            {
                "relpath": output_root + r"\final_result.psd",
                "is_dir": False,
            }
        ]

    async def fake_download(remote_path: str, local_path: str, *, timeout: float = 120):
        downloaded.append(remote_path)
        Path(local_path).write_bytes(b"8BPS")
        return True

    monkeypatch.setattr(sandbox, "list_dir", fake_list_dir)
    monkeypatch.setattr(sandbox, "download_to_local", fake_download)

    destination = tmp_path / "output"
    report = await pull_to_host(
        sandbox,
        TaskDataSpec(
            domain_name="domain",
            task_name="task",
            variant_name="variant",
        ),
        dest_dir=destination,
    )

    assert downloaded == [output_root + r"\final_result.psd"]
    assert (destination / "final_result.psd").read_bytes() == b"8BPS"
    assert report["files"] == 1


@pytest.mark.asyncio
async def test_windows_oss_output_uses_shared_runtime_and_task_run_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = SandboxHandle(
        id="windows-oss-test",
        endpoint="http://127.0.0.1:15000",
        os="windows",
        work_dir_base="",
        task_data_root=r"E:\agenthle",
        node="",
        python="",
        mcp_server_dir="",
        metadata={},
    )
    commands: list[str] = []
    ensured: list[str] = []
    written: list[tuple[str, bytes]] = []

    async def fake_ensure(target: SandboxHandle) -> None:
        ensured.append(target.id)

    def fake_command(target: SandboxHandle, arguments: str) -> str:
        assert target is sandbox
        return f"SHARED_OSS_RUNTIME {arguments}"

    async def fake_run(command: str, *, timeout: float = 60):
        commands.append(command)
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    async def fake_write(path: str, content: bytes) -> None:
        written.append((path, content))

    monkeypatch.setattr(
        "ale_run.environments.task_data.ossbucket.ensure_ossutil",
        fake_ensure,
    )
    monkeypatch.setattr(
        "ale_run.environments.task_data.ossbucket.oss_command",
        fake_command,
    )
    monkeypatch.setattr(sandbox, "run_command", fake_run)
    monkeypatch.setattr(sandbox, "write_file", fake_write)

    report = await push_to_oss(
        sandbox,
        TaskDataSpec(
            domain_name="平面设计",
            task_name="task-id",
            variant_name="variant-id",
        ),
        run_id="run-123",
        bucket="oss://ale-artifacts",
    )

    assert ensured == ["windows-oss-test"]
    assert len(written) == 1
    assert written[0][0] == r"C:\Windows\Temp\ale-output-manifest.py"
    manifest_script = written[0][1].decode("utf-8")
    assert "平面设计" in manifest_script
    assert "path.is_symlink()" in manifest_script
    assert 'manifest_path.with_suffix(".json.tmp")' in manifest_script
    assert "temporary_path.replace(manifest_path)" in manifest_script
    assert commands[0] == (
        r'"python" "C:\Windows\Temp\ale-output-manifest.py"'
    )
    assert commands[1].startswith("SHARED_OSS_RUNTIME cp -r -f ")
    assert (
        "oss://ale-artifacts/平面设计/task-id/variant-id/"
        "runs/run-123/output/"
    ) in commands[1]
    assert report["oss_path"].endswith(
        "/平面设计/task-id/variant-id/runs/run-123/output/"
    )
    assert report["manifest"] == "output/artifact_manifest.json"


@pytest.mark.asyncio
async def test_windows_oss_output_escapes_apostrophes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = SandboxHandle(
        id="windows-oss-quote-test",
        endpoint="http://127.0.0.1:15000",
        os="windows",
        work_dir_base="",
        task_data_root=r"E:\agenthle",
        node="",
        python="",
        mcp_server_dir="",
        metadata={},
    )
    commands: list[str] = []

    async def fake_run(command: str, *, timeout: float = 60):
        commands.append(command)
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    async def fake_noop(*args, **kwargs) -> None:
        return None

    monkeypatch.setattr(
        "ale_run.environments.task_data.ossbucket.ensure_ossutil",
        fake_noop,
    )
    monkeypatch.setattr(
        "ale_run.environments.output_pull._write_output_manifest",
        fake_noop,
    )
    monkeypatch.setattr(sandbox, "run_command", fake_run)

    await push_to_oss(
        sandbox,
        TaskDataSpec(
            domain_name="O'Brien中文",
            task_name="task",
            variant_name="variant",
        ),
        run_id="run",
        bucket="oss://bucket",
    )

    assert "O''Brien中文" in commands[-1]
