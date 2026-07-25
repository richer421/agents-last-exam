from pathlib import Path

import pytest

from ale_run.base_interface import SandboxHandle, TaskDataSpec
from ale_run.environments.output_pull import pull_to_host


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
