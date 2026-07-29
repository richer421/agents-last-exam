from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from ale_run.atomic.contracts import SolveRequest
from ale_run.atomic.oss_submission import publish_submission
from ale_run.base_interface import TaskDataSpec


class _MutatingSandbox:
    def __init__(self, output_dir: Path) -> None:
        self.is_linux = True
        self.python = sys.executable
        self.task_data_root = str(output_dir.parent)
        self.output_dir = output_dir
        self.commands: list[str] = []

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

    async def download_to_local(
        self,
        remote_path: str,
        local_path: str,
        *,
        timeout: float = 60,
    ) -> bool:
        shutil.copyfile(remote_path, local_path)
        Path(remote_path).write_bytes(b"mutated after trusted pull")
        return True


@pytest.mark.asyncio
async def test_manifest_hash_and_upload_use_same_immutable_host_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "task-repo"
    task_dir = repo / "tasks" / "toy"
    task_dir.mkdir(parents=True)
    (task_dir / "task_card.json").write_text(
        json.dumps(
            {
                "outputFiles": [
                    {
                        "path": "output/final.txt",
                        "mediaType": "text/plain",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "vm" / "output"
    output_dir.mkdir(parents=True)
    initial = b"immutable snapshot"
    (output_dir / "final.txt").write_bytes(initial)
    request = SolveRequest(
        submission_id=uuid4(),
        runtime_spec_path=tmp_path / "runtime.yaml",
        task_repo=repo,
        task_path="toy",
        variant_index=0,
        agent_id="agent",
        task_commit="a" * 40,
        image_id="m-image-1",
        submission_root="oss://bucket/task",
    )
    task_data = TaskDataSpec(remote_output_dir=str(output_dir))
    objects: dict[str, bytes] = {}
    metadata: dict[str, str] = {}
    host_calls: list[tuple[str, ...]] = []

    async def host_ossutil(*arguments: str):
        host_calls.append(arguments)
        operation = arguments[0]
        if operation == "cp":
            source, destination = arguments[1:3]
            if source.startswith("oss://"):
                if source not in objects:
                    return 1, b"", b"NoSuchKey"
                Path(destination).write_bytes(objects[source])
                return 0, b"", b""
            if "--forbid-overwrite" in arguments and destination in objects:
                return 1, b"", b"ObjectAlreadyExists"
            objects[destination] = Path(source).read_bytes()
            if "--meta" in arguments:
                value = arguments[arguments.index("--meta") + 1]
                metadata[destination] = value.split(":", 1)[1]
            return 0, b"", b""
        url = arguments[1]
        if url not in objects:
            return 1, b"", b"NoSuchKey"
        stdout = f"Content-Length: {len(objects[url])}\n"
        if url in metadata:
            stdout += f"x-oss-meta-sha256: {metadata[url]}\n"
        return 0, stdout.encode(), b""

    monkeypatch.setattr(
        "ale_run.atomic.oss_submission.run_host_ossutil",
        host_ossutil,
        raising=False,
    )
    monkeypatch.setattr(
        "ale_run.atomic.host_oss.run_host_ossutil",
        host_ossutil,
    )

    manifest = await publish_submission(
        _MutatingSandbox(output_dir),
        task_data,
        request,
        provenance={
            "ale_run_id": "run-1",
            "model_id": "model-1",
            "config_digest": "c" * 64,
            "started_at": datetime(2026, 7, 29, tzinfo=UTC),
            "completed_at": datetime(2026, 7, 29, 0, 1, tzinfo=UTC),
        },
    )

    artifact = manifest.artifacts[0]
    artifact_url = (
        f"{request.submission_root}/output/{request.submission_id}/artifacts/output/final.txt"
    )
    assert artifact.sha256 == hashlib.sha256(initial).hexdigest()
    assert artifact.media_type == "text/plain"
    assert objects[artifact_url] == initial
    assert metadata[artifact_url] == artifact.sha256
    assert (output_dir / "final.txt").read_bytes() != objects[artifact_url]
    assert any(call[0] == "stat" and call[1] == artifact_url for call in host_calls)
