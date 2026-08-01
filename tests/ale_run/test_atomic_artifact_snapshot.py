from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from ale_run.atomic.contracts import AtomicInfrastructureError, SolveRequest
from ale_run.atomic.oss_submission import download_range_to_local, publish_submission
from ale_run.base_interface import RangeResult, TaskDataSpec


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
        raise AssertionError("atomic publication must not use download_to_local")

    async def download_range(
        self,
        remote_path: str,
        *,
        start: int,
        max_chunk_bytes: int,
        timeout: float = 60,
    ) -> RangeResult:
        del timeout
        source = Path(remote_path)
        initial = source.read_bytes()
        chunk = initial[start : start + max_chunk_bytes]
        if start + len(chunk) == len(initial):
            source.write_bytes(b"mutated after trusted pull")
        return RangeResult(
            success=True,
            new_data=chunk,
            new_size=len(initial),
        )


class _RangeSandbox:
    def __init__(self, results: list[RangeResult]) -> None:
        self.results = iter(results)
        self.calls: list[tuple[str, int, int, float]] = []

    async def download_range(
        self,
        remote_path: str,
        *,
        start: int,
        max_chunk_bytes: int,
        timeout: float = 60,
    ) -> RangeResult:
        self.calls.append((remote_path, start, max_chunk_bytes, timeout))
        return next(self.results)

    async def download_to_local(self, *_args: object, **_kwargs: object) -> bool:
        raise AssertionError("atomic download must not use download_to_local")


@pytest.mark.asyncio
async def test_download_range_to_local_streams_multiple_chunks_directly_to_disk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "private" / "artifact.bin"
    sandbox = _RangeSandbox(
        [
            RangeResult(success=True, new_data=b"abcd", new_size=10),
            RangeResult(success=True, new_data=b"efgh", new_size=10),
            RangeResult(success=True, new_data=b"ij", new_size=10),
        ]
    )
    monkeypatch.setattr("ale_run.atomic.oss_submission._DOWNLOAD_CHUNK_BYTES", 4)

    await download_range_to_local(
        sandbox,
        "/remote/artifact.bin",
        destination,
        max_bytes=10,
        timeout=17,
    )

    assert destination.read_bytes() == b"abcdefghij"
    assert [call[1:3] for call in sandbox.calls] == [(0, 4), (4, 4), (8, 2)]
    assert destination.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_download_range_to_local_accepts_file_smaller_than_bound(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact.bin"
    sandbox = _RangeSandbox([RangeResult(success=True, new_data=b"abc", new_size=3)])

    await download_range_to_local(
        sandbox,
        "/remote/artifact.bin",
        destination,
        max_bytes=10,
        timeout=17,
    )

    assert destination.read_bytes() == b"abc"


@pytest.mark.asyncio
async def test_download_range_to_local_enforces_whole_operation_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifact.bin"
    sandbox = _RangeSandbox(
        [RangeResult(success=True, new_data=b"x", new_size=10) for _ in range(10)]
    )
    clock = iter([100.0, 100.0, 100.5, 101.0, 101.5, 102.0])
    monkeypatch.setattr(
        "ale_run.atomic.oss_submission._monotonic",
        lambda: next(clock),
        raising=False,
    )
    monkeypatch.setattr("ale_run.atomic.oss_submission._DOWNLOAD_CHUNK_BYTES", 4)

    with pytest.raises(AtomicInfrastructureError, match="deadline") as caught:
        await download_range_to_local(
            sandbox,
            "/remote/artifact.bin",
            destination,
            max_bytes=10,
            timeout=2,
        )

    assert caught.value.category == "submission_integrity"
    assert [call[3] for call in sandbox.calls] == [2, 1]
    assert not destination.exists()


@pytest.mark.asyncio
async def test_download_range_to_local_enforces_deadline_during_range_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifact.bin"
    sandbox = _RangeSandbox([RangeResult(success=True, new_data=b"a", new_size=2)])
    original_download_range = sandbox.download_range
    blocked = asyncio.Event()

    async def first_then_block(
        remote_path: str,
        *,
        start: int,
        max_chunk_bytes: int,
        timeout: float = 60,
    ) -> RangeResult:
        if start == 0:
            return await original_download_range(
                remote_path,
                start=start,
                max_chunk_bytes=max_chunk_bytes,
                timeout=timeout,
            )
        sandbox.calls.append((remote_path, start, max_chunk_bytes, timeout))
        await blocked.wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(sandbox, "download_range", first_then_block)
    monkeypatch.setattr("ale_run.atomic.oss_submission._DOWNLOAD_CHUNK_BYTES", 1)

    with pytest.raises(AtomicInfrastructureError, match="deadline"):
        await asyncio.wait_for(
            download_range_to_local(
                sandbox,
                "/remote/artifact.bin",
                destination,
                max_bytes=2,
                timeout=0.02,
            ),
            timeout=0.2,
        )

    assert len(sandbox.calls) == 2
    assert not destination.exists()


@pytest.mark.asyncio
async def test_download_range_to_local_rejects_final_chunk_returned_after_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifact.bin"
    sandbox = _RangeSandbox([RangeResult(success=True, new_data=b"a", new_size=1)])
    clock = iter([100.0, 100.0, 103.0])
    monkeypatch.setattr(
        "ale_run.atomic.oss_submission._monotonic",
        lambda: next(clock),
    )

    with pytest.raises(AtomicInfrastructureError, match="deadline"):
        await download_range_to_local(
            sandbox,
            "/remote/artifact.bin",
            destination,
            max_bytes=1,
            timeout=2,
        )

    assert len(sandbox.calls) == 1
    assert not destination.exists()


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
@pytest.mark.asyncio
async def test_download_range_to_local_rejects_nonpositive_or_nonfinite_timeout(
    timeout: float,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact.bin"
    sandbox = _RangeSandbox([RangeResult(success=True, new_data=b"", new_size=0)])

    with pytest.raises(ValueError, match="timeout"):
        await download_range_to_local(
            sandbox,
            "/remote/artifact.bin",
            destination,
            max_bytes=0,
            timeout=timeout,
        )

    assert sandbox.calls == []
    assert not destination.exists()


@pytest.mark.asyncio
async def test_download_range_to_local_rejects_stream_overrun(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact.bin"
    sandbox = _RangeSandbox([RangeResult(success=True, new_data=b"too much", new_size=7)])

    with pytest.raises(RuntimeError, match="exceeds"):
        await download_range_to_local(
            sandbox,
            "/remote/artifact.bin",
            destination,
            max_bytes=4,
            timeout=17,
        )

    assert not destination.exists()


@pytest.mark.asyncio
async def test_download_range_to_local_rejects_chunk_larger_than_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifact.bin"
    sandbox = _RangeSandbox([RangeResult(success=True, new_data=b"abcde", new_size=5)])
    monkeypatch.setattr("ale_run.atomic.oss_submission._DOWNLOAD_CHUNK_BYTES", 4)

    with pytest.raises(RuntimeError, match="requested range"):
        await download_range_to_local(
            sandbox,
            "/remote/artifact.bin",
            destination,
            max_bytes=10,
            timeout=17,
        )

    assert not destination.exists()


@pytest.mark.asyncio
async def test_download_range_to_local_rejects_bytes_beyond_reported_total(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifact.bin"
    sandbox = _RangeSandbox([RangeResult(success=True, new_data=b"abcd", new_size=3)])
    monkeypatch.setattr("ale_run.atomic.oss_submission._DOWNLOAD_CHUNK_BYTES", 4)

    with pytest.raises(RuntimeError, match="reported total"):
        await download_range_to_local(
            sandbox,
            "/remote/artifact.bin",
            destination,
            max_bytes=10,
            timeout=17,
        )

    assert not destination.exists()


@pytest.mark.asyncio
async def test_download_range_to_local_rejects_truncated_stream(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact.bin"
    sandbox = _RangeSandbox(
        [
            RangeResult(success=True, new_data=b"ab", new_size=4),
            RangeResult(success=True, new_data=b"", new_size=4),
        ]
    )

    with pytest.raises(RuntimeError, match="empty before"):
        await download_range_to_local(
            sandbox,
            "/remote/artifact.bin",
            destination,
            max_bytes=4,
            timeout=17,
        )

    assert not destination.exists()


@pytest.mark.asyncio
async def test_download_range_to_local_rejects_remote_size_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifact.bin"
    sandbox = _RangeSandbox(
        [
            RangeResult(success=True, new_data=b"a", new_size=3),
            RangeResult(success=True, new_data=b"b", new_size=2),
        ]
    )
    monkeypatch.setattr("ale_run.atomic.oss_submission._DOWNLOAD_CHUNK_BYTES", 1)

    with pytest.raises(RuntimeError, match="size changed"):
        await download_range_to_local(
            sandbox,
            "/remote/artifact.bin",
            destination,
            max_bytes=3,
            timeout=17,
        )

    assert not destination.exists()


@pytest.mark.asyncio
async def test_download_range_to_local_cleans_partial_file_on_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifact.bin"
    sandbox = _RangeSandbox([RangeResult(success=True, new_data=b"a", new_size=2)])
    monkeypatch.setattr("ale_run.atomic.oss_submission._DOWNLOAD_CHUNK_BYTES", 1)

    async def cancel(*_args: object, **_kwargs: object) -> RangeResult:
        if sandbox.calls:
            raise asyncio.CancelledError
        raise AssertionError("unreachable")

    original_download_range = sandbox.download_range

    async def first_then_cancel(
        remote_path: str,
        *,
        start: int,
        max_chunk_bytes: int,
        timeout: float = 60,
    ) -> RangeResult:
        if start == 0:
            return await original_download_range(
                remote_path,
                start=start,
                max_chunk_bytes=max_chunk_bytes,
                timeout=timeout,
            )
        return await cancel()

    monkeypatch.setattr(sandbox, "download_range", first_then_cancel)

    with pytest.raises(asyncio.CancelledError):
        await download_range_to_local(
            sandbox,
            "/remote/artifact.bin",
            destination,
            max_bytes=2,
            timeout=17,
        )

    assert not destination.exists()


@pytest.mark.asyncio
async def test_download_range_to_local_probes_zero_byte_bound_for_growth(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact.bin"
    sandbox = _RangeSandbox([RangeResult(success=True, new_data=b"x", new_size=1)])

    with pytest.raises(RuntimeError, match="exceeds"):
        await download_range_to_local(
            sandbox,
            "/remote/artifact.bin",
            destination,
            max_bytes=0,
            timeout=17,
        )

    assert [call[1:3] for call in sandbox.calls] == [(0, 1)]
    assert not destination.exists()


@pytest.mark.parametrize("existing_kind", ["file", "symlink"])
@pytest.mark.asyncio
async def test_download_range_to_local_rejects_existing_destination(
    existing_kind: str,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact.bin"
    target = tmp_path / "target.bin"
    if existing_kind == "file":
        destination.write_bytes(b"keep")
    else:
        target.write_bytes(b"target")
        destination.symlink_to(target)
    sandbox = _RangeSandbox([RangeResult(success=True, new_data=b"replace", new_size=7)])

    with pytest.raises(FileExistsError):
        await download_range_to_local(
            sandbox,
            "/remote/artifact.bin",
            destination,
            max_bytes=7,
            timeout=17,
        )

    assert sandbox.calls == []
    if existing_kind == "file":
        assert destination.read_bytes() == b"keep"
    else:
        assert destination.is_symlink()
        assert target.read_bytes() == b"target"


@pytest.mark.asyncio
async def test_download_range_to_local_accepts_positive_short_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "artifact.bin"
    sandbox = _RangeSandbox(
        [
            RangeResult(success=True, new_data=b"ab", new_size=5),
            RangeResult(success=True, new_data=b"cd", new_size=5),
            RangeResult(success=True, new_data=b"e", new_size=5),
        ]
    )
    monkeypatch.setattr("ale_run.atomic.oss_submission._DOWNLOAD_CHUNK_BYTES", 4)

    await download_range_to_local(
        sandbox,
        "/remote/artifact.bin",
        destination,
        max_bytes=5,
        timeout=17,
    )

    assert destination.read_bytes() == b"abcde"
    assert [call[1:3] for call in sandbox.calls] == [(0, 4), (2, 3), (4, 1)]


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
