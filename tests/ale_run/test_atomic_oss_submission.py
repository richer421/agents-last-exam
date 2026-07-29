from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from ale_run.atomic.contracts import (
    AtomicInfrastructureError,
    EvaluateRequest,
    EvaluationResult,
    HarborProvenance,
    SolveRequest,
)
from ale_run.atomic.oss_submission import (
    _PUBLISH_SCRIPT,
    _download,
    publish_evaluation_result,
    publish_submission,
    stage_submission,
)
from ale_run.base_interface import RangeResult, TaskDataSpec

SUBMISSION_ID = UUID("12345678-1234-5678-1234-567812345678")
TASK_COMMIT = "a" * 40
EVALUATOR_VERSION = "b" * 40


class FakeSandbox:
    def __init__(self, root: Path) -> None:
        self.id = "fake-sandbox"
        self.os = "linux"
        self.is_linux = True
        self.python = sys.executable
        self.task_data_root = str(root / "task-data")
        self.metadata: dict[str, str] = {}
        self.commands: list[str] = []
        self.range_calls: list[tuple[str, int, int, float]] = []

    async def write_file(self, path: str, content: str | bytes) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            destination.write_bytes(content)
        else:
            destination.write_text(content, encoding="utf-8")

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
        self.range_calls.append((remote_path, start, max_chunk_bytes, timeout))
        source = Path(remote_path)
        if not source.is_file():
            return RangeResult(success=False, error="file not found")
        content = await asyncio.to_thread(source.read_bytes)
        return RangeResult(
            success=True,
            new_data=content[start : start + max_chunk_bytes],
            new_size=len(content),
        )

    async def run_command(
        self, command: str, *, timeout: float = 60
    ) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        return await asyncio.to_thread(
            subprocess.run,
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )


class FakeWindowsSandbox:
    def __init__(self, returncodes: list[int]) -> None:
        self.id = "fake-windows-sandbox"
        self.os = "windows"
        self.is_linux = False
        self.python = r"C:\Python\python.exe"
        self.task_data_root = r"E:\ale-data"
        self.metadata: dict[str, str] = {}
        self.commands: list[str] = []
        self.writes: dict[str, bytes] = {}
        self._returncodes = iter(returncodes)

    async def write_file(self, path: str, content: str | bytes) -> None:
        self.writes[path] = content if isinstance(content, bytes) else content.encode()

    async def run_command(
        self, command: str, *, timeout: float = 60
    ) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        return subprocess.CompletedProcess(
            command,
            next(self._returncodes),
            stdout="",
            stderr="NoSuchKey",
        )


def _decode_powershell_command(command: str) -> str:
    assert re.fullmatch(r"powershell -NoProfile -EncodedCommand [A-Za-z0-9+/=]+", command)
    encoded = command.rsplit(" ", 1)[-1]
    return base64.b64decode(encoded).decode("utf-16-le")


@pytest.fixture
def fake_oss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    store = tmp_path / "oss"
    log = tmp_path / "oss-calls.jsonl"
    store.mkdir()
    cli = tmp_path / "fake_ossutil.py"
    cli.write_text(
        """
import json
import os
import shutil
import sys
import time
from pathlib import Path

store = Path(sys.argv[1])
log = Path(sys.argv[2])
args = sys.argv[3:]
op_index = next(i for i, value in enumerate(args) if value in {"cp", "stat"})
op = args[op_index]
op_args = args[op_index + 1:]

def object_path(url):
    assert url.startswith("oss://")
    bucket_and_key = url[len("oss://"):]
    return store / bucket_and_key

def metadata_path(path):
    return path.with_name(path.name + ".metadata.json")

entry = {"op": op, "args": op_args}
if op == "cp":
    src, dst = op_args[:2]
    entry.update({"src": src, "dst": dst})
    if src.startswith("oss://"):
        source = object_path(src)
        if not source.is_file():
            print("NoSuchKey", file=sys.stderr)
            sys.exit(1)
        destination = Path(dst)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        entry["direction"] = "download"
    else:
        source = Path(src)
        if not source.is_file():
            print("NoSuchFile", file=sys.stderr)
            sys.exit(1)
        destination = object_path(dst)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if (
            dst.endswith("/manifest.json")
            and (store.parent / "fail-manifest-upload").is_file()
        ):
            print("InjectedManifestUploadFailure", file=sys.stderr)
            sys.exit(1)
        barrier_flag = store.parent / "concurrent-manifest-barrier"
        if dst.endswith("/manifest.json") and barrier_flag.is_file():
            barrier = store.parent / "manifest-upload-barrier"
            barrier.mkdir(exist_ok=True)
            (barrier / str(os.getpid())).touch()
            deadline = time.monotonic() + 5
            while len(list(barrier.iterdir())) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
        if (
            dst.endswith("/final.txt")
            and (store.parent / "concurrent-artifact-barrier").is_file()
        ):
            barrier = store.parent / "artifact-upload-barrier"
            barrier.mkdir(exist_ok=True)
            (barrier / str(os.getpid())).touch()
            deadline = time.monotonic() + 5
            while len(list(barrier.iterdir())) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
        digest = None
        for i, value in enumerate(op_args[2:]):
            if value in {"--meta", "--metadata"}:
                digest = op_args[i + 3].split(":", 1)[-1].split("=", 1)[-1]
        if (
            digest is not None
            and (store.parent / "corrupt-nested-metadata").is_file()
            and dst.endswith("/nested.bin")
        ):
            digest = "0" * 64
        if "--forbid-overwrite" in op_args:
            lock = destination.with_name(destination.name + ".upload-lock")
            deadline = time.monotonic() + 5
            while True:
                try:
                    lock.mkdir()
                    break
                except FileExistsError:
                    if time.monotonic() >= deadline:
                        print("UploadLockTimeout", file=sys.stderr)
                        sys.exit(1)
                    time.sleep(0.01)
            if destination.exists():
                lock.rmdir()
                print("ObjectAlreadyExists", file=sys.stderr)
                sys.exit(1)
            temporary = destination.with_name(
                destination.name + f".{os.getpid()}.uploading"
            )
            try:
                shutil.copyfile(source, temporary)
                if digest is not None:
                    metadata_path(destination).write_text(
                        json.dumps({"x-oss-meta-sha256": digest}),
                        encoding="utf-8",
                    )
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
                lock.rmdir()
        else:
            shutil.copyfile(source, destination)
            if digest is not None:
                metadata_path(destination).write_text(
                    json.dumps({"x-oss-meta-sha256": digest}), encoding="utf-8"
                )
        entry["direction"] = "upload"
elif op == "stat":
    url = op_args[0]
    source = object_path(url)
    entry["src"] = url
    if not source.is_file():
        print("NoSuchKey", file=sys.stderr)
        sys.exit(1)
    if (store.parent / "oversized-stat-output").is_file():
        sys.stdout.write("x" * (64 * 1024 + 1))
        sys.exit(0)
    metadata = {}
    if metadata_path(source).is_file():
        metadata = json.loads(metadata_path(source).read_text(encoding="utf-8"))
    print(f"Content-Length: {source.stat().st_size}")
    for key, value in metadata.items():
        print(f"{key}: {value}")

with log.open("a", encoding="utf-8") as stream:
    stream.write(json.dumps(entry, sort_keys=True) + "\\n")
""".strip()
        + "\n",
        encoding="utf-8",
    )

    async def fake_ensure(_sandbox: object) -> None:
        return None

    def fake_command(_sandbox: object, arguments: str) -> str:
        return f"{sys.executable} {cli} {store} {log} {arguments}"

    async def fake_host_ossutil(*arguments: str):
        completed = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, str(cli), str(store), str(log), *arguments],
            capture_output=True,
            check=False,
        )
        if len(completed.stdout) > 64 * 1024 or len(completed.stderr) > 64 * 1024:
            raise AtomicInfrastructureError(
                "submission_storage",
                "host ossutil output exceeded 64 KiB",
            )
        return completed.returncode, completed.stdout, completed.stderr

    monkeypatch.setattr(
        "ale_run.atomic.oss_submission.ossbucket.ensure_ossutil",
        fake_ensure,
        raising=False,
    )
    monkeypatch.setattr(
        "ale_run.atomic.oss_submission.ossbucket.oss_command",
        fake_command,
        raising=False,
    )
    monkeypatch.setattr(
        "ale_run.atomic.host_oss.run_host_ossutil",
        fake_host_ossutil,
    )
    monkeypatch.setattr(
        "ale_run.atomic.oss_submission.run_host_ossutil",
        fake_host_ossutil,
    )
    return store, log


@pytest.fixture
def task_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "task-repo"
    task_dir = repo / "tasks" / "toy"
    task_dir.mkdir(parents=True)
    (task_dir / "task_card.json").write_text(
        json.dumps(
            {
                "outputFiles": [
                    {"name": "final.txt", "path": "output/final.txt"},
                    {"name": "nested.bin", "path": "output/nested/nested.bin"},
                ]
            }
        ),
        encoding="utf-8",
    )
    return repo


@pytest.fixture
def solve_request(task_repo: Path, tmp_path: Path) -> SolveRequest:
    return SolveRequest(
        submission_id=SUBMISSION_ID,
        runtime_spec_path=tmp_path / "runtime.yaml",
        task_repo=task_repo,
        task_path="toy",
        variant_index=2,
        agent_id="agent-1",
        task_commit=TASK_COMMIT,
        image_id="m-image-1",
        submission_root="oss://bucket/task-prefix",
    )


@pytest.fixture
def evaluate_request(task_repo: Path, tmp_path: Path) -> EvaluateRequest:
    return EvaluateRequest(
        submission_id=SUBMISSION_ID,
        runtime_spec_path=tmp_path / "runtime.yaml",
        task_repo=task_repo,
        task_path="toy",
        variant_index=2,
        task_commit=TASK_COMMIT,
        image_id="m-image-1",
        submission_root="oss://bucket/task-prefix",
        evaluator_id="evaluator/main",
        evaluator_version=EVALUATOR_VERSION,
    )


@pytest.fixture
def provenance() -> dict[str, object]:
    return {
        "ale_run_id": "run-1",
        "model_id": "model-1",
        "config_digest": "config-sha256",
        "started_at": datetime(2026, 7, 29, 1, 2, 3, tzinfo=UTC),
        "completed_at": datetime(2026, 7, 29, 1, 3, 4, tzinfo=UTC),
    }


def _task_data(output_dir: Path) -> TaskDataSpec:
    return TaskDataSpec(
        requires_task_data=True,
        domain_name="domain",
        task_name="toy",
        variant_name="2",
        remote_output_dir=str(output_dir),
    )


def _object_path(store: Path, url: str) -> Path:
    return store / url.removeprefix("oss://")


def _calls(log: Path) -> list[dict[str, object]]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line]


@pytest.mark.asyncio
async def test_publish_uploads_exact_declared_files_and_commits_manifest_last(
    tmp_path: Path,
    fake_oss: tuple[Path, Path],
    solve_request: SolveRequest,
    provenance: dict[str, object],
) -> None:
    store, log = fake_oss
    output_dir = tmp_path / "vm output 'quoted'"
    (output_dir / "nested").mkdir(parents=True)
    (output_dir / "final.txt").write_text("finished\n", encoding="utf-8")
    nested_bytes = b"\x00nested\xff"
    (output_dir / "nested" / "nested.bin").write_bytes(nested_bytes)
    (output_dir / "undeclared.txt").write_text("do not publish", encoding="utf-8")
    sandbox = FakeSandbox(tmp_path / "vm")

    manifest = await publish_submission(
        sandbox,
        _task_data(output_dir),
        solve_request,
        provenance=provenance,
    )

    assert manifest.submission_id == SUBMISSION_ID
    assert [artifact.path for artifact in manifest.artifacts] == [
        "output/final.txt",
        "output/nested/nested.bin",
    ]
    assert manifest.artifacts[0].size_bytes == len(b"finished\n")
    assert manifest.artifacts[0].sha256 == hashlib.sha256(b"finished\n").hexdigest()
    assert manifest.artifacts[1].size_bytes == len(nested_bytes)
    assert manifest.artifacts[1].sha256 == hashlib.sha256(nested_bytes).hexdigest()

    base = f"oss://bucket/task-prefix/output/{SUBMISSION_ID}"
    assert _object_path(store, f"{base}/artifacts/output/final.txt").read_bytes() == (b"finished\n")
    assert (
        _object_path(store, f"{base}/artifacts/output/nested/nested.bin").read_bytes()
        == nested_bytes
    )
    assert json.loads(
        _object_path(store, f"{base}/artifacts/output/final.txt")
        .with_name("final.txt.metadata.json")
        .read_text(encoding="utf-8")
    ) == {"x-oss-meta-sha256": hashlib.sha256(b"finished\n").hexdigest()}
    assert not _object_path(store, f"{base}/artifacts/output/undeclared.txt").exists()

    writes = [call["dst"] for call in _calls(log) if call.get("direction") == "upload"]
    assert writes[-1] == f"{base}/manifest.json"
    assert writes[:-1] == [
        f"{base}/artifacts/output/final.txt",
        f"{base}/artifacts/output/nested/nested.bin",
    ]


@pytest.mark.asyncio
async def test_publish_missing_required_output_never_commits_manifest(
    tmp_path: Path,
    fake_oss: tuple[Path, Path],
    solve_request: SolveRequest,
    provenance: dict[str, object],
) -> None:
    store, _log = fake_oss
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "final.txt").write_text("present", encoding="utf-8")

    with pytest.raises(AtomicInfrastructureError, match="nested.bin"):
        await publish_submission(
            FakeSandbox(tmp_path / "vm"),
            _task_data(output_dir),
            solve_request,
            provenance=provenance,
        )

    manifest_url = f"oss://bucket/task-prefix/output/{SUBMISSION_ID}/manifest.json"
    assert not _object_path(store, manifest_url).exists()


def _inspector_row(
    path: str,
    size_bytes: object,
    *,
    artifact_type: object = "file",
) -> dict[str, object]:
    return {
        "path": path,
        "type": artifact_type,
        "size_bytes": size_bytes,
        "sha256": hashlib.sha256(b"").hexdigest(),
    }


@pytest.mark.asyncio
async def test_publish_rejects_oversized_file_metadata_before_first_range_call(
    tmp_path: Path,
    fake_oss: tuple[Path, Path],
    solve_request: SolveRequest,
    provenance: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def inspect(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
        return [
            _inspector_row("output/final.txt", 6),
            _inspector_row("output/nested/nested.bin", 1),
        ]

    monkeypatch.setattr("ale_run.atomic.oss_submission._inspect_submission_artifacts", inspect)
    monkeypatch.setattr("ale_run.atomic.oss_submission._MAX_ARTIFACT_BYTES", 5)
    sandbox = FakeSandbox(tmp_path / "vm")

    with pytest.raises(AtomicInfrastructureError, match="per-file"):
        await publish_submission(
            sandbox,
            _task_data(tmp_path / "output"),
            solve_request,
            provenance=provenance,
        )

    assert sandbox.range_calls == []


@pytest.mark.asyncio
async def test_publish_rejects_cumulative_metadata_oversize_before_first_range_call(
    tmp_path: Path,
    fake_oss: tuple[Path, Path],
    solve_request: SolveRequest,
    provenance: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def inspect(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
        return [
            _inspector_row("output/final.txt", 4),
            _inspector_row("output/nested/nested.bin", 4),
        ]

    monkeypatch.setattr("ale_run.atomic.oss_submission._inspect_submission_artifacts", inspect)
    monkeypatch.setattr("ale_run.atomic.oss_submission._MAX_ARTIFACT_BYTES", 5)
    monkeypatch.setattr("ale_run.atomic.oss_submission._MAX_ARTIFACT_TOTAL_BYTES", 7)
    sandbox = FakeSandbox(tmp_path / "vm")

    with pytest.raises(AtomicInfrastructureError, match="total"):
        await publish_submission(
            sandbox,
            _task_data(tmp_path / "output"),
            solve_request,
            provenance=provenance,
        )

    assert sandbox.range_calls == []


@pytest.mark.parametrize(
    "rows",
    [
        [
            _inspector_row("output/other.txt", 0),
            _inspector_row("output/nested/nested.bin", 0),
        ],
        [
            _inspector_row("output/final.txt", 0, artifact_type="directory"),
            _inspector_row("output/nested/nested.bin", 0),
        ],
        [
            _inspector_row("output/final.txt", True),
            _inspector_row("output/nested/nested.bin", 0),
        ],
    ],
    ids=["path-mismatch", "not-regular-file", "non-integer-size"],
)
@pytest.mark.asyncio
async def test_publish_rejects_unsafe_inspector_metadata_before_first_range_call(
    rows: list[dict[str, object]],
    tmp_path: Path,
    fake_oss: tuple[Path, Path],
    solve_request: SolveRequest,
    provenance: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def inspect(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
        return rows

    monkeypatch.setattr("ale_run.atomic.oss_submission._inspect_submission_artifacts", inspect)
    sandbox = FakeSandbox(tmp_path / "vm")

    with pytest.raises(AtomicInfrastructureError, match="inspector"):
        await publish_submission(
            sandbox,
            _task_data(tmp_path / "output"),
            solve_request,
            provenance=provenance,
        )

    assert sandbox.range_calls == []


@pytest.mark.asyncio
async def test_publish_categorizes_malformed_range_response_and_cleans_staging(
    tmp_path: Path,
    fake_oss: tuple[Path, Path],
    solve_request: SolveRequest,
    provenance: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _store, log = fake_oss
    output_dir = tmp_path / "output"
    (output_dir / "nested").mkdir(parents=True)
    (output_dir / "final.txt").write_text("finished\n", encoding="utf-8")
    (output_dir / "nested" / "nested.bin").write_bytes(b"nested")
    sandbox = FakeSandbox(tmp_path / "vm")

    async def malformed_range(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(sandbox, "download_range", malformed_range)

    with pytest.raises(AtomicInfrastructureError) as caught:
        await publish_submission(
            sandbox,
            _task_data(output_dir),
            solve_request,
            provenance=provenance,
        )

    assert caught.value.category == "submission_integrity"
    assert "range response" in caught.value.message
    assert not [call for call in _calls(log) if call.get("direction") == "upload"]


@pytest.mark.asyncio
async def test_publish_validates_full_manifest_before_any_manifest_write(
    tmp_path: Path,
    fake_oss: tuple[Path, Path],
    solve_request: SolveRequest,
    provenance: dict[str, object],
) -> None:
    store, log = fake_oss
    output_dir = tmp_path / "output"
    (output_dir / "nested").mkdir(parents=True)
    (output_dir / "final.txt").write_text("finished\n", encoding="utf-8")
    (output_dir / "nested" / "nested.bin").write_bytes(b"nested")
    sandbox = FakeSandbox(tmp_path / "vm")
    task_data = _task_data(output_dir)

    with pytest.raises(AtomicInfrastructureError) as caught:
        await publish_submission(
            sandbox,
            task_data,
            solve_request,
            provenance=provenance | {"started_at": "not-a-datetime"},
        )

    assert caught.value.category == "submission_integrity"
    assert not [
        call
        for call in _calls(log)
        if call.get("direction") == "upload" and str(call["dst"]).endswith("/manifest.json")
    ]

    manifest = await publish_submission(
        sandbox,
        task_data,
        solve_request,
        provenance=provenance,
    )

    assert manifest.started_at == provenance["started_at"]
    manifest_url = (
        f"{solve_request.submission_root}/output/{solve_request.submission_id}/manifest.json"
    )
    assert _object_path(store, manifest_url).is_file()


@pytest.mark.parametrize("task_identity", ["traversal", "absolute", "symlink"])
@pytest.mark.asyncio
async def test_publish_rejects_noncanonical_task_identity_without_oss_writes(
    task_identity: str,
    tmp_path: Path,
    fake_oss: tuple[Path, Path],
    solve_request: SolveRequest,
    task_repo: Path,
    provenance: dict[str, object],
) -> None:
    _store, log = fake_oss
    if task_identity == "traversal":
        task_path = "folder/../toy"
    elif task_identity == "absolute":
        task_path = str(task_repo / "tasks" / "toy")
    else:
        (task_repo / "tasks" / "alias").symlink_to("toy", target_is_directory=True)
        task_path = "alias"
    request = solve_request.model_copy(update={"task_path": task_path})
    output_dir = tmp_path / "output"
    (output_dir / "nested").mkdir(parents=True)
    (output_dir / "final.txt").write_text("finished\n", encoding="utf-8")
    (output_dir / "nested" / "nested.bin").write_bytes(b"nested")

    with pytest.raises(AtomicInfrastructureError) as caught:
        await publish_submission(
            FakeSandbox(tmp_path / "vm"),
            _task_data(output_dir),
            request,
            provenance=provenance,
        )

    assert caught.value.category == "submission_integrity"
    assert not [call for call in _calls(log) if call.get("direction") == "upload"]


@pytest.mark.asyncio
async def test_publish_remote_verification_failure_leaves_no_manifest(
    tmp_path: Path,
    fake_oss: tuple[Path, Path],
    solve_request: SolveRequest,
    provenance: dict[str, object],
) -> None:
    store, _log = fake_oss
    (store.parent / "corrupt-nested-metadata").touch()
    output_dir = tmp_path / "output"
    (output_dir / "nested").mkdir(parents=True)
    (output_dir / "final.txt").write_text("finished\n", encoding="utf-8")
    (output_dir / "nested" / "nested.bin").write_bytes(b"nested")

    with pytest.raises(AtomicInfrastructureError) as caught:
        await publish_submission(
            FakeSandbox(tmp_path / "vm"),
            _task_data(output_dir),
            solve_request,
            provenance=provenance,
        )

    assert caught.value.category == "submission_storage"
    base = f"{solve_request.submission_root}/output/{solve_request.submission_id}"
    assert _object_path(store, f"{base}/artifacts/output/final.txt").is_file()
    assert _object_path(store, f"{base}/artifacts/output/nested/nested.bin").is_file()
    assert not _object_path(store, f"{base}/manifest.json").exists()


@pytest.mark.asyncio
async def test_publish_bounds_ossutil_output_before_manifest_commit(
    tmp_path: Path,
    fake_oss: tuple[Path, Path],
    solve_request: SolveRequest,
    provenance: dict[str, object],
) -> None:
    store, _log = fake_oss
    (store.parent / "oversized-stat-output").touch()
    output_dir = tmp_path / "output"
    (output_dir / "nested").mkdir(parents=True)
    (output_dir / "final.txt").write_text("finished\n", encoding="utf-8")
    (output_dir / "nested" / "nested.bin").write_bytes(b"nested")

    with pytest.raises(AtomicInfrastructureError, match="exceeded 64 KiB"):
        await publish_submission(
            FakeSandbox(tmp_path / "vm"),
            _task_data(output_dir),
            solve_request,
            provenance=provenance,
        )

    base = f"{solve_request.submission_root}/output/{solve_request.submission_id}"
    assert not _object_path(store, f"{base}/manifest.json").exists()
    assert "capture_output=True" not in _PUBLISH_SCRIPT


@pytest.mark.asyncio
async def test_publish_manifest_commit_failure_is_retryable_storage_failure(
    tmp_path: Path,
    fake_oss: tuple[Path, Path],
    solve_request: SolveRequest,
    provenance: dict[str, object],
) -> None:
    store, _log = fake_oss
    (store.parent / "fail-manifest-upload").touch()
    output_dir = tmp_path / "output"
    (output_dir / "nested").mkdir(parents=True)
    (output_dir / "final.txt").write_text("finished\n", encoding="utf-8")
    (output_dir / "nested" / "nested.bin").write_bytes(b"nested")

    with pytest.raises(AtomicInfrastructureError) as caught:
        await publish_submission(
            FakeSandbox(tmp_path / "vm"),
            _task_data(output_dir),
            solve_request,
            provenance=provenance,
        )

    assert caught.value.category == "submission_storage"
    base = f"{solve_request.submission_root}/output/{solve_request.submission_id}"
    assert _object_path(store, f"{base}/artifacts/output/final.txt").is_file()
    assert _object_path(store, f"{base}/artifacts/output/nested/nested.bin").is_file()
    assert not _object_path(store, f"{base}/manifest.json").exists()


@pytest.mark.asyncio
async def test_publish_rejects_symlink_without_any_oss_write(
    tmp_path: Path,
    fake_oss: tuple[Path, Path],
    solve_request: SolveRequest,
    provenance: dict[str, object],
) -> None:
    _store, log = fake_oss
    output_dir = tmp_path / "output"
    (output_dir / "nested").mkdir(parents=True)
    (output_dir / "final.txt").write_text("present", encoding="utf-8")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"secret")
    os.symlink(outside, output_dir / "nested" / "nested.bin")

    with pytest.raises(AtomicInfrastructureError, match="symlink"):
        await publish_submission(
            FakeSandbox(tmp_path / "vm"),
            _task_data(output_dir),
            solve_request,
            provenance=provenance,
        )

    assert not [call for call in _calls(log) if call.get("direction") == "upload"]


@pytest.mark.asyncio
async def test_publish_retry_returns_existing_identical_manifest_without_writes(
    tmp_path: Path,
    fake_oss: tuple[Path, Path],
    solve_request: SolveRequest,
    provenance: dict[str, object],
) -> None:
    _store, log = fake_oss
    output_dir = tmp_path / "output"
    (output_dir / "nested").mkdir(parents=True)
    (output_dir / "final.txt").write_text("finished\n", encoding="utf-8")
    (output_dir / "nested" / "nested.bin").write_bytes(b"nested")
    sandbox = FakeSandbox(tmp_path / "vm")
    task_data = _task_data(output_dir)
    first = await publish_submission(sandbox, task_data, solve_request, provenance=provenance)
    writes_before_retry = len([call for call in _calls(log) if call.get("direction") == "upload"])

    second = await publish_submission(sandbox, task_data, solve_request, provenance=provenance)

    assert second == first
    assert (
        len([call for call in _calls(log) if call.get("direction") == "upload"])
        == writes_before_retry
    )


@pytest.mark.asyncio
async def test_publish_existing_conflicting_manifest_is_idempotency_error(
    tmp_path: Path,
    fake_oss: tuple[Path, Path],
    solve_request: SolveRequest,
    provenance: dict[str, object],
) -> None:
    store, _log = fake_oss
    output_dir = tmp_path / "output"
    (output_dir / "nested").mkdir(parents=True)
    (output_dir / "final.txt").write_text("finished\n", encoding="utf-8")
    (output_dir / "nested" / "nested.bin").write_bytes(b"nested")
    sandbox = FakeSandbox(tmp_path / "vm")
    task_data = _task_data(output_dir)
    await publish_submission(sandbox, task_data, solve_request, provenance=provenance)
    manifest_path = _object_path(
        store,
        f"oss://bucket/task-prefix/output/{SUBMISSION_ID}/manifest.json",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_id"] = "different-model"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(AtomicInfrastructureError) as caught:
        await publish_submission(sandbox, task_data, solve_request, provenance=provenance)

    assert caught.value.category == "idempotency_conflict"


@pytest.mark.asyncio
async def test_concurrent_publish_uses_isolated_local_manifest_files(
    tmp_path: Path,
    fake_oss: tuple[Path, Path],
    solve_request: SolveRequest,
    provenance: dict[str, object],
) -> None:
    store, _log = fake_oss
    (store.parent / "concurrent-manifest-barrier").touch()
    output_dir = tmp_path / "output"
    (output_dir / "nested").mkdir(parents=True)
    (output_dir / "final.txt").write_text("finished\n", encoding="utf-8")
    (output_dir / "nested" / "nested.bin").write_bytes(b"nested")
    sandbox = FakeSandbox(tmp_path / "vm")

    manifests = await asyncio.gather(
        publish_submission(
            sandbox,
            _task_data(output_dir),
            solve_request,
            provenance=provenance,
        ),
        publish_submission(
            sandbox,
            _task_data(output_dir),
            solve_request,
            provenance=provenance,
        ),
    )

    committed = json.loads(
        _object_path(
            store,
            f"{solve_request.submission_root}/output/{solve_request.submission_id}/manifest.json",
        ).read_bytes()
    )
    assert manifests[0] == manifests[1]
    assert committed == manifests[0].model_dump(mode="json")


@pytest.mark.asyncio
async def test_concurrent_same_key_different_bytes_conflict_without_overwrite(
    tmp_path: Path,
    fake_oss: tuple[Path, Path],
    solve_request: SolveRequest,
    provenance: dict[str, object],
) -> None:
    store, _log = fake_oss
    (store.parent / "concurrent-artifact-barrier").touch()
    task_card = solve_request.task_repo / "tasks" / "toy" / "task_card.json"
    task_card.write_text(
        json.dumps(
            {
                "outputFiles": [
                    {"name": "final.txt", "path": "output/final.txt"},
                ]
            }
        ),
        encoding="utf-8",
    )
    first_output = tmp_path / "first-output"
    second_output = tmp_path / "second-output"
    first_output.mkdir()
    second_output.mkdir()
    (first_output / "final.txt").write_bytes(b"first bytes")
    (second_output / "final.txt").write_bytes(b"second bytes")
    sandbox = FakeSandbox(tmp_path / "vm")

    outcomes = await asyncio.gather(
        publish_submission(
            sandbox,
            _task_data(first_output),
            solve_request,
            provenance=provenance,
        ),
        publish_submission(
            sandbox,
            _task_data(second_output),
            solve_request,
            provenance=provenance,
        ),
        return_exceptions=True,
    )

    manifests = [outcome for outcome in outcomes if not isinstance(outcome, Exception)]
    conflicts = [outcome for outcome in outcomes if isinstance(outcome, AtomicInfrastructureError)]
    assert len(manifests) == 1
    assert len(conflicts) == 1
    assert conflicts[0].category == "idempotency_conflict"
    artifact_url = (
        f"{solve_request.submission_root}/output/{solve_request.submission_id}/"
        "artifacts/output/final.txt"
    )
    committed_bytes = _object_path(store, artifact_url).read_bytes()
    assert committed_bytes in {b"first bytes", b"second bytes"}
    assert hashlib.sha256(committed_bytes).hexdigest() == manifests[0].artifacts[0].sha256


def _seed_submission(
    store: Path,
    *,
    solve_request: SolveRequest,
    provenance: dict[str, object],
    artifact_bytes: bytes = b"staged content",
) -> str:
    base = f"oss://bucket/task-prefix/output/{SUBMISSION_ID}"
    artifact_url = f"{base}/artifacts/output/final.txt"
    artifact_path = _object_path(store, artifact_url)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(artifact_bytes)
    manifest = {
        "schema_version": 1,
        "status": "submitted",
        "submission_id": str(SUBMISSION_ID),
        "task_path": solve_request.task_path,
        "variant_index": solve_request.variant_index,
        "task_commit": solve_request.task_commit,
        "image_id": solve_request.image_id,
        "ale_run_id": provenance["ale_run_id"],
        "agent_id": solve_request.agent_id,
        "model_id": provenance["model_id"],
        "config_digest": provenance["config_digest"],
        "started_at": provenance["started_at"].isoformat(),
        "completed_at": provenance["completed_at"].isoformat(),
        "artifacts": [
            {
                "path": "output/final.txt",
                "size_bytes": len(artifact_bytes),
                "sha256": hashlib.sha256(artifact_bytes).hexdigest(),
                "media_type": "text/plain",
            }
        ],
    }
    manifest_path = _object_path(store, f"{base}/manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return base


def _evaluation_result(
    request: EvaluateRequest,
    *,
    score: float,
    outcome: str,
) -> EvaluationResult:
    reward = {
        "score": score,
        "outcome": outcome,
        "report": {"hard_gate_passed": outcome != "invalid_output"},
    }
    return EvaluationResult(
        status="scored",
        submission_id=request.submission_id,
        task_path=request.task_path,
        variant_index=request.variant_index,
        task_commit=request.task_commit,
        image_id=request.image_id,
        evaluator_id=request.evaluator_id,
        evaluator_version=request.evaluator_version,
        outcome=outcome,
        score=score,
        rubric_hash="d" * 64,
        harbor=HarborProvenance(
            reward=reward,
            details_path="evidence/reward-details.json",
        ),
    )


@pytest.mark.asyncio
async def test_stage_downloads_manifest_first_and_only_declared_artifacts(
    tmp_path: Path,
    fake_oss: tuple[Path, Path],
    solve_request: SolveRequest,
    evaluate_request: EvaluateRequest,
    provenance: dict[str, object],
) -> None:
    store, log = fake_oss
    base = _seed_submission(store, solve_request=solve_request, provenance=provenance)
    undeclared = _object_path(store, f"{base}/artifacts/output/private.txt")
    undeclared.parent.mkdir(parents=True, exist_ok=True)
    undeclared.write_text("private", encoding="utf-8")
    output_dir = tmp_path / "eval output 'quoted'"

    manifest = await stage_submission(
        FakeSandbox(tmp_path / "vm"),
        _task_data(output_dir),
        evaluate_request,
    )

    assert manifest.submission_id == SUBMISSION_ID
    assert (output_dir / "final.txt").read_bytes() == b"staged content"
    assert not (output_dir / "private.txt").exists()
    downloads = [call for call in _calls(log) if call.get("direction") == "download"]
    assert downloads[0]["src"] == f"{base}/manifest.json"
    assert [call["src"] for call in downloads[1:]] == [f"{base}/artifacts/output/final.txt"]


@pytest.mark.asyncio
async def test_stage_rejects_request_identity_mismatch_before_artifact_download(
    tmp_path: Path,
    fake_oss: tuple[Path, Path],
    solve_request: SolveRequest,
    evaluate_request: EvaluateRequest,
    provenance: dict[str, object],
) -> None:
    store, log = fake_oss
    base = _seed_submission(store, solve_request=solve_request, provenance=provenance)
    mismatched = evaluate_request.model_copy(update={"task_commit": "c" * 40})

    with pytest.raises(AtomicInfrastructureError) as caught:
        await stage_submission(
            FakeSandbox(tmp_path / "vm"),
            _task_data(tmp_path / "output"),
            mismatched,
        )

    assert caught.value.category == "submission_integrity"
    downloads = [call for call in _calls(log) if call.get("direction") == "download"]
    assert [call["src"] for call in downloads] == [f"{base}/manifest.json"]


@pytest.mark.asyncio
async def test_stage_checks_solve_request_agent_identity(
    tmp_path: Path,
    fake_oss: tuple[Path, Path],
    solve_request: SolveRequest,
    provenance: dict[str, object],
) -> None:
    store, log = fake_oss
    base = _seed_submission(store, solve_request=solve_request, provenance=provenance)
    mismatched = solve_request.model_copy(update={"agent_id": "different-agent"})

    with pytest.raises(AtomicInfrastructureError) as caught:
        await stage_submission(
            FakeSandbox(tmp_path / "vm"),
            _task_data(tmp_path / "output"),
            mismatched,
        )

    assert caught.value.category == "submission_integrity"
    assert "agent_id" in caught.value.message
    downloads = [call for call in _calls(log) if call.get("direction") == "download"]
    assert [call["src"] for call in downloads] == [f"{base}/manifest.json"]


@pytest.mark.asyncio
async def test_stage_rejects_internal_task_symlink_identity_before_artifact_download(
    tmp_path: Path,
    fake_oss: tuple[Path, Path],
    solve_request: SolveRequest,
    evaluate_request: EvaluateRequest,
    task_repo: Path,
    provenance: dict[str, object],
) -> None:
    store, log = fake_oss
    (task_repo / "tasks" / "alias").symlink_to("toy", target_is_directory=True)
    aliased_solve = solve_request.model_copy(update={"task_path": "alias"})
    aliased_evaluate = evaluate_request.model_copy(update={"task_path": "alias"})
    base = _seed_submission(
        store,
        solve_request=aliased_solve,
        provenance=provenance,
    )

    with pytest.raises(AtomicInfrastructureError) as caught:
        await stage_submission(
            FakeSandbox(tmp_path / "vm"),
            _task_data(tmp_path / "output"),
            aliased_evaluate,
        )

    assert caught.value.category == "submission_integrity"
    downloads = [call for call in _calls(log) if call.get("direction") == "download"]
    assert [call["src"] for call in downloads] == [f"{base}/manifest.json"]


@pytest.mark.asyncio
async def test_stage_rejects_manifest_path_traversal_before_artifact_download(
    tmp_path: Path,
    fake_oss: tuple[Path, Path],
    solve_request: SolveRequest,
    evaluate_request: EvaluateRequest,
    provenance: dict[str, object],
) -> None:
    store, log = fake_oss
    base = _seed_submission(store, solve_request=solve_request, provenance=provenance)
    manifest_path = _object_path(store, f"{base}/manifest.json")
    manifest = json.loads(manifest_path.read_bytes())
    manifest["artifacts"][0]["path"] = "output/../../outside.txt"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(AtomicInfrastructureError) as caught:
        await stage_submission(
            FakeSandbox(tmp_path / "vm"),
            _task_data(tmp_path / "output"),
            evaluate_request,
        )

    assert caught.value.category == "submission_integrity"
    assert "unsafe artifact path" in caught.value.message
    downloads = [call for call in _calls(log) if call.get("direction") == "download"]
    assert [call["src"] for call in downloads] == [f"{base}/manifest.json"]


@pytest.mark.asyncio
async def test_stage_rejects_unbounded_artifact_count_before_artifact_download(
    tmp_path: Path,
    fake_oss: tuple[Path, Path],
    solve_request: SolveRequest,
    evaluate_request: EvaluateRequest,
    provenance: dict[str, object],
) -> None:
    store, log = fake_oss
    base = _seed_submission(store, solve_request=solve_request, provenance=provenance)
    manifest_path = _object_path(store, f"{base}/manifest.json")
    manifest = json.loads(manifest_path.read_bytes())
    manifest["artifacts"] = [
        {
            "path": f"output/file-{index}.txt",
            "size_bytes": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
            "media_type": "text/plain",
        }
        for index in range(1025)
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(AtomicInfrastructureError) as caught:
        await stage_submission(
            FakeSandbox(tmp_path / "vm"),
            _task_data(tmp_path / "output"),
            evaluate_request,
        )

    assert caught.value.category == "submission_integrity"
    assert "artifact count" in caught.value.message
    downloads = [call for call in _calls(log) if call.get("direction") == "download"]
    assert [call["src"] for call in downloads] == [f"{base}/manifest.json"]


@pytest.mark.asyncio
async def test_stage_recomputes_digest_in_vm_and_rejects_mismatch(
    tmp_path: Path,
    fake_oss: tuple[Path, Path],
    solve_request: SolveRequest,
    evaluate_request: EvaluateRequest,
    provenance: dict[str, object],
) -> None:
    store, _log = fake_oss
    base = _seed_submission(store, solve_request=solve_request, provenance=provenance)
    _object_path(store, f"{base}/artifacts/output/final.txt").write_bytes(b"tampered content")

    with pytest.raises(AtomicInfrastructureError) as caught:
        await stage_submission(
            FakeSandbox(tmp_path / "vm"),
            _task_data(tmp_path / "output"),
            evaluate_request,
        )

    assert caught.value.category == "submission_integrity"
    assert "final.txt" in caught.value.message


@pytest.mark.asyncio
async def test_result_publish_is_canonical_and_idempotent(
    tmp_path: Path,
    fake_oss: tuple[Path, Path],
    evaluate_request: EvaluateRequest,
) -> None:
    store, log = fake_oss
    sandbox = FakeSandbox(tmp_path / "vm")
    result = _evaluation_result(evaluate_request, score=0.75, outcome="valid")
    expected_url = (
        "oss://bucket/task-prefix/output/"
        f"{SUBMISSION_ID}/evaluations/evaluator%2Fmain/"
        f"{EVALUATOR_VERSION}/result.json"
    )

    first_url = await publish_evaluation_result(sandbox, evaluate_request, result)
    writes_before_retry = len([call for call in _calls(log) if call.get("direction") == "upload"])
    second_url = await publish_evaluation_result(sandbox, evaluate_request, result)

    assert first_url == second_url == expected_url
    assert _object_path(store, expected_url).read_bytes() == json.dumps(
        result.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert (
        len([call for call in _calls(log) if call.get("direction") == "upload"])
        == writes_before_retry
    )


@pytest.mark.asyncio
async def test_result_publish_rejects_existing_different_bytes(
    tmp_path: Path,
    fake_oss: tuple[Path, Path],
    evaluate_request: EvaluateRequest,
) -> None:
    store, _log = fake_oss
    sandbox = FakeSandbox(tmp_path / "vm")
    first = _evaluation_result(evaluate_request, score=0.75, outcome="valid")
    await publish_evaluation_result(sandbox, evaluate_request, first)

    with pytest.raises(AtomicInfrastructureError) as caught:
        await publish_evaluation_result(
            sandbox,
            evaluate_request,
            _evaluation_result(
                evaluate_request,
                score=0.0,
                outcome="invalid_output",
            ),
        )

    assert caught.value.category == "idempotency_conflict"
    result_url = (
        "oss://bucket/task-prefix/output/"
        f"{SUBMISSION_ID}/evaluations/evaluator%2Fmain/"
        f"{EVALUATOR_VERSION}/result.json"
    )
    assert json.loads(_object_path(store, result_url).read_bytes())["score"] == 0.75


@pytest.mark.asyncio
async def test_result_publish_rejects_result_identity_mismatch_before_upload(
    tmp_path: Path,
    fake_oss: tuple[Path, Path],
    evaluate_request: EvaluateRequest,
) -> None:
    _store, log = fake_oss
    result = _evaluation_result(
        evaluate_request,
        score=0.75,
        outcome="valid",
    ).model_copy(update={"evaluator_id": "different-evaluator"})

    with pytest.raises(AtomicInfrastructureError) as caught:
        await publish_evaluation_result(
            FakeSandbox(tmp_path / "vm"),
            evaluate_request,
            result,
        )

    assert caught.value.category == "submission_integrity"
    assert "evaluator_id" in caught.value.message
    assert not [call for call in _calls(log) if call.get("direction") == "upload"]


@pytest.mark.asyncio
async def test_windows_commands_encode_dynamic_root_path_and_evaluator_identity(
    evaluate_request: EvaluateRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_ensure(_sandbox: object) -> None:
        return None

    monkeypatch.setattr(
        "ale_run.atomic.oss_submission.ossbucket.ensure_ossutil",
        fake_ensure,
    )
    unsafe_root = "oss://bucket/%ROOT%&|<>^"
    unsafe_evaluator = "eval/%EVALUATOR%&|<>^"
    unsafe_path = r"E:\output\%TEMP%&|<>^\result.json"
    request = evaluate_request.model_copy(
        update={
            "submission_root": unsafe_root,
            "evaluator_id": unsafe_evaluator,
        }
    )
    sandbox = FakeWindowsSandbox([1, 0, 1])

    result_url = await publish_evaluation_result(
        sandbox,
        request,
        _evaluation_result(request, score=1.0, outcome="valid"),
    )
    await _download(sandbox, f"{unsafe_root}/%OBJECT%&|<>^", unsafe_path)

    assert "%25EVALUATOR%25%26%7C%3C%3E%5E" in result_url
    decoded = [_decode_powershell_command(command) for command in sandbox.commands]
    assert any(result_url in script for script in decoded)
    assert any(unsafe_path.replace("'", "''") in script for script in decoded)
    assert all(
        dynamic not in command
        for command in sandbox.commands
        for dynamic in (unsafe_root, unsafe_evaluator, unsafe_path)
    )
    assert "shell=True" not in _PUBLISH_SCRIPT
