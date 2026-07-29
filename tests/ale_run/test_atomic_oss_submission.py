from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
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
    SolveRequest,
)
from ale_run.atomic.oss_submission import (
    publish_evaluation_result,
    publish_submission,
    stage_submission,
)
from ale_run.base_interface import TaskDataSpec

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

    async def write_file(self, path: str, content: str | bytes) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            destination.write_bytes(content)
        else:
            destination.write_text(content, encoding="utf-8")

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
        shutil.copyfile(source, destination)
        entry["direction"] = "upload"
        digest = None
        for i, value in enumerate(op_args[2:]):
            if value in {"--meta", "--metadata"}:
                digest = op_args[i + 3].split(":", 1)[-1].split("=", 1)[-1]
        if digest is not None:
            if (
                (store.parent / "corrupt-nested-metadata").is_file()
                and dst.endswith("/nested.bin")
            ):
                digest = "0" * 64
            metadata_path(destination).write_text(
                json.dumps({"x-oss-meta-sha256": digest}), encoding="utf-8"
            )
elif op == "stat":
    url = op_args[0]
    source = object_path(url)
    entry["src"] = url
    if not source.is_file():
        print("NoSuchKey", file=sys.stderr)
        sys.exit(1)
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
    shutil.rmtree(output_dir)

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
    other_request = solve_request.model_copy(
        update={"submission_id": UUID("87654321-4321-8765-4321-876543218765")}
    )
    other_provenance = provenance | {"model_id": "other-model"}

    await asyncio.gather(
        publish_submission(
            sandbox,
            _task_data(output_dir),
            solve_request,
            provenance=provenance,
        ),
        publish_submission(
            sandbox,
            _task_data(output_dir),
            other_request,
            provenance=other_provenance,
        ),
    )

    first_manifest = json.loads(
        _object_path(
            store,
            f"{solve_request.submission_root}/output/{solve_request.submission_id}/manifest.json",
        ).read_bytes()
    )
    second_manifest = json.loads(
        _object_path(
            store,
            f"{other_request.submission_root}/output/{other_request.submission_id}/manifest.json",
        ).read_bytes()
    )
    assert first_manifest["submission_id"] == str(solve_request.submission_id)
    assert second_manifest["submission_id"] == str(other_request.submission_id)


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
    result = EvaluationResult(status="scored", outcome="valid", score=0.75)
    expected_url = (
        "oss://bucket/task-prefix/output/"
        f"{SUBMISSION_ID}/evaluations/evaluator%2Fmain/"
        f"{EVALUATOR_VERSION}/result.json"
    )

    first_url = await publish_evaluation_result(sandbox, evaluate_request, result)
    writes_before_retry = len([call for call in _calls(log) if call.get("direction") == "upload"])
    second_url = await publish_evaluation_result(sandbox, evaluate_request, result)

    assert first_url == second_url == expected_url
    assert _object_path(store, expected_url).read_bytes() == (
        b'{"outcome":"valid","schema_version":1,"score":0.75,"status":"scored"}'
    )
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
    first = EvaluationResult(status="scored", outcome="valid", score=0.75)
    await publish_evaluation_result(sandbox, evaluate_request, first)

    with pytest.raises(AtomicInfrastructureError) as caught:
        await publish_evaluation_result(
            sandbox,
            evaluate_request,
            EvaluationResult(status="scored", outcome="invalid_output", score=0.0),
        )

    assert caught.value.category == "idempotency_conflict"
    result_url = (
        "oss://bucket/task-prefix/output/"
        f"{SUBMISSION_ID}/evaluations/evaluator%2Fmain/"
        f"{EVALUATOR_VERSION}/result.json"
    )
    assert json.loads(_object_path(store, result_url).read_bytes())["score"] == 0.75
