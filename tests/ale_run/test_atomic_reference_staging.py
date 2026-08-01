from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ale_run.atomic.contracts import (
    AtomicInfrastructureError,
    EvaluatorRegistryRecord,
)
from ale_run.atomic.trusted_staging import (
    prepare_atomic_reference,
    stage_atomic_reference,
)
from tests.ale_run.test_atomic_trusted_staging import (
    _LocalSandbox,
    _task_data,
)


def _record(manifest_bytes: bytes) -> EvaluatorRegistryRecord:
    return EvaluatorRegistryRecord(
        status="ready",
        task_path="toy",
        variant_index=0,
        task_commit="a" * 40,
        evaluator_id="rubric",
        evaluator_version="b" * 40,
        rubric_hash="c" * 64,
        reference_manifest_uri="oss://trusted/demo/toy/base/reference/manifest.json",
        reference_manifest_hash=hashlib.sha256(manifest_bytes).hexdigest(),
        evaluator_sdk_version="1.2.3",
        harbor_version="0.20.0",
        rewardkit_version="0.4.0",
        image_id="m-image-123",
        pull_request_url="https://github.com/example/tasks/pull/42",
        ci_run_id="123456",
        ready_at=datetime(2026, 7, 29, tzinfo=UTC),
    )


def _manifest(reference_bytes: bytes) -> bytes:
    return json.dumps(
        {
            "schema_version": 1,
            "files": [
                {
                    "path": "reference/answer.json",
                    "size_bytes": len(reference_bytes),
                    "sha256": hashlib.sha256(reference_bytes).hexdigest(),
                }
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _host_store_runner(objects: dict[str, bytes]):
    async def run(*arguments: str):
        operation = arguments[0]
        if operation == "stat":
            url = arguments[1]
            if url not in objects:
                return 1, b"", b"NoSuchKey"
            return 0, f"Content-Length: {len(objects[url])}\n".encode(), b""
        source, destination = arguments[1:3]
        if source not in objects:
            return 1, b"", b"NoSuchKey"
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        Path(destination).write_bytes(objects[source])
        return 0, b"", b""

    return run


@pytest.mark.asyncio
async def test_reference_manifest_and_files_are_host_verified_then_vm_reverified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_bytes = b'{"answer":42}\n'
    manifest_bytes = _manifest(reference_bytes)
    record = _record(manifest_bytes)
    objects = {
        record.reference_manifest_uri: manifest_bytes,
        "oss://trusted/demo/toy/base/reference/answer.json": reference_bytes,
    }
    monkeypatch.setattr(
        "ale_run.atomic.host_oss.run_host_ossutil",
        _host_store_runner(objects),
    )
    monkeypatch.setattr(
        "ale_run.atomic.trusted_staging.run_host_ossutil",
        _host_store_runner(objects),
    )
    sandbox = _LocalSandbox(tmp_path / "vm")

    async with prepare_atomic_reference(
        record=record,
        source="oss://trusted",
        task_data=_task_data(),
        declared_reference_paths=("reference/answer.json",),
    ) as prepared:
        await stage_atomic_reference(
            sandbox,
            _task_data(),
            source="oss://trusted",
            declared_reference_paths=("reference/answer.json",),
            prepared=prepared,
        )

    reference = Path(sandbox.task_data_root) / "demo" / "toy" / "base" / "reference" / "answer.json"
    assert reference.read_bytes() == reference_bytes
    assert "ossutil" not in "\n".join(sandbox.commands).lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing", "hash_mismatch", "partial_manifest"])
async def test_reference_staging_fails_closed_for_missing_partial_or_mismatched_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    expected = b"expected"
    manifest_bytes = _manifest(expected)
    record = _record(manifest_bytes)
    objects = {record.reference_manifest_uri: manifest_bytes}
    if failure == "hash_mismatch":
        objects["oss://trusted/demo/toy/base/reference/answer.json"] = b"wrong-but-same-size"
    elif failure == "partial_manifest":
        objects["oss://trusted/demo/toy/base/reference/answer.json"] = expected
    monkeypatch.setattr(
        "ale_run.atomic.host_oss.run_host_ossutil",
        _host_store_runner(objects),
    )
    monkeypatch.setattr(
        "ale_run.atomic.trusted_staging.run_host_ossutil",
        _host_store_runner(objects),
    )
    declared = (
        ("reference/answer.json", "reference/second.json")
        if failure == "partial_manifest"
        else ("reference/answer.json",)
    )

    with pytest.raises(AtomicInfrastructureError) as caught:
        async with prepare_atomic_reference(
            record=record,
            source="oss://trusted",
            task_data=_task_data(),
            declared_reference_paths=declared,
        ):
            pass

    assert caught.value.category == "reference"
