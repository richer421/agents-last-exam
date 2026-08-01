import asyncio
import json
import multiprocessing
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ale_run.atomic.author_registry import (
    FilesystemAuthorRegistry,
    author_registry_key,
)
from ale_run.atomic.contracts import (
    AtomicInfrastructureError,
    AuthorEvaluatorRegistryIdentity,
    EvaluateRequest,
    EvaluatorRegistryRecord,
)
from ale_run.atomic.evaluator_registry import _read_registry_record


def _identity() -> AuthorEvaluatorRegistryIdentity:
    return AuthorEvaluatorRegistryIdentity(
        task_commit="a" * 40,
        rubric_hash="b" * 64,
        reference_manifest_hash="c" * 64,
        evaluator_sdk_version="1.2.3",
        image_id="m-authoring-image",
    )


def _record() -> EvaluatorRegistryRecord:
    return EvaluatorRegistryRecord(
        status="ready",
        task_path="demo/example",
        variant_index=0,
        task_commit="a" * 40,
        evaluator_id="rubric",
        evaluator_version="d" * 40,
        rubric_hash="b" * 64,
        reference_manifest_uri="oss://ale-reference/example/manifest.json",
        reference_manifest_hash="c" * 64,
        evaluator_sdk_version="1.2.3",
        harbor_version="2.0.0",
        rewardkit_version="1.0.0",
        image_id="m-authoring-image",
        pull_request_url="https://github.com/openai/ale-tasks/pull/123",
        ci_run_id="987654",
        ready_at=datetime(2026, 8, 2, tzinfo=UTC),
    )


async def _hold_claim(registry, identity, entered: asyncio.Event, release: asyncio.Event) -> None:
    async with registry.claim(identity):
        entered.set()
        await release.wait()


def _hold_claim_in_process(root: str, identity_json: str, entered, release) -> None:
    identity = AuthorEvaluatorRegistryIdentity.model_validate_json(identity_json)

    async def hold() -> None:
        registry = FilesystemAuthorRegistry(root)
        async with registry.claim(identity):
            entered.set()
            while not release.is_set():
                await asyncio.sleep(0.01)

    asyncio.run(hold())


def test_author_registry_key_is_canonical_and_uses_only_identity_tuple() -> None:
    identity = _identity()

    assert author_registry_key(identity) == author_registry_key(
        AuthorEvaluatorRegistryIdentity.model_validate(json.loads(identity.model_dump_json()))
    )
    assert len(author_registry_key(identity)) == 64


def test_filesystem_registry_atomically_publishes_and_reuses_ready_record(
    tmp_path: Path,
) -> None:
    root = tmp_path / "author-registry"
    registry = FilesystemAuthorRegistry(root)

    published = registry.publish_ready(_identity(), _record())

    assert published == _record()
    assert registry.get_ready(_identity()) == _record()
    assert root.stat().st_mode & 0o777 == 0o700
    records = sorted(root.glob("*.json"))
    assert len(records) == 2
    assert all(record.stat().st_mode & 0o777 == 0o600 for record in records)
    assert registry.publish_ready(_identity(), _record()) == _record()
    assert sorted(root.glob("*.json")) == records


def test_author_publication_is_readable_by_evaluator_registry(tmp_path: Path) -> None:
    root = tmp_path / "control-plane"
    registry = FilesystemAuthorRegistry(root)
    record = _record()
    request = EvaluateRequest(
        submission_id="00000000-0000-0000-0000-000000000001",
        runtime_spec_path=tmp_path / "runtime.yaml",
        task_repo=tmp_path / "task-repo",
        task_path=record.task_path,
        variant_index=record.variant_index,
        task_commit=record.task_commit,
        image_id=record.image_id,
        submission_root="oss://ale-submissions/example",
        evaluator_id=record.evaluator_id,
        evaluator_version=record.evaluator_version,
    )

    registry.publish_ready(_identity(), record)

    assert (
        EvaluatorRegistryRecord.model_validate_json(_read_registry_record(request, root)) == record
    )


def test_filesystem_registry_reuses_ready_record_when_only_ready_at_differs(
    tmp_path: Path,
) -> None:
    registry = FilesystemAuthorRegistry(tmp_path / "author-registry")
    original = _record()
    later = original.model_copy(update={"ready_at": datetime(2026, 8, 3, tzinfo=UTC)})

    registry.publish_ready(_identity(), original)

    assert registry.publish_ready(_identity(), later) == original


def test_filesystem_registry_serializes_concurrent_same_key_publication(
    tmp_path: Path,
) -> None:
    registry = FilesystemAuthorRegistry(tmp_path / "author-registry")

    with ThreadPoolExecutor(max_workers=8) as executor:
        published = list(
            executor.map(
                lambda _index: registry.publish_ready(_identity(), _record()),
                range(24),
            )
        )

    assert published == [_record()] * 24
    assert len(list(registry.root.glob("*.json"))) == 2


@pytest.mark.asyncio
async def test_filesystem_registry_claim_serializes_same_key_across_instances(
    tmp_path: Path,
) -> None:
    root = tmp_path / "author-registry"
    first_registry = FilesystemAuthorRegistry(root)
    second_registry = FilesystemAuthorRegistry(root)
    entered = asyncio.Event()
    release = asyncio.Event()

    async with first_registry.claim(_identity()):
        waiter = asyncio.create_task(_hold_claim(second_registry, _identity(), entered, release))
        await asyncio.sleep(0.05)
        assert not entered.is_set()

    await asyncio.wait_for(entered.wait(), timeout=1)
    release.set()
    await asyncio.wait_for(waiter, timeout=1)


@pytest.mark.asyncio
async def test_filesystem_registry_claim_does_not_serialize_different_keys(
    tmp_path: Path,
) -> None:
    registry = FilesystemAuthorRegistry(tmp_path / "author-registry")
    other_identity = _identity().model_copy(update={"rubric_hash": "e" * 64})
    entered = asyncio.Event()
    release = asyncio.Event()

    async with registry.claim(_identity()):
        waiter = asyncio.create_task(_hold_claim(registry, other_identity, entered, release))
        await asyncio.wait_for(entered.wait(), timeout=1)
        release.set()
        await asyncio.wait_for(waiter, timeout=1)


def test_filesystem_registry_claim_serializes_across_processes(tmp_path: Path) -> None:
    root = tmp_path / "author-registry"
    registry = FilesystemAuthorRegistry(root)
    process_context = multiprocessing.get_context("spawn")
    entered = process_context.Event()
    release_process = process_context.Event()
    process = process_context.Process(
        target=_hold_claim_in_process,
        args=(str(root), _identity().model_dump_json(), entered, release_process),
    )
    process.start()
    try:
        assert entered.wait(timeout=10)

        async def acquire_after_process() -> None:
            acquired = asyncio.Event()
            release = asyncio.Event()
            waiter = asyncio.create_task(_hold_claim(registry, _identity(), acquired, release))
            await asyncio.sleep(0.05)
            assert not acquired.is_set()
            release_process.set()
            await asyncio.wait_for(acquired.wait(), timeout=5)
            release.set()
            await asyncio.wait_for(waiter, timeout=1)

        asyncio.run(acquire_after_process())
    finally:
        release_process.set()
        process.join(timeout=10)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
    assert process.exitcode == 0


def test_cancelled_filesystem_claim_does_not_leave_blocked_executor_work(
    tmp_path: Path,
) -> None:
    registry = FilesystemAuthorRegistry(tmp_path / "author-registry")

    async def cancel_waiter() -> None:
        loop = asyncio.get_running_loop()
        single_worker = ThreadPoolExecutor(max_workers=1)
        loop.set_default_executor(single_worker)
        try:
            async with registry.claim(_identity()):
                waiter = asyncio.create_task(registry.claim(_identity()).__aenter__())
                await asyncio.sleep(0.05)
                waiter.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await waiter

                assert (
                    await asyncio.wait_for(
                        asyncio.to_thread(lambda: "available"),
                        timeout=0.2,
                    )
                    == "available"
                )
        finally:
            loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
            single_worker.shutdown(wait=True)

    asyncio.run(cancel_waiter())


def test_filesystem_registry_rejects_conflicting_record_for_same_identity(
    tmp_path: Path,
) -> None:
    registry = FilesystemAuthorRegistry(tmp_path / "author-registry")
    registry.publish_ready(_identity(), _record())

    with pytest.raises(AtomicInfrastructureError, match="conflicting ready record"):
        registry.publish_ready(
            _identity(),
            _record().model_copy(update={"evaluator_version": "e" * 40}),
        )


@pytest.mark.parametrize("corruption", ["invalid", "oversized", "permissions", "symlink"])
def test_filesystem_registry_fails_closed_for_corrupt_record(
    tmp_path: Path,
    corruption: str,
) -> None:
    registry = FilesystemAuthorRegistry(tmp_path / "author-registry")
    registry.publish_ready(_identity(), _record())
    record_path = registry.root / f"{author_registry_key(_identity())}.json"
    if corruption == "invalid":
        record_path.write_text("{}", encoding="utf-8")
    elif corruption == "oversized":
        record_path.write_bytes(b"x" * (1024 * 1024 + 1))
    elif corruption == "permissions":
        record_path.chmod(0o644)
    else:
        outside = tmp_path / "outside.json"
        record_path.rename(outside)
        record_path.symlink_to(outside)

    with pytest.raises(AtomicInfrastructureError) as caught:
        registry.get_ready(_identity())

    assert caught.value.category == "author_registry"


def test_filesystem_registry_rejects_identity_record_mismatch(tmp_path: Path) -> None:
    registry = FilesystemAuthorRegistry(tmp_path / "author-registry")

    with pytest.raises(AtomicInfrastructureError, match="identity mismatch"):
        registry.publish_ready(
            _identity(),
            _record().model_copy(update={"rubric_hash": "f" * 64}),
        )


def test_filesystem_registry_requires_private_owned_root(tmp_path: Path) -> None:
    root = tmp_path / "author-registry"
    root.mkdir(mode=0o755)
    root.chmod(0o755)

    with pytest.raises(AtomicInfrastructureError, match="permissions"):
        FilesystemAuthorRegistry(root)


def test_filesystem_registry_rejects_relative_root() -> None:
    with pytest.raises(AtomicInfrastructureError, match="absolute"):
        FilesystemAuthorRegistry(Path("relative-registry"))


def test_missing_registry_record_returns_none(tmp_path: Path) -> None:
    registry = FilesystemAuthorRegistry(tmp_path / "author-registry")

    assert registry.get_ready(_identity()) is None
    assert os.listdir(registry.root) == [f"{author_registry_key(_identity())}.lock"]
