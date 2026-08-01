import json
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
    EvaluatorRegistryRecord,
)


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
    records = list(root.glob("*.json"))
    assert len(records) == 1
    assert records[0].stat().st_mode & 0o777 == 0o600
    assert registry.publish_ready(_identity(), _record()) == _record()
    assert list(root.glob("*.json")) == records


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
    assert len(list(registry.root.glob("*.json"))) == 1


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
    record_path = next(registry.root.glob("*.json"))
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
