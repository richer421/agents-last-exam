from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest

from ale_run.atomic.contracts import AtomicInfrastructureError, EvaluateRequest
from ale_run.atomic.evaluator_registry import (
    materialize_evaluator_checkout,
    validate_evaluator_registry,
)


def _commit(repo: Path, message: str) -> str:
    subprocess.run(["git", "-C", str(repo), "add", "tasks/toy"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            message,
        ],
        check=True,
    )
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _registry_record_filename(identity: dict[str, object]) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return f"{hashlib.sha256(encoded).hexdigest()}.json"


def _registry_request(
    tmp_path: Path,
) -> tuple[EvaluateRequest, dict[str, object], Path]:
    repo = tmp_path / "task-repo"
    task_dir = repo / "tasks" / "toy"
    task_dir.mkdir(parents=True)
    (task_dir / "task_card.json").write_text(
        '{"vm":{"snapshot":"test-image"}}\n',
        encoding="utf-8",
    )
    (task_dir / "evaluator.py").write_text("VERSION = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-qb", "main", str(repo)], check=True)
    evaluator_version = _commit(repo, "ready evaluator")
    (task_dir / "evaluator.py").write_text("VERSION = 2\n", encoding="utf-8")
    main_commit = _commit(repo, "later task change")
    assert evaluator_version != main_commit

    record = {
        "schema_version": 1,
        "status": "ready",
        "task_path": "toy",
        "variant_index": 0,
        "task_commit": "a" * 40,
        "evaluator_id": "rubric",
        "evaluator_version": evaluator_version,
        "rubric_hash": "b" * 64,
        "reference_manifest_uri": "oss://trusted/reference/manifest.json",
        "reference_manifest_hash": "c" * 64,
        "evaluator_sdk_version": "1.2.3",
        "harbor_version": "0.20.0",
        "rewardkit_version": "0.4.0",
        "image_id": "m-image-123",
        "pull_request_url": "https://github.com/example/tasks/pull/42",
        "ci_run_id": "123456",
        "ready_at": datetime(2026, 7, 29, tzinfo=UTC).isoformat(),
    }
    request = EvaluateRequest(
        submission_id=uuid4(),
        runtime_spec_path=tmp_path / "runtime.yaml",
        task_repo=repo,
        task_path="toy",
        variant_index=0,
        task_commit="a" * 40,
        image_id="m-image-123",
        submission_root="oss://submissions",
        evaluator_id="rubric",
        evaluator_version=evaluator_version,
    )
    registry_root = tmp_path / "control-plane"
    registry_root.mkdir()
    identity = {
        field: getattr(request, field)
        for field in (
            "task_path",
            "variant_index",
            "task_commit",
            "evaluator_id",
            "evaluator_version",
            "image_id",
        )
    }
    record_path = registry_root / _registry_record_filename(identity)
    record_path.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return request, record, registry_root


def test_registry_gate_validates_ready_record_and_exact_main_ancestry(tmp_path: Path) -> None:
    request, record_payload, registry_root = _registry_request(tmp_path)

    record = validate_evaluator_registry(request, registry_root)

    assert record.status == "ready"
    assert record.task_path == record_payload["task_path"]
    assert record.evaluator_version == record_payload["evaluator_version"]
    assert record.rubric_hash == record_payload["rubric_hash"]
    assert record.ready_at == datetime(2026, 7, 29, tzinfo=UTC)


@pytest.mark.parametrize(
    "failure",
    [
        "missing_record",
        "record_symlink",
        "record_directory",
        "oversized",
        "invalid_schema",
        "identity_mismatch",
        "dirty",
        "not_ancestor",
    ],
)
def test_registry_gate_fails_closed_for_untrusted_state_before_provider(
    tmp_path: Path,
    failure: str,
) -> None:
    request, _record, registry_root = _registry_request(tmp_path)
    record_path = next(registry_root.iterdir())
    if failure == "missing_record":
        record_path.unlink()
    elif failure == "record_symlink":
        outside = tmp_path / "self-signed.json"
        record_path.rename(outside)
        record_path.symlink_to(outside)
    elif failure == "record_directory":
        record_path.unlink()
        record_path.mkdir()
    elif failure == "oversized":
        record_path.write_bytes(b" " * (1024 * 1024 + 1))
    elif failure == "invalid_schema":
        record_path.write_text("{}", encoding="utf-8")
    elif failure == "identity_mismatch":
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        payload["evaluator_id"] = "different-evaluator"
        record_path.write_text(json.dumps(payload), encoding="utf-8")
    elif failure == "dirty":
        (request.task_repo / "untracked.txt").write_text("dirty", encoding="utf-8")
    else:
        side = request.task_repo.parent / "side"
        subprocess.run(
            ["git", "-C", str(request.task_repo), "worktree", "add", "-qb", "side", str(side)],
            check=True,
        )
        try:
            side_task = side / "tasks" / "toy" / "evaluator.py"
            side_task.write_text("VERSION = 99\n", encoding="utf-8")
            unrelated = _commit(side, "unrelated evaluator")
        finally:
            subprocess.run(
                ["git", "-C", str(request.task_repo), "worktree", "remove", str(side)],
                check=True,
            )
        request = request.model_copy(update={"evaluator_version": unrelated})

    with pytest.raises(AtomicInfrastructureError) as caught:
        validate_evaluator_registry(request, registry_root)

    assert caught.value.category == "evaluator_registry"


def test_evaluator_code_is_materialized_from_exact_commit_not_local_main(tmp_path: Path) -> None:
    request, _record, registry_root = _registry_request(tmp_path)
    validate_evaluator_registry(request, registry_root)

    with materialize_evaluator_checkout(request) as checkout:
        assert checkout != request.task_repo
        assert (checkout / "tasks" / "toy" / "evaluator.py").read_text(
            encoding="utf-8"
        ) == "VERSION = 1\n"
        assert not (checkout / ".git").exists()


@pytest.mark.parametrize("invalid_root", ["relative", "missing", "file", "symlink"])
def test_registry_gate_rejects_invalid_authoritative_root(
    tmp_path: Path,
    invalid_root: str,
) -> None:
    request, _record, registry_root = _registry_request(tmp_path)
    if invalid_root == "relative":
        root = Path("relative-registry")
    elif invalid_root == "missing":
        root = tmp_path / "missing"
    elif invalid_root == "file":
        root = tmp_path / "registry-file"
        root.write_text("not a directory", encoding="utf-8")
    else:
        target = tmp_path / "registry-target"
        registry_root.rename(target)
        registry_root.symlink_to(target, target_is_directory=True)
        root = registry_root

    with pytest.raises(AtomicInfrastructureError) as caught:
        validate_evaluator_registry(request, root)

    assert caught.value.category == "evaluator_registry"


def test_registry_gate_requires_configured_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, _record, _registry_root = _registry_request(tmp_path)
    monkeypatch.delenv("ALE_EVALUATOR_REGISTRY_ROOT", raising=False)

    with pytest.raises(AtomicInfrastructureError) as caught:
        validate_evaluator_registry(request)

    assert caught.value.category == "evaluator_registry"


def test_registry_gate_rejects_symlink_in_root_ancestor(tmp_path: Path) -> None:
    request, _record, registry_root = _registry_request(tmp_path)
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    registry_root.rename(real_parent / "control-plane")
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(AtomicInfrastructureError) as caught:
        validate_evaluator_registry(request, linked_parent / "control-plane")

    assert caught.value.category == "evaluator_registry"


def test_registry_gate_rejects_fifo_record_without_blocking(tmp_path: Path) -> None:
    request, _record, registry_root = _registry_request(tmp_path)
    record_path = next(registry_root.iterdir())
    record_path.unlink()
    os.mkfifo(record_path)
    errors: list[BaseException] = []

    def validate() -> None:
        try:
            validate_evaluator_registry(request, registry_root)
        except BaseException as exc:  # noqa: BLE001 - capture worker failure for assertion.
            errors.append(exc)

    worker = threading.Thread(target=validate, daemon=True)
    worker.start()
    worker.join(timeout=1)

    assert not worker.is_alive(), "FIFO registry record blocked validation"
    assert len(errors) == 1
    assert isinstance(errors[0], AtomicInfrastructureError)
    assert errors[0].category == "evaluator_registry"
