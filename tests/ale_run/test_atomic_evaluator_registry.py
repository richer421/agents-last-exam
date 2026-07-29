from __future__ import annotations

import hashlib
import json
import subprocess
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


def _registry_request(tmp_path: Path) -> tuple[EvaluateRequest, dict[str, object]]:
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
    record_bytes = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    record_path = tmp_path / "control-plane" / "evaluator.json"
    record_path.parent.mkdir()
    record_path.write_bytes(record_bytes)
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
        evaluator_registry_record_path=record_path,
        evaluator_registry_record_sha256=hashlib.sha256(record_bytes).hexdigest(),
    )
    return request, record


def test_registry_gate_validates_ready_record_and_exact_main_ancestry(tmp_path: Path) -> None:
    request, record_payload = _registry_request(tmp_path)

    record = validate_evaluator_registry(request)

    assert record.status == "ready"
    assert record.task_path == record_payload["task_path"]
    assert record.evaluator_version == record_payload["evaluator_version"]
    assert record.rubric_hash == record_payload["rubric_hash"]
    assert record.ready_at == datetime(2026, 7, 29, tzinfo=UTC)


@pytest.mark.parametrize("failure", ["hash", "inside_checkout", "dirty", "not_ancestor"])
def test_registry_gate_fails_closed_for_untrusted_state_before_provider(
    tmp_path: Path,
    failure: str,
) -> None:
    request, _record = _registry_request(tmp_path)
    if failure == "hash":
        request = request.model_copy(update={"evaluator_registry_record_sha256": "f" * 64})
    elif failure == "inside_checkout":
        inside = request.task_repo / "registry.json"
        inside.write_bytes(request.evaluator_registry_record_path.read_bytes())
        request = request.model_copy(update={"evaluator_registry_record_path": inside})
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
        validate_evaluator_registry(request)

    assert caught.value.category == "evaluator_registry"


def test_evaluator_code_is_materialized_from_exact_commit_not_local_main(tmp_path: Path) -> None:
    request, _record = _registry_request(tmp_path)
    validate_evaluator_registry(request)

    with materialize_evaluator_checkout(request) as checkout:
        assert checkout != request.task_repo
        assert (checkout / "tasks" / "toy" / "evaluator.py").read_text(
            encoding="utf-8"
        ) == "VERSION = 1\n"
        assert not (checkout / ".git").exists()
