import hashlib
import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from ale_run.atomic.contracts import (
    ArtifactEntry,
    EvaluationResult,
    EvaluatorRegistryRecord,
    HarborProvenance,
    SubmissionManifest,
)


def _evaluation_identity() -> dict[str, object]:
    return {
        "submission_id": uuid4(),
        "task_path": "visual_media/demo",
        "variant_index": 0,
        "task_commit": "a" * 40,
        "image_id": "m-image-123",
        "evaluator_id": "rubric",
        "evaluator_version": "b" * 40,
    }


def test_submission_envelope_records_status_and_artifact_media_type() -> None:
    manifest = SubmissionManifest(
        submission_id=uuid4(),
        task_path="visual_media/demo",
        variant_index=0,
        task_commit="a" * 40,
        image_id="m-image-123",
        ale_run_id="run-123",
        agent_id="codex",
        model_id="gpt-5",
        config_digest="b" * 64,
        started_at=datetime(2026, 7, 29, tzinfo=UTC),
        completed_at=datetime(2026, 7, 29, 0, 1, tzinfo=UTC),
        artifacts=(
            ArtifactEntry(
                path="output/final.png",
                size_bytes=4,
                sha256="c" * 64,
                media_type="image/png",
            ),
        ),
    )

    assert manifest.status == "submitted"
    assert manifest.artifacts[0].media_type == "image/png"
    with pytest.raises(ValidationError):
        ArtifactEntry(path="output/final.png", size_bytes=4, sha256="c" * 64)


def test_evaluator_registry_record_is_complete_and_frozen(tmp_path) -> None:
    record = EvaluatorRegistryRecord(
        status="ready",
        task_path="visual_media/demo",
        variant_index=0,
        task_commit="a" * 40,
        evaluator_id="rubric",
        evaluator_version="b" * 40,
        rubric_hash="c" * 64,
        reference_manifest_uri="oss://trusted/reference/manifest.json",
        reference_manifest_hash="d" * 64,
        evaluator_sdk_version="1.2.3",
        harbor_version="0.20.0",
        rewardkit_version="0.4.0",
        image_id="m-image-123",
        pull_request_url="https://github.com/example/tasks/pull/42",
        ci_run_id="123456",
        ready_at=datetime(2026, 7, 29, tzinfo=UTC),
    )

    assert record.schema_version == 1
    with pytest.raises(ValidationError):
        record.status = "draft"
    with pytest.raises(ValidationError):
        EvaluatorRegistryRecord(**(record.model_dump() | {"status": "draft"}))


def test_scored_result_requires_complete_identity_and_harbor_provenance() -> None:
    identity = _evaluation_identity()
    result = EvaluationResult(
        **identity,
        status="scored",
        outcome="valid",
        score=0.82,
        rubric_hash="c" * 64,
        harbor=HarborProvenance(
            reward={"rubric": 0.82},
            reward_path="evidence/reward.json",
            reward_size_bytes=len(b'{"rubric":0.82}'),
            reward_sha256=hashlib.sha256(b'{"rubric":0.82}').hexdigest(),
            details_path="evidence/reward-details.json",
            details_size_bytes=2,
            details_sha256=hashlib.sha256(b"{}").hexdigest(),
        ),
    )

    assert result.score == 0.82
    with pytest.raises(ValidationError):
        EvaluationResult(
            **identity,
            status="scored",
            outcome="valid",
            score=0.82,
            rubric_hash="c" * 64,
        )
    with pytest.raises(ValidationError):
        EvaluationResult(
            **identity,
            status="scored",
            outcome="invalid_output",
            score=0.01,
            rubric_hash="c" * 64,
            harbor=HarborProvenance(
                reward={"hard_gate": 0.0},
                reward_path="evidence/reward.json",
                reward_size_bytes=len(b'{"hard_gate":0.0}'),
                reward_sha256=hashlib.sha256(b'{"hard_gate":0.0}').hexdigest(),
                details_path="evidence/reward-details.json",
                details_size_bytes=2,
                details_sha256=hashlib.sha256(b"{}").hexdigest(),
            ),
        )


def test_harbor_provenance_requires_canonical_evidence_paths_and_digests() -> None:
    digest = hashlib.sha256(b"{}").hexdigest()
    provenance = HarborProvenance(
        reward={"reward": 0.82},
        reward_path="evidence/reward.json",
        reward_size_bytes=2,
        reward_sha256=digest,
        details_path="evidence/reward-details.json",
        details_size_bytes=2,
        details_sha256=digest,
    )

    assert provenance.reward_path == "evidence/reward.json"
    assert provenance.reward_size_bytes == 2
    assert provenance.details_path == "evidence/reward-details.json"
    with pytest.raises(ValidationError):
        HarborProvenance(
            reward={"reward": 0.82},
            reward_path="evidence/other.json",
            reward_size_bytes=2,
            reward_sha256=digest,
            details_path="evidence/reward-details.json",
            details_size_bytes=2,
            details_sha256=digest,
        )
    with pytest.raises(ValidationError):
        HarborProvenance(
            reward={},
            reward_path="evidence/reward.json",
            reward_size_bytes=2,
            reward_sha256=digest,
            details_path="evidence/reward-details.json",
            details_size_bytes=2,
            details_sha256=digest,
        )
    with pytest.raises(ValidationError):
        HarborProvenance(
            reward={"reward": 0.82},
            reward_path="evidence/reward.json",
            reward_size_bytes=2,
            reward_sha256="not-a-digest",
            details_path="evidence/reward-details.json",
            details_size_bytes=2,
            details_sha256=digest,
        )
    with pytest.raises(ValidationError):
        HarborProvenance(
            reward={"reward": 0.82},
            reward_path="evidence/reward.json",
            reward_size_bytes=True,
            reward_sha256=digest,
            details_path="evidence/reward-details.json",
            details_size_bytes=2,
            details_sha256=digest,
        )


def test_infra_failed_result_requires_bounded_error_and_omits_score_fields() -> None:
    identity = _evaluation_identity()
    result = EvaluationResult(
        **identity,
        status="infra_failed",
        error_category="reference",
        error_detail="reference hash mismatch",
        attempt_id="attempt-123",
    )

    payload = json.loads(result.model_dump_json())
    assert payload["error_category"] == "reference"
    assert "outcome" not in payload
    assert "score" not in payload
    with pytest.raises(ValidationError):
        EvaluationResult(**identity, status="infra_failed")
    with pytest.raises(ValidationError):
        EvaluationResult(
            **identity,
            status="infra_failed",
            error_category="reference",
            error_detail="x" * 4_001,
            attempt_id="attempt-123",
        )
