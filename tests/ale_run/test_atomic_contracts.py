from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from ale_run.atomic.contracts import (
    ArtifactEntry,
    AtomicInfrastructureError,
    EvaluateRequest,
    EvaluationResult,
    SolveRequest,
    SolveResult,
    SubmissionManifest,
)


def test_solve_request_rejects_mutable_or_short_identity(tmp_path):
    with pytest.raises(ValidationError):
        SolveRequest(
            submission_id=uuid4(),
            runtime_spec_path=tmp_path / "exp.yaml",
            task_repo=tmp_path,
            task_path="visual_media/demo",
            variant_index=0,
            agent_id="codex",
            task_commit="main",
            image_id="adobe-v1",
            submission_root="oss://bucket",
        )


def test_evaluate_request_has_no_agent_configuration(tmp_path):
    request = EvaluateRequest(
        submission_id=uuid4(),
        runtime_spec_path=tmp_path / "exp.yaml",
        task_repo=tmp_path,
        task_path="visual_media/demo",
        variant_index=0,
        task_commit="a" * 40,
        image_id="m-image-123",
        submission_root="oss://bucket",
        evaluator_id="rubric",
        evaluator_version="b" * 40,
    )

    assert "agent_id" not in type(request).model_fields


def test_requests_require_versioned_immutable_aliyun_identity(tmp_path):
    request = SolveRequest(
        submission_id=uuid4(),
        runtime_spec_path=tmp_path / "exp.yaml",
        task_repo=tmp_path,
        task_path="visual_media/demo",
        variant_index=0,
        agent_id="codex",
        task_commit="a" * 40,
        image_id="m-image-123",
        submission_root="oss://bucket",
    )

    assert request.schema_version == 1
    assert request.variant_index == 0
    with pytest.raises(ValidationError):
        request.task_commit = "b" * 40
    with pytest.raises(ValidationError):
        SolveRequest(**(request.model_dump() | {"image_id": "image-123"}))
    with pytest.raises(ValidationError):
        SolveRequest(**(request.model_dump() | {"submission_root": "s3://bucket"}))
    with pytest.raises(ValidationError):
        SolveRequest(**(request.model_dump() | {"variant_index": -1}))


def test_submission_manifest_records_solve_provenance_and_artifacts_without_evaluator():
    artifact = ArtifactEntry(path="output/final.png", digest="a" * 64)
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
        artifacts=(artifact,),
    )

    assert manifest.artifacts == (artifact,)
    assert "evaluator_id" not in type(manifest).model_fields
    with pytest.raises(ValidationError):
        SubmissionManifest(**(manifest.model_dump() | {"artifacts": ()}))


def test_evaluation_result_enforces_score_and_infrastructure_failure_shape():
    result = EvaluationResult(status="scored", outcome="valid", score=1.0)

    assert result.score == 1.0
    with pytest.raises(ValidationError):
        EvaluationResult(status="scored", outcome="valid", score=1.1)
    with pytest.raises(ValidationError):
        EvaluationResult(status="infra_failed", outcome="valid", score=None)
    with pytest.raises(ValidationError):
        EvaluationResult(status="infra_failed", outcome=None, score=0.0)


def test_solve_result_and_infrastructure_error_are_versioned_contracts():
    submission_id = uuid4()
    result = SolveResult(submission_id=submission_id)
    error = AtomicInfrastructureError(message="storage unavailable")

    assert result.schema_version == 1
    assert result.submission_id == submission_id
    assert error.schema_version == 1
    assert error.message == "storage unavailable"
