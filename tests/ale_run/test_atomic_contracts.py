from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from ale_run.atomic.contracts import (
    ArtifactEntry,
    AtomicInfrastructureError,
    EvaluateRequest,
    EvaluationResult,
    HarborProvenance,
    SolveRequest,
    SolveResult,
    SubmissionManifest,
)


@pytest.mark.parametrize("request_type", [SolveRequest, EvaluateRequest])
@pytest.mark.parametrize(
    "task_path",
    [
        "",
        "/absolute/task",
        "../task",
        "domain/../task",
        "domain//task",
        "domain/./task",
        r"domain\task",
    ],
)
def test_atomic_requests_reject_unsafe_task_paths_at_envelope(
    tmp_path,
    request_type,
    task_path,
):
    payload = {
        "submission_id": uuid4(),
        "runtime_spec_path": tmp_path / "exp.yaml",
        "task_repo": tmp_path,
        "task_path": task_path,
        "variant_index": 0,
        "task_commit": "a" * 40,
        "image_id": "m-image-123",
        "submission_root": "oss://bucket",
    }
    if request_type is SolveRequest:
        payload["agent_id"] = "codex"
    else:
        payload["evaluator_id"] = "rubric"
        payload["evaluator_version"] = "b" * 40

    with pytest.raises(ValidationError):
        request_type(**payload)


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


def test_evaluate_request_has_no_request_selected_registry_authority(tmp_path):
    payload = {
        "submission_id": uuid4(),
        "runtime_spec_path": tmp_path / "exp.yaml",
        "task_repo": tmp_path,
        "task_path": "visual_media/demo",
        "variant_index": 0,
        "task_commit": "a" * 40,
        "image_id": "m-image-123",
        "submission_root": "oss://bucket",
        "evaluator_id": "rubric",
        "evaluator_version": "b" * 40,
    }

    request = EvaluateRequest(**payload)

    assert "evaluator_registry_record_path" not in type(request).model_fields
    assert "evaluator_registry_record_sha256" not in type(request).model_fields
    with pytest.raises(ValidationError):
        EvaluateRequest(
            **payload,
            evaluator_registry_record_path=tmp_path / "self-signed.json",
            evaluator_registry_record_sha256="c" * 64,
        )


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
    artifact = ArtifactEntry(
        path="output/final.png",
        size_bytes=1024,
        sha256="a" * 64,
        media_type="image/png",
    )
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


def test_artifact_entry_requires_a_sha256_and_non_negative_size():
    entry = ArtifactEntry(
        path="output/final.png",
        size_bytes=0,
        sha256="a" * 64,
        media_type="image/png",
    )
    assert entry.size_bytes == 0

    with pytest.raises(ValidationError):
        ArtifactEntry(
            path="output/final.png",
            size_bytes=1,
            sha256="",
            media_type="image/png",
        )
    with pytest.raises(ValidationError):
        ArtifactEntry(
            path="output/final.png",
            size_bytes=1,
            sha256="g" * 64,
            media_type="image/png",
        )
    with pytest.raises(ValidationError):
        ArtifactEntry(
            path="output/final.png",
            size_bytes=-1,
            sha256="a" * 64,
            media_type="image/png",
        )


def test_evaluation_result_enforces_score_and_infrastructure_failure_shape():
    identity = {
        "submission_id": uuid4(),
        "task_path": "visual_media/demo",
        "variant_index": 0,
        "task_commit": "a" * 40,
        "image_id": "m-image-123",
        "evaluator_id": "rubric",
        "evaluator_version": "b" * 40,
    }
    scored = {
        **identity,
        "status": "scored",
        "outcome": "valid",
        "score": 1.0,
        "rubric_hash": "c" * 64,
        "harbor": HarborProvenance(
            reward={"score": 1.0},
            details_path="evidence/reward-details.json",
        ),
    }
    infra = {
        **identity,
        "status": "infra_failed",
        "error_category": "runtime",
        "error_detail": "unavailable",
        "attempt_id": "attempt-1",
    }
    result = EvaluationResult(**scored)

    assert result.score == 1.0
    with pytest.raises(ValidationError):
        EvaluationResult(**(scored | {"score": 1.1}))
    with pytest.raises(ValidationError):
        EvaluationResult(**(infra | {"outcome": "valid"}))
    with pytest.raises(ValidationError):
        EvaluationResult(**(infra | {"score": 0.0}))
    with pytest.raises(ValidationError):
        EvaluationResult(**(scored | {"outcome": None}))
    with pytest.raises(ValidationError):
        EvaluationResult(**(scored | {"score": None}))


def test_solve_result_requires_a_manifest_for_submitted_status():
    submission_id = uuid4()
    manifest = SubmissionManifest(
        submission_id=submission_id,
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
                size_bytes=1,
                sha256="a" * 64,
                media_type="image/png",
            ),
        ),
    )
    result = SolveResult(status="submitted", submission_id=submission_id, manifest=manifest)

    assert result.schema_version == 1
    assert result.submission_id == submission_id
    assert result.manifest is manifest
    assert result.error is None
    with pytest.raises(ValidationError):
        SolveResult(status="submitted", submission_id=submission_id)
    with pytest.raises(ValidationError):
        SolveResult(
            status="submitted",
            submission_id=submission_id,
            manifest=manifest,
            error="unexpected error",
        )
    with pytest.raises(ValidationError):
        SolveResult(
            status="submitted",
            submission_id=submission_id,
            manifest=manifest.model_copy(update={"submission_id": uuid4()}),
        )


def test_solve_result_requires_an_error_for_failed_status():
    submission_id = uuid4()
    result = SolveResult(status="failed", submission_id=submission_id, error="agent timeout")

    assert result.manifest is None
    assert result.error == "agent timeout"
    with pytest.raises(ValidationError):
        SolveResult(status="failed", submission_id=submission_id)
    with pytest.raises(ValidationError):
        SolveResult(
            status="failed",
            submission_id=submission_id,
            manifest=SubmissionManifest(
                submission_id=submission_id,
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
                        size_bytes=1,
                        sha256="a" * 64,
                        media_type="image/png",
                    ),
                ),
            ),
            error="agent timeout",
        )


def test_infrastructure_error_is_catchable_and_preserves_category_and_message():
    error = AtomicInfrastructureError("storage", "storage unavailable")

    assert error.category == "storage"
    assert error.message == "storage unavailable"
    assert str(error) == "storage: storage unavailable"
    with pytest.raises(AtomicInfrastructureError) as caught:
        raise error
    assert caught.value is error
