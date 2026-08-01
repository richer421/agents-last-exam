from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from ale_run.atomic import (
    AuthorEvaluatorRegistryIdentity,
    AuthorEvaluatorRequest,
    AuthorEvaluatorResult,
    RubricPlan,
)

COMMIT = "a" * 40
RUBRIC_HASH = "b" * 64
REFERENCE_MANIFEST_HASH = "c" * 64


def rubric_plan() -> RubricPlan:
    return RubricPlan(
        rubric_uri="oss://ale-rubrics/example/rubric.json",
        rubric_hash=RUBRIC_HASH,
        reference_manifest_uri="oss://ale-references/example/manifest.json",
        reference_manifest_hash=REFERENCE_MANIFEST_HASH,
    )


def authoring_request() -> AuthorEvaluatorRequest:
    return AuthorEvaluatorRequest(
        task_repository_url="https://github.com/openai/ale-tasks",
        task_path="tasks/demo/example",
        task_commit=COMMIT,
        image_id="m-authoring-image",
        evaluator_sdk_version="1.2.3",
        rubric_plan=rubric_plan(),
        max_retries=2,
        timeout_seconds=600,
    )


def test_authoring_request_is_frozen_and_rejects_unrelated_inputs() -> None:
    request = authoring_request()

    with pytest.raises(ValidationError):
        AuthorEvaluatorRequest(**(request.model_dump() | {"unexpected": "value"}))
    with pytest.raises(ValidationError):
        request.task_path = "tasks/demo/other"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_repository_url", "http://github.com/openai/ale-tasks"),
        ("task_repository_url", "https://example.com/openai/ale-tasks"),
        ("task_path", "tasks/../private"),
        ("max_retries", -1),
        ("max_retries", 6),
        ("timeout_seconds", 0),
        ("timeout_seconds", 3_601),
    ],
)
def test_authoring_request_validates_task_and_execution_bounds(
    field: str, value: str | int
) -> None:
    payload = authoring_request().model_dump() | {field: value}

    with pytest.raises(ValidationError):
        AuthorEvaluatorRequest(**payload)


def test_rubric_plan_accepts_only_oss_uris_and_hashes() -> None:
    plan = rubric_plan()

    assert plan.rubric_uri.startswith("oss://")
    with pytest.raises(ValidationError):
        RubricPlan(**(plan.model_dump() | {"rubric_uri": "s3://rubrics/plan.json"}))
    with pytest.raises(ValidationError):
        RubricPlan(**(plan.model_dump() | {"reference_manifest_hash": "not-a-hash"}))


def test_registry_identity_uses_only_the_versioned_identity_tuple() -> None:
    identity = AuthorEvaluatorRegistryIdentity(
        task_commit=COMMIT,
        rubric_hash=RUBRIC_HASH,
        reference_manifest_hash=REFERENCE_MANIFEST_HASH,
        evaluator_sdk_version="1.2.3",
        image_id="m-authoring-image",
    )

    assert identity.model_dump(exclude={"schema_version"}) == {
        "task_commit": COMMIT,
        "rubric_hash": RUBRIC_HASH,
        "reference_manifest_hash": REFERENCE_MANIFEST_HASH,
        "evaluator_sdk_version": "1.2.3",
        "image_id": "m-authoring-image",
    }


def test_authoring_result_enforces_ready_or_failed_consistency() -> None:
    identity = AuthorEvaluatorRegistryIdentity(
        task_commit=COMMIT,
        rubric_hash=RUBRIC_HASH,
        reference_manifest_hash=REFERENCE_MANIFEST_HASH,
        evaluator_sdk_version="1.2.3",
        image_id="m-authoring-image",
    )
    ready = AuthorEvaluatorResult(
        status="ready",
        registry_identity=identity,
        completed_at=datetime.now(UTC),
    )

    assert ready.status == "ready"
    with pytest.raises(ValidationError):
        AuthorEvaluatorResult(status="ready", completed_at=datetime.now(UTC))
    with pytest.raises(ValidationError):
        AuthorEvaluatorResult(
            status="authoring_failed",
            registry_identity=identity,
            error="generation failed",
            completed_at=datetime.now(UTC),
        )


def test_authoring_contracts_contain_no_submission_related_fields() -> None:
    models = (
        AuthorEvaluatorRequest,
        AuthorEvaluatorResult,
        RubricPlan,
        AuthorEvaluatorRegistryIdentity,
    )
    prohibited = {
        "submission_id",
        "submission_root",
        "candidate_artifacts",
        "agent_trajectory",
        "screenshots",
        "historical_scores",
        "bad_cases",
    }

    for model in models:
        assert prohibited.isdisjoint(model.model_fields)
