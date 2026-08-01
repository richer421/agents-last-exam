from uuid import UUID

import pytest
from pydantic import ValidationError

from ale_run.atomic import (
    AuthorEvaluatorRegistryIdentity,
    AuthorEvaluatorRequest,
    AuthorEvaluatorResult,
    RubricPlan,
    RubricPlanEntry,
)

COMMIT = "a" * 40
RUBRIC_HASH = "b" * 64
REFERENCE_MANIFEST_HASH = "c" * 64


def rubric_plan() -> RubricPlan:
    return RubricPlan(
        rubric_hash=RUBRIC_HASH,
        items=(
            RubricPlanEntry(
                rubric_id="visual_quality",
                implementation_mode="llm_judge",
                weight=0.75,
                score_min=0,
                score_max=1,
                required=True,
                rationale="Requires semantic visual comparison.",
            ),
        ),
    )


def authoring_request() -> AuthorEvaluatorRequest:
    return AuthorEvaluatorRequest(
        authoring_id=UUID("00000000-0000-0000-0000-000000000001"),
        task_repository_url="https://github.com/openai/ale-tasks",
        task_path="tasks/demo/example",
        variant_index=0,
        task_commit=COMMIT,
        image_id="m-authoring-image",
        evaluator_sdk_version="1.2.3",
        rubric_uri="oss://ale-rubrics/example/rubric.json",
        rubric_hash=RUBRIC_HASH,
        reference_manifest_uri="oss://ale-references/example/manifest.json",
        reference_manifest_hash=REFERENCE_MANIFEST_HASH,
        evaluator_id="rubric",
        max_retries=2,
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
        ("timeout_seconds", 18_001),
    ],
)
def test_authoring_request_validates_task_and_execution_bounds(
    field: str, value: str | int
) -> None:
    payload = authoring_request().model_dump() | {field: value}

    with pytest.raises(ValidationError):
        AuthorEvaluatorRequest(**payload)


def test_authoring_request_defaults_to_five_hour_capability_budget() -> None:
    assert authoring_request().timeout_seconds == 18_000


def test_authoring_request_carries_authoritative_uris_and_hashes() -> None:
    request = authoring_request()

    assert request.rubric_uri.startswith("oss://")
    with pytest.raises(ValidationError):
        AuthorEvaluatorRequest(**(request.model_dump() | {"rubric_uri": "s3://rubrics/plan.json"}))
    with pytest.raises(ValidationError):
        AuthorEvaluatorRequest(**(request.model_dump() | {"reference_manifest_hash": "not-a-hash"}))


def test_rubric_plan_requires_unique_entries_with_valid_ranges() -> None:
    plan = rubric_plan()

    assert plan.items[0].rubric_id == "visual_quality"
    with pytest.raises(ValidationError):
        RubricPlan(
            rubric_hash=RUBRIC_HASH,
            items=(plan.items[0], plan.items[0]),
        )
    with pytest.raises(ValidationError):
        RubricPlanEntry(**(plan.items[0].model_dump() | {"score_min": 2, "score_max": 1}))


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
    authoring_id = UUID("00000000-0000-0000-0000-000000000001")
    ready = AuthorEvaluatorResult(
        authoring_id=authoring_id,
        status="ready",
        evaluator_id="rubric",
        evaluator_version="d" * 40,
        pull_request_url="https://github.com/openai/ale-tasks/pull/123",
        ci_run_id="987654",
    )

    assert ready.status == "ready"
    with pytest.raises(ValidationError):
        AuthorEvaluatorResult(
            authoring_id=authoring_id,
            status="ready",
            evaluator_id="rubric",
        )
    with pytest.raises(ValidationError):
        AuthorEvaluatorResult(
            authoring_id=authoring_id,
            status="authoring_failed",
            evaluator_id="rubric",
            evaluator_version="d" * 40,
            error_category="authoring",
            error_detail="generation failed",
        )


def test_authoring_failure_requires_bounded_structured_error() -> None:
    failed = AuthorEvaluatorResult(
        authoring_id=UUID("00000000-0000-0000-0000-000000000001"),
        status="authoring_failed",
        evaluator_id="rubric",
        error_category="authoring",
        error_detail="generation failed",
    )

    assert failed.error_category == "authoring"
    with pytest.raises(ValidationError):
        AuthorEvaluatorResult(**(failed.model_dump() | {"error_detail": "x" * 4_001}))


def test_authoring_contracts_contain_no_submission_related_fields() -> None:
    models = (
        AuthorEvaluatorRequest,
        AuthorEvaluatorResult,
        RubricPlan,
        RubricPlanEntry,
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
