"""Independent one-shot evaluate capability."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from ..environments import task_data as task_data_pkg
from ..executors.sandbox import SandboxExecutor, _ale_src_root_for
from ..executors.sandbox_evaluator import evaluate_in_sandbox
from ..orchestration.lifecycle import _EVAL_TIMEOUT_S, _task_data_source
from .contracts import (
    AtomicInfrastructureError,
    EvaluateRequest,
    EvaluationResult,
    EvaluatorRegistryRecord,
    HarborProvenance,
    SubmissionManifest,
)
from .evaluator_registry import (
    materialize_evaluator_checkout,
    validate_evaluator_registry,
)
from .oss_submission import (
    _evaluator_segment,
    _submission_prefix,
    _task_card_path,
    _validate_manifest_identity,
    download_range_to_local,
    publish_evaluation_result,
    stage_submission,
)
from .runtime import AtomicRuntime

logger = logging.getLogger(__name__)

_MAX_HOST_OSS_OUTPUT_BYTES = 64 * 1024
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_RESULT_BYTES = 64 * 1024
_MAX_TASK_CARD_BYTES = 1024 * 1024
_MAX_REWARD_EVIDENCE_BYTES = 32 * 1024
_MAX_DETAILS_EVIDENCE_BYTES = 8 * 1024 * 1024
_MAX_HARBOR_EVIDENCE_BYTES = 8 * 1024 * 1024
_EVIDENCE_DOWNLOAD_TIMEOUT_S = 120
_HOST_OSS_TIMEOUT_S = 300
_CANONICAL_RESULT = object()
_EVALUATOR_ENV_KEYS = frozenset(
    {
        "CHROMA_SOFT_EVAL_MODEL",
        "GCLOUD_PROJECT",
        "GCP_PROJECT",
        "GEMINI_API_KEY",
        "GEMINI_AUTH_MODE",
        "GEMINI_EVAL_MODEL",
        "GEMINI_MODEL",
        "GOOGLE_API_KEY",
        "GOOGLE_CLOUD_LOCATION",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_REGION",
        "GOOGLE_VERTEX_LOCATION",
        "LLM_JUDGE_MODEL",
        "LLM_JUDGE_WIRE_API",
        "OPENAI_API_BASE",
        "OPENAI_API_KEY",
        "OPENROUTER_API_KEY",
        "VIDEO_STORYBOARD_JUDGE_MODEL",
    }
)


async def evaluate(request: EvaluateRequest) -> EvaluationResult:
    """Evaluate one committed submission in a fresh immutable environment."""
    attempt_id = uuid4().hex
    try:
        registry_record = validate_evaluator_registry(request)
        expected_manifest = await _read_submission_manifest(request)
        _validate_submission_manifest(expected_manifest, request)

        existing = await _read_existing_evaluation_result(request, registry_record)
        if existing is not None:
            return existing

        committed_result: EvaluationResult | None = None
        try:
            with materialize_evaluator_checkout(request) as evaluator_checkout:
                runtime_request = request.model_copy(update={"task_repo": evaluator_checkout})
                async with AtomicRuntime.open(
                    request=runtime_request,
                    run_setup=False,
                    evaluator_registry_record=registry_record,
                ) as runtime:
                    try:
                        staged_manifest = await stage_submission(
                            runtime.env.sandbox,
                            runtime.task_data,
                            runtime_request,
                        )
                    except AtomicInfrastructureError:
                        raise
                    except Exception as exc:
                        raise AtomicInfrastructureError(
                            "submission_integrity",
                            f"submission staging failed: {type(exc).__name__}: {exc}",
                        ) from exc
                    if staged_manifest != expected_manifest:
                        raise AtomicInfrastructureError(
                            "submission_integrity",
                            "submission manifest changed between control-plane preflight and VM staging",
                        )

                    sandbox = runtime.env.sandbox
                    sep = "/" if sandbox.is_linux else "\\"
                    executor = SandboxExecutor(
                        config=None,
                        work_dir=(
                            f"{sandbox.work_dir_base.rstrip(sep)}{sep}evaluate{sep}{attempt_id}"
                        ),
                        sandbox=sandbox,
                        env={},
                    )
                    try:
                        await executor.stage_runtime()
                    except Exception as exc:
                        raise AtomicInfrastructureError(
                            "transport",
                            f"ALE runtime staging failed: {type(exc).__name__}: {exc}",
                        ) from exc

                    with tempfile.TemporaryDirectory(
                        prefix="ale-evaluate-evidence-"
                    ) as evidence_dir:
                        try:
                            evaluated = await evaluate_in_sandbox(
                                sandbox=sandbox,
                                ale_src_root=_ale_src_root_for(sandbox),
                                task_path=runtime.task_dir,
                                variant=request.variant_index,
                                timeout_s=float(_EVAL_TIMEOUT_S),
                                evaluator_env=_evaluator_environment(),
                            )
                            raw_result = evaluated.result
                            if raw_result.get("error"):
                                raise AtomicInfrastructureError(
                                    "evaluator",
                                    str(raw_result["error"]),
                                )
                            score, outcome, hard_gate_passed = _validate_raw_evaluator_result(
                                raw_result
                            )
                            harbor, evidence_paths = await _stage_harbor_evidence(
                                sandbox,
                                request,
                                raw_result,
                                outcome,
                                Path(evidence_dir),
                            )
                            result = EvaluationResult(
                                **_evaluation_identity(request),
                                status="scored",
                                outcome=outcome,
                                score=score,
                                rubric_hash=registry_record.rubric_hash,
                                harbor=harbor,
                            )
                            _validate_scored_result_state(
                                result,
                                hard_gate_passed=hard_gate_passed,
                                category="evaluator",
                            )
                        except AtomicInfrastructureError:
                            raise
                        except Exception as exc:
                            raise AtomicInfrastructureError(
                                "evaluator",
                                f"{type(exc).__name__}: {exc}",
                            ) from exc

                        try:

                            def mark_result_committed() -> None:
                                nonlocal committed_result
                                committed_result = result

                            await publish_evaluation_result(
                                request,
                                result,
                                evidence_paths,
                                on_result_committed=mark_result_committed,
                            )
                        except AtomicInfrastructureError:
                            raise
                        except Exception as exc:
                            raise AtomicInfrastructureError(
                                "submission_storage",
                                f"result publication failed: {type(exc).__name__}: {exc}",
                            ) from exc
                        committed_result = result
        except Exception as exc:
            if committed_result is None:
                raise
            cleanup_failure = AtomicInfrastructureError(
                "cleanup",
                f"runtime cleanup failed after result commit: {type(exc).__name__}: {exc}",
            )
            try:
                await _record_cleanup_diagnostic(
                    request,
                    attempt_id,
                    cleanup_failure,
                )
            except Exception:
                logger.exception(
                    "failed to publish atomic evaluate cleanup diagnostic attempt=%s",
                    attempt_id,
                )
            return committed_result
        assert committed_result is not None
        return committed_result
    except Exception as exc:  # noqa: BLE001
        failure = (
            exc
            if isinstance(exc, AtomicInfrastructureError)
            else AtomicInfrastructureError(
                "runtime",
                f"{type(exc).__name__}: {exc}",
            )
        )
        try:
            await _record_attempt_diagnostic(request, attempt_id, failure)
        except Exception:
            logger.exception(
                "failed to publish atomic evaluate diagnostic attempt=%s category=%s",
                attempt_id,
                failure.category,
            )
        return EvaluationResult(
            **_evaluation_identity(request),
            status="infra_failed",
            error_category=failure.category,
            error_detail=(failure.message[:4000] or failure.category),
            attempt_id=attempt_id,
        )


def _evaluation_identity(request: EvaluateRequest) -> dict[str, object]:
    return {
        "submission_id": request.submission_id,
        "task_path": request.task_path,
        "variant_index": request.variant_index,
        "task_commit": request.task_commit,
        "image_id": request.image_id,
        "evaluator_id": request.evaluator_id,
        "evaluator_version": request.evaluator_version,
    }


def _validate_submission_manifest(
    manifest: SubmissionManifest,
    request: EvaluateRequest,
) -> None:
    _validate_manifest_identity(manifest, request)
    provenance = {
        "ale_run_id": manifest.ale_run_id,
        "agent_id": manifest.agent_id,
        "model_id": manifest.model_id,
        "config_digest": manifest.config_digest,
    }
    missing = [field for field, value in provenance.items() if not value]
    if missing:
        raise AtomicInfrastructureError(
            "submission_integrity",
            f"submission manifest has empty solve provenance: {', '.join(missing)}",
        )
    if len(manifest.config_digest) != 64 or any(
        character not in "0123456789abcdef" for character in manifest.config_digest
    ):
        raise AtomicInfrastructureError(
            "submission_integrity",
            "submission manifest config_digest is not a lowercase SHA-256",
        )
    if manifest.completed_at < manifest.started_at:
        raise AtomicInfrastructureError(
            "submission_integrity",
            "submission manifest completed_at precedes started_at",
        )


def _validate_scored_result_state(
    result: EvaluationResult,
    *,
    hard_gate_passed: object = _CANONICAL_RESULT,
    category: str,
) -> None:
    invalid_output = result.outcome == "invalid_output"
    hard_gate_failed = hard_gate_passed is False
    if invalid_output:
        valid = result.score == 0.0 and (hard_gate_passed is _CANONICAL_RESULT or hard_gate_failed)
    else:
        valid = hard_gate_passed is _CANONICAL_RESULT or not hard_gate_failed
    if not valid:
        raise AtomicInfrastructureError(
            category,
            "illegal scored result state: invalid_output requires an exact "
            "0.0 score and evaluator hard-gate failure",
        )


def _validate_raw_evaluator_result(
    raw_result: object,
) -> tuple[float, str, bool | object]:
    if not isinstance(raw_result, dict):
        raise AtomicInfrastructureError("evaluator", "evaluator result must be a JSON object")
    score = raw_result.get("score")
    if type(score) not in (int, float) or not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise AtomicInfrastructureError(
            "evaluator", "evaluator result score must be a finite number from 0 to 1"
        )
    outcome = raw_result.get("outcome")
    if type(outcome) is not str or outcome not in {"valid", "invalid_output"}:
        raise AtomicInfrastructureError(
            "evaluator", "evaluator result outcome must be valid or invalid_output"
        )
    report = raw_result.get("report", _CANONICAL_RESULT)
    if report is not _CANONICAL_RESULT and not isinstance(report, dict):
        raise AtomicInfrastructureError(
            "evaluator", "evaluator result report must be a JSON object"
        )
    hard_gate_passed: bool | object = _CANONICAL_RESULT
    if isinstance(report, dict) and "hard_gate_passed" in report:
        hard_gate_passed = report["hard_gate_passed"]
        if type(hard_gate_passed) is not bool:
            raise AtomicInfrastructureError(
                "evaluator", "evaluator hard_gate_passed must be a boolean"
            )
    if outcome == "invalid_output":
        if score != 0.0 or hard_gate_passed is not False:
            raise AtomicInfrastructureError(
                "evaluator",
                "invalid_output requires an exact 0.0 score and hard_gate_passed false",
            )
    elif hard_gate_passed is False:
        raise AtomicInfrastructureError(
            "evaluator", "valid outcome cannot have hard_gate_passed false"
        )
    return float(score), outcome, hard_gate_passed


async def _stage_harbor_evidence(
    sandbox: Any,
    request: EvaluateRequest,
    raw_result: dict[str, Any],
    outcome: str,
    staging_dir: Path,
) -> tuple[HarborProvenance, dict[str, Path]]:
    staging_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    reward_path = staging_dir / "reward.json"
    details_path = staging_dir / "reward-details.json"
    try:
        await download_range_to_local(
            sandbox,
            "/logs/verifier/reward.json",
            reward_path,
            max_bytes=_MAX_REWARD_EVIDENCE_BYTES,
            timeout=_EVIDENCE_DOWNLOAD_TIMEOUT_S,
        )
    except Exception as exc:
        raise AtomicInfrastructureError(
            "evaluator",
            f"reward.json download failed: {type(exc).__name__}: {exc}",
        ) from exc
    reward_size = reward_path.stat().st_size
    details_limit = min(
        _MAX_DETAILS_EVIDENCE_BYTES,
        _MAX_HARBOR_EVIDENCE_BYTES - reward_size,
    )
    try:
        await download_range_to_local(
            sandbox,
            "/logs/verifier/reward-details.json",
            details_path,
            max_bytes=details_limit,
            timeout=_EVIDENCE_DOWNLOAD_TIMEOUT_S,
        )
    except Exception as exc:
        raise AtomicInfrastructureError(
            "evaluator",
            f"reward-details.json download failed: {type(exc).__name__}: {exc}",
        ) from exc

    reward_bytes = reward_path.read_bytes()
    details_bytes = details_path.read_bytes()
    if len(reward_bytes) > _MAX_REWARD_EVIDENCE_BYTES:
        raise AtomicInfrastructureError("evaluator", "reward.json exceeds its 32 KiB limit")
    if len(details_bytes) > _MAX_DETAILS_EVIDENCE_BYTES:
        raise AtomicInfrastructureError("evaluator", "reward-details.json exceeds its 8 MiB limit")
    if len(reward_bytes) + len(details_bytes) > _MAX_HARBOR_EVIDENCE_BYTES:
        raise AtomicInfrastructureError(
            "evaluator", "Harbor evidence exceeds its combined 8 MiB limit"
        )

    reward = _parse_evidence_object(reward_bytes, name="reward.json")
    details = _parse_evidence_object(details_bytes, name="reward-details.json")
    if not reward:
        raise AtomicInfrastructureError(
            "evaluator", "reward.json must contain a non-empty JSON object"
        )
    _validate_evidence_identity(reward, request, name="reward.json")
    _validate_evidence_identity(details, request, name="reward-details.json")
    _validate_evidence_protocol(
        reward,
        raw_result,
        outcome,
        name="reward.json",
    )
    _validate_evidence_protocol(
        details,
        raw_result,
        outcome,
        name="reward-details.json",
    )

    reward_sha256 = hashlib.sha256(reward_bytes).hexdigest()
    details_sha256 = hashlib.sha256(details_bytes).hexdigest()
    if reward_path.stat().st_size != len(reward_bytes):
        raise AtomicInfrastructureError("evaluator", "reward.json size changed after staging")
    if details_path.stat().st_size != len(details_bytes):
        raise AtomicInfrastructureError(
            "evaluator", "reward-details.json size changed after staging"
        )
    if hashlib.sha256(reward_path.read_bytes()).hexdigest() != reward_sha256:
        raise AtomicInfrastructureError("evaluator", "reward.json changed after staging")
    if hashlib.sha256(details_path.read_bytes()).hexdigest() != details_sha256:
        raise AtomicInfrastructureError("evaluator", "reward-details.json changed after staging")

    return (
        HarborProvenance(
            reward=reward,
            reward_path="evidence/reward.json",
            reward_size_bytes=len(reward_bytes),
            reward_sha256=reward_sha256,
            details_path="evidence/reward-details.json",
            details_size_bytes=len(details_bytes),
            details_sha256=details_sha256,
        ),
        {"reward.json": reward_path, "reward-details.json": details_path},
    )


def _parse_evidence_object(raw: bytes, *, name: str) -> dict[str, Any]:
    return _parse_strict_json_object(raw, name=name, category="evaluator")


def _parse_strict_json_object(
    raw: bytes,
    *,
    name: str,
    category: str,
) -> dict[str, Any]:
    def reject_non_finite(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        value = json.loads(raw, parse_constant=reject_non_finite)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AtomicInfrastructureError(
            category,
            f"{name} must be one complete finite JSON object: {exc}",
        ) from exc
    if not isinstance(value, dict):
        raise AtomicInfrastructureError(category, f"{name} must be a JSON object")
    return value


def _validate_evidence_identity(
    evidence: dict[str, Any],
    request: EvaluateRequest,
    *,
    name: str,
    category: str = "evaluator",
) -> None:
    expected = {
        "submission_id": str(request.submission_id),
        "task_path": request.task_path,
        "variant_index": request.variant_index,
        "task_commit": request.task_commit,
        "image_id": request.image_id,
        "evaluator_id": request.evaluator_id,
        "evaluator_version": request.evaluator_version,
    }
    mismatches = []
    for field, expected_value in expected.items():
        if field not in evidence:
            continue
        actual = evidence[field]
        expected_type = int if field == "variant_index" else str
        if type(actual) is not expected_type or actual != expected_value:
            mismatches.append(field)
    if mismatches:
        raise AtomicInfrastructureError(
            category, f"{name} identity mismatch: {', '.join(mismatches)}"
        )


def _validate_evidence_protocol(
    evidence: dict[str, Any],
    raw_result: dict[str, Any],
    outcome: str,
    *,
    name: str,
    category: str = "evaluator",
) -> None:
    if "score" in evidence:
        score = evidence["score"]
        if type(score) not in (int, float) or not math.isfinite(score):
            raise AtomicInfrastructureError(category, f"{name} score must be a finite number")
        if score != raw_result["score"]:
            raise AtomicInfrastructureError(
                category, f"{name} score mismatch with evaluator result"
            )
    if "outcome" in evidence and (
        type(evidence["outcome"]) is not str or evidence["outcome"] != outcome
    ):
        raise AtomicInfrastructureError(category, f"{name} outcome mismatch with evaluator result")

    raw_report = raw_result.get("report")
    expected_hard_gate = outcome != "invalid_output"
    if isinstance(raw_report, dict) and "hard_gate_passed" in raw_report:
        expected_hard_gate = raw_report["hard_gate_passed"]
    evidence_hard_gates: list[object] = []
    if "hard_gate_passed" in evidence:
        evidence_hard_gates.append(evidence["hard_gate_passed"])
    if "report" in evidence:
        evidence_report = evidence["report"]
        if not isinstance(evidence_report, dict):
            raise AtomicInfrastructureError(category, f"{name} report must be a JSON object")
        if "hard_gate_passed" in evidence_report:
            evidence_hard_gates.append(evidence_report["hard_gate_passed"])
    if any(
        type(value) is not bool or value is not expected_hard_gate for value in evidence_hard_gates
    ):
        raise AtomicInfrastructureError(
            category, f"{name} hard-gate mismatch with evaluator result"
        )


async def _stage_reference_strict(runtime: Any, request: EvaluateRequest) -> None:
    reference_files = _declared_reference_files(request)
    task_data = runtime.task_data
    if task_data is None or not task_data.requires_task_data:
        if reference_files:
            raise AtomicInfrastructureError(
                "reference",
                "task declares referenceFiles but no task-data staging is configured",
            )
        return

    try:
        source = _task_data_source(runtime.runtime_spec.artifacts)
        backend = task_data_pkg.select(source)
        report = await backend.stage_reference(
            runtime.env.sandbox,
            task_data,
            source=source,
        )
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"[:2000]
        raise AtomicInfrastructureError(
            "reference",
            f"reference staging failed: {detail}",
        ) from exc
    if not isinstance(report, dict):
        raise AtomicInfrastructureError(
            "reference",
            "reference backend returned a non-object report",
        )
    if reference_files and report.get("skipped"):
        raise AtomicInfrastructureError(
            "reference",
            f"declared referenceFiles were not staged: {report.get('reason', 'unknown')}",
        )


def _declared_reference_files(request: EvaluateRequest) -> tuple[str, ...]:
    task_card = _task_card_path(request)
    try:
        with task_card.open("rb") as stream:
            raw = stream.read(_MAX_TASK_CARD_BYTES + 1)
    except OSError as exc:
        raise AtomicInfrastructureError(
            "reference",
            f"cannot read task_card.json referenceFiles: {exc}",
        ) from exc
    if len(raw) > _MAX_TASK_CARD_BYTES:
        raise AtomicInfrastructureError(
            "reference",
            "task_card.json exceeds the 1 MiB limit",
        )
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AtomicInfrastructureError(
            "reference",
            f"invalid task_card.json: {exc}",
        ) from exc
    reference_files = payload.get("referenceFiles", []) if isinstance(payload, dict) else []
    if not isinstance(reference_files, list):
        raise AtomicInfrastructureError(
            "reference",
            "task_card.json referenceFiles must be a list",
        )
    declared: list[str] = []
    for entry in reference_files:
        path = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(path, str) or not path:
            raise AtomicInfrastructureError(
                "reference",
                "each referenceFiles entry must have a non-empty string path",
            )
        declared.append(path)
    return tuple(declared)


def _evaluator_environment() -> dict[str, str]:
    return {key: os.environ[key] for key in sorted(_EVALUATOR_ENV_KEYS) if key in os.environ}


async def _read_submission_manifest(request: EvaluateRequest) -> SubmissionManifest:
    raw = await _read_oss_object(
        f"{_submission_prefix(request)}/manifest.json",
        limit=_MAX_MANIFEST_BYTES,
        missing_ok=False,
    )
    assert raw is not None
    try:
        return SubmissionManifest.model_validate_json(raw)
    except ValidationError as exc:
        raise AtomicInfrastructureError(
            "submission_integrity",
            f"invalid submission manifest: {exc}",
        ) from exc


async def _read_existing_evaluation_result(
    request: EvaluateRequest,
    registry_record: EvaluatorRegistryRecord,
) -> EvaluationResult | None:
    raw = await _read_oss_object(
        _evaluation_prefix(request) + "/result.json",
        limit=_MAX_RESULT_BYTES,
        missing_ok=True,
    )
    if raw is None:
        return None
    try:
        payload = _parse_strict_json_object(
            raw,
            name="result.json",
            category="idempotency_conflict",
        )
        result = EvaluationResult.model_validate(payload)
    except ValidationError as exc:
        raise AtomicInfrastructureError(
            "idempotency_conflict",
            f"existing result.json is invalid: {exc}",
        ) from exc
    if result.status != "scored":
        raise AtomicInfrastructureError(
            "idempotency_conflict",
            "canonical result.json contains a non-scored result",
        )
    _validate_scored_result_state(result, category="idempotency_conflict")
    expected = {
        **_evaluation_identity(request),
        "rubric_hash": registry_record.rubric_hash,
    }
    mismatches = [field for field, value in expected.items() if getattr(result, field) != value]
    if mismatches:
        raise AtomicInfrastructureError(
            "idempotency_conflict",
            f"existing result identity mismatch: {', '.join(mismatches)}",
        )
    try:
        canonical_result = _canonical_json(result.model_dump(mode="json"))
    except ValueError as exc:
        raise AtomicInfrastructureError(
            "idempotency_conflict",
            f"existing result.json contains non-finite JSON: {exc}",
        ) from exc
    if raw != canonical_result:
        raise AtomicInfrastructureError(
            "idempotency_conflict",
            "existing result.json is not canonical",
        )
    await _verify_existing_harbor_evidence(request, result)
    return result


async def _verify_existing_harbor_evidence(
    request: EvaluateRequest,
    result: EvaluationResult,
) -> None:
    harbor = result.harbor
    assert harbor is not None
    if harbor.reward_size_bytes > _MAX_REWARD_EVIDENCE_BYTES:
        raise AtomicInfrastructureError(
            "idempotency_conflict",
            "reward.json declared size exceeds its 32 KiB limit",
        )
    if harbor.details_size_bytes > _MAX_DETAILS_EVIDENCE_BYTES:
        raise AtomicInfrastructureError(
            "idempotency_conflict",
            "reward-details.json declared size exceeds its 8 MiB limit",
        )
    if harbor.reward_size_bytes + harbor.details_size_bytes > _MAX_HARBOR_EVIDENCE_BYTES:
        raise AtomicInfrastructureError(
            "idempotency_conflict",
            "canonical Harbor evidence declared size exceeds its combined 8 MiB limit",
        )

    prefix = _evaluation_prefix(request)
    evidence_specs = (
        (
            "reward.json",
            harbor.reward_size_bytes,
            harbor.reward_sha256,
        ),
        (
            "reward-details.json",
            harbor.details_size_bytes,
            harbor.details_sha256,
        ),
    )
    evidence: dict[str, dict[str, Any]] = {}
    for name, expected_size, expected_sha256 in evidence_specs:
        try:
            raw = await _read_oss_object(
                f"{prefix}/evidence/{name}",
                limit=expected_size,
                missing_ok=False,
            )
        except AtomicInfrastructureError as exc:
            raise AtomicInfrastructureError(
                "idempotency_conflict",
                f"{name} verification failed: {exc.message}",
            ) from exc
        if raw is None:
            raise AtomicInfrastructureError(
                "idempotency_conflict",
                f"{name} is missing",
            )
        if len(raw) != expected_size:
            raise AtomicInfrastructureError(
                "idempotency_conflict",
                f"{name} size mismatch with result provenance",
            )
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise AtomicInfrastructureError(
                "idempotency_conflict",
                f"{name} SHA-256 mismatch with result provenance",
            )
        evidence[name] = _parse_strict_json_object(
            raw,
            name=name,
            category="idempotency_conflict",
        )

    reward = evidence["reward.json"]
    try:
        reward_matches = _canonical_json(reward) == _canonical_json(harbor.reward)
    except ValueError as exc:
        raise AtomicInfrastructureError(
            "idempotency_conflict",
            f"reward.json contains non-finite JSON: {exc}",
        ) from exc
    if not reward_matches:
        raise AtomicInfrastructureError(
            "idempotency_conflict",
            "reward.json does not match embedded Harbor reward",
        )

    raw_result = {"score": result.score, "outcome": result.outcome}
    for name, value in evidence.items():
        _validate_evidence_identity(
            value,
            request,
            name=name,
            category="idempotency_conflict",
        )
        _validate_evidence_protocol(
            value,
            raw_result,
            result.outcome,
            name=name,
            category="idempotency_conflict",
        )


async def _record_attempt_diagnostic(
    request: EvaluateRequest,
    attempt_id: str,
    error: AtomicInfrastructureError,
) -> None:
    await _record_diagnostic(
        request,
        attempt_id,
        error,
        object_name=f"{attempt_id}.json",
        phase="evaluate",
        status="infra_failed",
    )


async def _record_cleanup_diagnostic(
    request: EvaluateRequest,
    attempt_id: str,
    error: AtomicInfrastructureError,
) -> None:
    await _record_diagnostic(
        request,
        attempt_id,
        error,
        object_name=f"{attempt_id}/cleanup.json",
        phase="cleanup",
        status="scored_cleanup_failed",
    )


async def _record_diagnostic(
    request: EvaluateRequest,
    attempt_id: str,
    error: AtomicInfrastructureError,
    *,
    object_name: str,
    phase: str,
    status: str,
) -> None:
    payload = _canonical_json(
        {
            "attempt_id": attempt_id,
            "category": error.category,
            "created_at": datetime.now(UTC).isoformat(),
            "error": error.message[:4000],
            "evaluator_id": request.evaluator_id,
            "evaluator_version": request.evaluator_version,
            "phase": phase,
            "schema_version": 1,
            "status": status,
            "submission_id": str(request.submission_id),
        }
    )
    if len(payload) > _MAX_RESULT_BYTES:
        raise AtomicInfrastructureError(
            "diagnostic",
            "attempt diagnostic exceeds the 64 KiB limit",
        )
    with tempfile.TemporaryDirectory(prefix="ale-evaluate-diagnostic-") as temp_dir:
        path = Path(temp_dir) / "attempt.json"
        path.write_bytes(payload)
        result = await _run_host_ossutil(
            "cp",
            str(path),
            f"{_evaluation_prefix(request)}/evidence/attempts/{object_name}",
            "-f",
        )
    if result[0] != 0:
        raise AtomicInfrastructureError(
            "diagnostic",
            f"attempt diagnostic upload failed: {_host_command_diagnostic(result)}",
        )


async def _read_oss_object(
    url: str,
    *,
    limit: int,
    missing_ok: bool,
) -> bytes | None:
    stat_result = await _run_host_ossutil("stat", url)
    if stat_result[0] != 0:
        diagnostic = _host_command_diagnostic(stat_result)
        if _is_missing_object(diagnostic):
            if missing_ok:
                return None
            raise AtomicInfrastructureError(
                "submission_integrity",
                f"required OSS object is missing: {url}",
            )
        raise AtomicInfrastructureError(
            "submission_storage",
            f"cannot stat OSS object {url}: {diagnostic}",
        )
    stat_output = stat_result[1].decode("utf-8", errors="replace")
    size_match = re.search(
        r"(?im)^\s*(?:content[- ]?length|size)\s*[:=]\s*(\d+)\s*$",
        stat_output,
    )
    if size_match is None:
        raise AtomicInfrastructureError(
            "submission_storage",
            f"OSS stat omitted a trustworthy content length: {url}",
        )
    expected_size = int(size_match.group(1))
    if expected_size > limit:
        raise AtomicInfrastructureError(
            "submission_integrity",
            f"OSS object exceeds {limit} bytes: {url}",
        )

    with tempfile.TemporaryDirectory(prefix="ale-evaluate-read-") as temp_dir:
        path = Path(temp_dir) / "object.json"
        result = await _run_host_ossutil("cp", url, str(path), "-f")
        if result[0] != 0:
            diagnostic = _host_command_diagnostic(result)
            raise AtomicInfrastructureError(
                "submission_storage",
                f"cannot read OSS object {url}: {diagnostic}",
            )
        try:
            size = path.stat().st_size
            if size > limit:
                raise AtomicInfrastructureError(
                    "submission_integrity",
                    f"OSS object exceeds {limit} bytes: {url}",
                )
            if size != expected_size:
                raise AtomicInfrastructureError(
                    "submission_integrity",
                    f"OSS object size changed after stat: {url}",
                )
            return path.read_bytes()
        except AtomicInfrastructureError:
            raise
        except OSError as exc:
            raise AtomicInfrastructureError(
                "submission_storage",
                f"cannot read downloaded OSS object {url}: {exc}",
            ) from exc


async def _run_host_ossutil(*arguments: str) -> tuple[int, bytes, bytes]:
    try:
        process = await asyncio.create_subprocess_exec(
            "ossutil",
            *arguments,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        raise AtomicInfrastructureError(
            "submission_storage",
            f"cannot start host ossutil: {exc}",
        ) from exc

    assert process.stdout is not None
    assert process.stderr is not None
    stdout_task = asyncio.create_task(_drain_bounded(process.stdout))
    stderr_task = asyncio.create_task(_drain_bounded(process.stderr))
    try:
        returncode = await asyncio.wait_for(
            process.wait(),
            timeout=_HOST_OSS_TIMEOUT_S,
        )
    except TimeoutError as exc:
        process.kill()
        await process.wait()
        await asyncio.gather(stdout_task, stderr_task)
        raise AtomicInfrastructureError(
            "submission_storage",
            f"host ossutil exceeded {_HOST_OSS_TIMEOUT_S} seconds",
        ) from exc
    except BaseException:
        process.kill()
        await process.wait()
        await asyncio.gather(stdout_task, stderr_task)
        raise

    stdout, stdout_exceeded = await stdout_task
    stderr, stderr_exceeded = await stderr_task
    if stdout_exceeded or stderr_exceeded:
        raise AtomicInfrastructureError(
            "submission_storage",
            "host ossutil output exceeded 64 KiB",
        )
    return returncode, stdout, stderr


async def _drain_bounded(
    stream: asyncio.StreamReader,
) -> tuple[bytes, bool]:
    kept = bytearray()
    exceeded = False
    while chunk := await stream.read(64 * 1024):
        remaining = _MAX_HOST_OSS_OUTPUT_BYTES - len(kept)
        if remaining > 0:
            kept.extend(chunk[:remaining])
        if len(chunk) > remaining:
            exceeded = True
    return bytes(kept), exceeded


def _host_command_diagnostic(result: tuple[int, bytes, bytes]) -> str:
    output = result[2] or result[1]
    if not output:
        return f"ossutil exited {result[0]}"
    return output.decode("utf-8", errors="replace").strip()[:1000]


def _is_missing_object(diagnostic: str) -> bool:
    lowered = diagnostic.lower()
    return any(
        marker in lowered
        for marker in (
            "nosuchkey",
            "nosuchobject",
            "not found",
            "status=404",
            "status: 404",
            "statuscode=404",
        )
    )


def _evaluation_prefix(request: EvaluateRequest) -> str:
    return (
        f"{_submission_prefix(request)}/evaluations/"
        f"{_evaluator_segment(request.evaluator_id)}/{request.evaluator_version}"
    )


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
