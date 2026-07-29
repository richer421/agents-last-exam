"""Independent one-shot evaluate capability."""

from __future__ import annotations

import asyncio
import json
import logging
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
    SubmissionManifest,
)
from .oss_submission import (
    _evaluator_segment,
    _submission_prefix,
    _task_card_path,
    _validate_manifest_identity,
    publish_evaluation_result,
    stage_submission,
)
from .runtime import AtomicRuntime

logger = logging.getLogger(__name__)

_MAX_HOST_OSS_OUTPUT_BYTES = 64 * 1024
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_RESULT_BYTES = 64 * 1024
_MAX_TASK_CARD_BYTES = 1024 * 1024
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
        expected_manifest = await _read_submission_manifest(request)
        _validate_submission_manifest(expected_manifest, request)

        existing = await _read_existing_evaluation_result(request)
        if existing is not None:
            return existing

        committed_result: EvaluationResult | None = None
        try:
            async with AtomicRuntime.open(request=request, run_setup=False) as runtime:
                await _stage_reference_strict(runtime, request)

                try:
                    staged_manifest = await stage_submission(
                        runtime.env.sandbox,
                        runtime.task_data,
                        request,
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
                    work_dir=(f"{sandbox.work_dir_base.rstrip(sep)}{sep}evaluate{sep}{attempt_id}"),
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
                    report = raw_result.get("report") or {}
                    outcome = (
                        "invalid_output"
                        if report.get("hard_gate_passed") is False
                        else str(raw_result.get("outcome") or "valid")
                    )
                    result = EvaluationResult(
                        status="scored",
                        outcome=outcome,
                        score=float(raw_result["score"]),
                    )
                    _validate_scored_result_state(
                        result,
                        hard_gate_passed=report.get("hard_gate_passed"),
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
                    await publish_evaluation_result(sandbox, request, result)
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
        return EvaluationResult(status="infra_failed")


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
) -> EvaluationResult | None:
    raw = await _read_oss_object(
        _evaluation_prefix(request) + "/result.json",
        limit=_MAX_RESULT_BYTES,
        missing_ok=True,
    )
    if raw is None:
        return None
    try:
        result = EvaluationResult.model_validate_json(raw)
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
    if raw != _canonical_json(result.model_dump(mode="json")):
        raise AtomicInfrastructureError(
            "idempotency_conflict",
            "existing result.json is not canonical",
        )
    return result


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
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
