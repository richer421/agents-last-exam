"""RunWriter: owns one run's directory + events.jsonl + finalize files.

LOG_SPEC.md is the source of truth; this class is its only writer. Layout:

    <output_root>/<agent_id>/<model_slug>/<task_slug>/v<i>/<YYYYMMDD_HHMMSS>/
        events.jsonl       append-only, fsync per line
        run.json           schema_version=2, written once at finalize
        trajectory.json    ATIF-v1.0 from Trajectory.model_dump_json(indent=2)
        eval_result.json   {eval_status, score, eval_duration_s, error}
        origin_log/<agent_name>/    deployer work_dir pulled from VM
        output/                     agent output, when output_path="local"

The constructor refuses to overwrite an existing run dir
(``FileExistsError``). Each finalize write is wrapped in try/except; the
``events.jsonl`` is the authoritative trace even if one of the other writes
fails.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MAX_EVAL_DETAILS_BYTES = 256 * 1024
_SENSITIVE_DETAIL_KEY = re.compile(
    r"(?i)^(?:access[_-]?key(?:_id)?|api[_-]?key|authorization|credentials?|"
    r"password|secret|security[_-]?token|signature|token)$"
)
_SENSITIVE_QUERY_VALUE = re.compile(
    r"(?i)(AccessKeyId|Signature|SecurityToken|AccessKeySecret|api[_-]?key|"
    r"password|secret|token)=([^&\s\"']+)"
)
_AUTH_HEADER_VALUE = re.compile(r"(?i)(Authorization\s*:\s*(?:Bearer|Basic)\s+)\S+")
_BARE_AUTH_VALUE = re.compile(r"(?i)\b(Bearer|Basic)\s+[^\s,;]+")
_COLON_SECRET_VALUE = re.compile(
    r"(?i)\b(token|secret|password|api[_-]?key|cookie)\s*:\s*[^\s,;]+"
)
_PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN [^-\r\n]*PRIVATE KEY-----.*?-----END [^-\r\n]*PRIVATE KEY-----",
    re.DOTALL,
)


def _redact_text(value: str) -> str:
    value = _SENSITIVE_QUERY_VALUE.sub(r"\1=[REDACTED]", value)
    value = _AUTH_HEADER_VALUE.sub(r"\1[REDACTED]", value)
    value = _BARE_AUTH_VALUE.sub(r"\1 [REDACTED]", value)
    value = _COLON_SECRET_VALUE.sub(r"\1: [REDACTED]", value)
    return _PRIVATE_KEY_BLOCK.sub("[REDACTED PRIVATE KEY]", value)


def _redact_detail_values(
    value: Any,
    *,
    seen: set[int] | None = None,
    depth: int = 0,
) -> Any:
    if depth > 64:
        raise ValueError("metadata nesting exceeds 64 levels")
    if isinstance(value, str):
        return _redact_text(value)
    if seen is None:
        seen = set()
    if isinstance(value, (dict, list, tuple)):
        identity = id(value)
        if identity in seen:
            raise ValueError("cyclic metadata")
        seen.add(identity)
    else:
        identity = None
    if isinstance(value, dict):
        result = {
            _redact_text(str(key)): (
                "[REDACTED]"
                if _SENSITIVE_DETAIL_KEY.search(str(key))
                else _redact_detail_values(item, seen=seen, depth=depth + 1)
            )
            for key, item in value.items()
        }
    elif isinstance(value, (list, tuple)):
        result = [
            _redact_detail_values(item, seen=seen, depth=depth + 1) for item in value
        ]
    elif value is None or isinstance(value, (bool, int, float)):
        result = value
    else:
        result = _redact_text(str(value))
    if identity is not None:
        seen.remove(identity)
    return result


def _bounded_eval_details(details: dict[str, Any] | None) -> dict[str, Any] | None:
    if details is None:
        return None
    try:
        redacted = _redact_detail_values(details)
        encoded = json.dumps(
            redacted, ensure_ascii=False, default=str, separators=(",", ":")
        ).encode("utf-8")
    except Exception as exc:  # noqa: BLE001 - persistence must never break finalize
        return {"_omitted": f"evaluator details were not serializable: {type(exc).__name__}"}
    if len(encoded) > _MAX_EVAL_DETAILS_BYTES:
        return {
            "_omitted": f"evaluator details exceeded {_MAX_EVAL_DETAILS_BYTES} bytes",
            "_bytes": len(encoded),
        }
    return redacted


def _bounded_eval_error(error: dict[str, Any] | None) -> dict[str, Any] | None:
    if error is None:
        return None
    try:
        redacted = _redact_detail_values(error)
        encoded = json.dumps(
            redacted, ensure_ascii=False, default=str, separators=(",", ":")
        ).encode("utf-8")
    except Exception as exc:  # noqa: BLE001 - persistence must never break finalize
        return {"type": type(exc).__name__, "message": "evaluation error metadata omitted"}
    if len(encoded) > _MAX_EVAL_DETAILS_BYTES:
        return {
            "type": "OversizedMetadata",
            "message": f"evaluation error exceeded {_MAX_EVAL_DETAILS_BYTES} bytes",
        }
    return redacted


def sanitize_terminal_metadata(value: Any) -> Any:
    """Redact JSON terminal metadata without allowing diagnostics to break writes."""
    try:
        return _redact_detail_values(value)
    except Exception as exc:  # noqa: BLE001
        return {"_omitted": f"terminal metadata was not serializable: {type(exc).__name__}"}


def sanitize_evaluation_log(value: str) -> str:
    return _redact_text(value)

# ----------------------------------------------------------------- slugs

_SLUG_RE = re.compile(r"[^a-z0-9-]+")


def slug_model(model: str) -> str:
    if not model:
        return "unknown-model"
    s = model.lower().replace(".", "-").replace("/", "-").replace("_", "-")
    s = _SLUG_RE.sub("-", s).strip("-")
    return s or "unknown-model"


def slug_task(task_path: str) -> str:
    return task_path.strip("/").replace("/", "__")


def slug_agent(agent_name: str) -> str:
    s = (agent_name or "unknown").lower().replace("-", "_")
    return re.sub(r"[^a-z0-9_]+", "_", s).strip("_") or "unknown"


def build_run_id(*, agent_id: str, model: str, task_path: str, variant_index: int, ts: str) -> str:
    return (
        f"{slug_agent(agent_id)}__{slug_model(model)}__"
        f"{slug_task(task_path)}__v{variant_index}__{ts}"
    )


class RunWriter:
    def __init__(
        self,
        *,
        output_root: Path,
        agent_id: str,
        model: str,
        task_path: str,
        variant_index: int,
    ):
        self._ts = (
            f"{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}-"
            f"{uuid.uuid4().hex[:8]}"
        )
        self._slug_agent = slug_agent(agent_id)
        self._slug_model = slug_model(model)
        self._slug_task = slug_task(task_path)
        self._variant_index = variant_index

        self._run_dir = (
            output_root
            / self._slug_agent
            / self._slug_model
            / self._slug_task
            / f"v{variant_index}"
            / self._ts
        )
        # Refuse to overwrite — LOG_SPEC §1 collision policy.
        if self._run_dir.exists():
            raise FileExistsError(f"run dir already exists: {self._run_dir}")
        self._run_dir.mkdir(parents=True, exist_ok=False)
        (self._run_dir / "origin_log").mkdir(parents=True, exist_ok=True)
        (self._run_dir / "output").mkdir(parents=True, exist_ok=True)

        self._run_id = build_run_id(
            agent_id=agent_id,
            model=model,
            task_path=task_path,
            variant_index=variant_index,
            ts=self._ts,
        )

        self._events_path = self._run_dir / "events.jsonl"
        # Line-buffered append; fsync after each write for SIGTERM safety.
        self._events_fh = self._events_path.open("a", buffering=1, encoding="utf-8")

    # ------------------------------------------------------------------ props

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    @property
    def ts(self) -> str:
        return self._ts

    # ----------------------------------------------------------------- events

    def emit_event(self, event_type: str, **data: Any) -> None:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "type": event_type,
            "run_id": self._run_id,
        }
        if data:
            payload["data"] = sanitize_terminal_metadata(data)
        line = json.dumps(payload, default=str, ensure_ascii=False)
        try:
            self._events_fh.write(line + "\n")
            self._events_fh.flush()
            os.fsync(self._events_fh.fileno())
        except (OSError, ValueError) as e:
            # ValueError when fh is closed; OSError on fsync against a bad fd.
            logger.warning("emit_event(%s) failed: %s", event_type, e)

    # --------------------------------------------------------------- finalize

    def write_run_json(self, meta: dict[str, Any]) -> None:
        path = self._run_dir / "run.json"
        try:
            path.write_text(
                json.dumps(
                    sanitize_terminal_metadata(meta),
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                ),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning("write_run_json failed: %s", e)

    def write_trajectory(self, traj: Any) -> None:
        path = self._run_dir / "trajectory.json"
        try:
            blob = traj.model_dump_json(indent=2)
        except AttributeError:
            blob = json.dumps(traj, indent=2, ensure_ascii=False, default=str)
        try:
            path.write_text(blob, encoding="utf-8")
        except OSError as e:
            logger.warning("write_trajectory failed: %s", e)

    def write_eval_result(
        self,
        *,
        eval_status: str,
        score: float | None,
        eval_duration_s: float | None,
        error: dict[str, Any] | None,
        details: dict[str, Any] | None = None,
    ) -> None:
        path = self._run_dir / "eval_result.json"
        payload = {
            "eval_status": eval_status,
            "score": score,
            "eval_duration_s": eval_duration_s,
            "error": _bounded_eval_error(error),
            "details": _bounded_eval_details(details),
        }
        try:
            path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning("write_eval_result failed: %s", e)

    def close(self) -> None:
        try:
            self._events_fh.close()
        except OSError:
            pass
