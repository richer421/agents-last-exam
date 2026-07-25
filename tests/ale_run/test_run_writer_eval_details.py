from __future__ import annotations

import json

from ale_run.orchestration.run_writer import RunWriter


def test_eval_result_preserves_structured_evaluator_details(tmp_path) -> None:
    writer = RunWriter(
        output_root=tmp_path,
        agent_id="agent",
        model="model",
        task_path="task",
        variant_index=0,
    )
    writer.write_eval_result(
        eval_status="success",
        score=0.8,
        eval_duration_s=1.5,
        error=None,
        details={"report": {"alpha_similarity": 0.9}},
    )

    payload = json.loads((writer.run_dir / "eval_result.json").read_text())
    assert payload["details"] == {"report": {"alpha_similarity": 0.9}}
    writer.close()


def test_eval_result_redacts_sensitive_detail_keys(tmp_path) -> None:
    writer = RunWriter(
        output_root=tmp_path,
        agent_id="agent",
        model="model",
        task_path="task",
        variant_index=0,
    )
    writer.write_eval_result(
        eval_status="failed",
        score=None,
        eval_duration_s=1.5,
        error=None,
        details={"report": {"api_key": "secret", "metric": 0.7}},
    )

    payload = json.loads((writer.run_dir / "eval_result.json").read_text())
    assert payload["details"] == {
        "report": {"api_key": "[REDACTED]", "metric": 0.7}
    }
    writer.close()


def test_eval_result_redacts_credentials_inside_text_values(tmp_path) -> None:
    writer = RunWriter(
        output_root=tmp_path,
        agent_id="agent",
        model="model",
        task_path="task",
        variant_index=0,
    )
    writer.write_eval_result(
        eval_status="failed",
        score=None,
        eval_duration_s=1.5,
        error={"message": "Authorization: Bearer bearer-secret"},
        details={
            "diagnostic": "https://ecs.invalid/?AccessKeyId=abc&Signature=xyz"
        },
    )

    payload = json.loads((writer.run_dir / "eval_result.json").read_text())
    serialized = json.dumps(payload)
    assert "bearer-secret" not in serialized
    assert "abc" not in serialized
    assert "xyz" not in serialized
    writer.close()


def test_eval_result_omits_cyclic_details_without_interrupting_finalize(tmp_path) -> None:
    writer = RunWriter(
        output_root=tmp_path,
        agent_id="agent",
        model="model",
        task_path="task",
        variant_index=0,
    )
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic

    writer.write_eval_result(
        eval_status="failed",
        score=None,
        eval_duration_s=1.5,
        error=None,
        details=cyclic,
    )

    payload = json.loads((writer.run_dir / "eval_result.json").read_text())
    assert payload["details"]["_omitted"].startswith(
        "evaluator details were not serializable"
    )
    writer.close()


def test_all_terminal_json_writes_redact_credentials_without_redacting_metrics(
    tmp_path,
) -> None:
    writer = RunWriter(
        output_root=tmp_path,
        agent_id="agent",
        model="model",
        task_path="task",
        variant_index=0,
    )
    writer.emit_event("failed", message="Bearer bare-secret")
    writer.write_run_json(
        {"error": "token: colon-secret", "usage": {"total_tokens": 123}}
    )

    events = (writer.run_dir / "events.jsonl").read_text()
    run = json.loads((writer.run_dir / "run.json").read_text())
    assert "bare-secret" not in events
    assert "colon-secret" not in json.dumps(run)
    assert run["usage"]["total_tokens"] == 123
    writer.close()


def test_terminal_writer_omits_objects_with_broken_string_conversion(tmp_path) -> None:
    class Broken:
        def __str__(self) -> str:
            raise RuntimeError("do not stringify me")

    writer = RunWriter(
        output_root=tmp_path,
        agent_id="agent",
        model="model",
        task_path="task",
        variant_index=0,
    )
    writer.write_eval_result(
        eval_status="failed",
        score=None,
        eval_duration_s=1.0,
        error=None,
        details={"value": Broken()},
    )
    payload = json.loads((writer.run_dir / "eval_result.json").read_text())
    assert payload["details"]["_omitted"].startswith(
        "evaluator details were not serializable"
    )
    writer.close()


def test_eval_result_omits_oversized_details(tmp_path) -> None:
    writer = RunWriter(
        output_root=tmp_path,
        agent_id="agent",
        model="model",
        task_path="task",
        variant_index=0,
    )
    writer.write_eval_result(
        eval_status="success",
        score=0.8,
        eval_duration_s=1.5,
        error=None,
        details={"report": {"diagnostic": "x" * 300_000}},
    )

    payload = json.loads((writer.run_dir / "eval_result.json").read_text())
    assert payload["details"]["_omitted"] == "evaluator details exceeded 262144 bytes"
    assert payload["details"]["_bytes"] > 262144
    assert (writer.run_dir / "eval_result.json").stat().st_size < 4096
    writer.close()
