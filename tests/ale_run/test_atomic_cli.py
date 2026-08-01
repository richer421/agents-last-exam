from __future__ import annotations

import json
import sys
from types import ModuleType
from uuid import UUID, uuid4

import pytest

from ale_run import cli
from ale_run.atomic.contracts import (
    AtomicInfrastructureError,
    AuthorEvaluatorResult,
    EvaluateRequest,
    EvaluationResult,
    SolveResult,
)
from ale_run.cli import main


class _BlockedModule(ModuleType):
    def __getattr__(self, name: str):
        raise AssertionError(f"unexpected import of {self.__name__}.{name}")


class _Request:
    authoring_id = UUID("00000000-0000-0000-0000-000000000002")
    submission_id = UUID("00000000-0000-0000-0000-000000000001")
    task_path = "visual_media/demo"
    variant_index = 0
    task_commit = "a" * 40
    image_id = "m-image-123"
    evaluator_id = "rubric"
    evaluator_version = "b" * 40

    @classmethod
    def model_validate_json(cls, raw: str) -> _Request:
        if raw == "malformed":
            raise ValueError("invalid request")
        return cls()


class _Result:
    def __init__(self, status: str) -> None:
        self.status = status

    def model_dump_json(self) -> str:
        return json.dumps({"status": self.status})


def _install_atomic_modules(
    monkeypatch: pytest.MonkeyPatch,
    *,
    command: str,
    status: str,
    error: Exception | None = None,
) -> list[_Request]:
    for name in (
        "ale_run.atomic",
        "ale_run.atomic.contracts",
        "ale_run.atomic.author_evaluator",
        "ale_run.atomic.solve",
        "ale_run.atomic.evaluate",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)

    calls: list[_Request] = []

    async def operation(request: _Request) -> _Result:
        calls.append(request)
        if error is not None:
            raise error
        return _Result(status)

    contracts = ModuleType("ale_run.atomic.contracts")
    contracts.AuthorEvaluatorRequest = _Request
    contracts.SolveRequest = _Request
    contracts.EvaluateRequest = _Request
    contracts.AuthorEvaluatorResult = AuthorEvaluatorResult
    contracts.SolveResult = SolveResult
    contracts.EvaluationResult = EvaluationResult
    implementation_name = command.replace("-", "_")
    implementation = ModuleType(f"ale_run.atomic.{implementation_name}")
    setattr(implementation, implementation_name, operation)

    monkeypatch.setitem(sys.modules, "ale_run.atomic.contracts", contracts)
    monkeypatch.setitem(
        sys.modules,
        f"ale_run.atomic.{implementation_name}",
        implementation,
    )
    for other in {"author_evaluator", "solve", "evaluate"} - {implementation_name}:
        monkeypatch.setitem(
            sys.modules,
            f"ale_run.atomic.{other}",
            _BlockedModule(f"ale_run.atomic.{other}"),
        )
    return calls


@pytest.mark.parametrize(
    ("command", "status", "expected_exit"),
    [
        ("solve", "submitted", 0),
        ("solve", "failed", 1),
        ("evaluate", "scored", 0),
        ("evaluate", "infra_failed", 1),
        ("author-evaluator", "ready", 0),
        ("author-evaluator", "authoring_failed", 1),
    ],
)
def test_atomic_command_dispatches_only_its_own_capability(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    status: str,
    expected_exit: int,
) -> None:
    calls = _install_atomic_modules(monkeypatch, command=command, status=status)
    request_path = tmp_path / "request.json"
    request_path.write_text("{}", encoding="utf-8")

    assert main([command, str(request_path)]) == expected_exit
    assert len(calls) == 1
    assert capsys.readouterr().out == json.dumps({"status": status}) + "\n"


@pytest.mark.parametrize("command", ["solve", "evaluate", "author-evaluator"])
def test_atomic_command_rejects_malformed_request(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
) -> None:
    calls = _install_atomic_modules(monkeypatch, command=command, status="submitted")
    request_path = tmp_path / "request.json"
    request_path.write_text("malformed", encoding="utf-8")

    assert main([command, str(request_path)]) == 2
    assert calls == []
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err


def test_evaluate_command_accepts_envelope_without_registry_authority(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[EvaluateRequest] = []

    async def evaluate(request: EvaluateRequest) -> _Result:
        calls.append(request)
        return _Result("scored")

    implementation = ModuleType("ale_run.atomic.evaluate")
    implementation.evaluate = evaluate
    monkeypatch.setitem(sys.modules, "ale_run.atomic.evaluate", implementation)
    request_path = tmp_path / "evaluate.json"
    request_path.write_text(
        json.dumps(
            {
                "submission_id": str(uuid4()),
                "runtime_spec_path": str(tmp_path / "runtime.yaml"),
                "task_repo": str(tmp_path / "task-repo"),
                "task_path": "toy",
                "variant_index": 0,
                "task_commit": "a" * 40,
                "image_id": "m-image-123",
                "submission_root": "oss://submissions",
                "evaluator_id": "rubric",
                "evaluator_version": "b" * 40,
            }
        ),
        encoding="utf-8",
    )

    assert main(["evaluate", str(request_path)]) == 0
    assert len(calls) == 1
    assert "evaluator_registry_record_path" not in calls[0].model_fields_set
    assert capsys.readouterr().out == json.dumps({"status": "scored"}) + "\n"


@pytest.mark.parametrize(
    ("command", "error", "expected"),
    [
        (
            "solve",
            AtomicInfrastructureError("runtime", "sandbox unavailable"),
            {"status": "failed"},
        ),
        ("solve", ValueError("invalid runtime input"), {"status": "failed"}),
        ("solve", RuntimeError("x" * 2_000), {"status": "failed"}),
        ("evaluate", RuntimeError("operation exploded"), {"status": "infra_failed"}),
        (
            "author-evaluator",
            RuntimeError("operation exploded"),
            {"status": "authoring_failed"},
        ),
    ],
)
def test_atomic_command_converts_operation_exception_to_result_json(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    command: str,
    error: Exception,
    expected: dict[str, str],
) -> None:
    calls = _install_atomic_modules(
        monkeypatch,
        command=command,
        status="submitted",
        error=error,
    )
    request_path = tmp_path / "request.json"
    request_path.write_text("{}", encoding="utf-8")

    assert main([command, str(request_path)]) == 1
    assert len(calls) == 1
    output = capsys.readouterr()
    result = json.loads(output.out)
    assert expected.items() <= result.items()
    assert output.out.count("\n") == 1
    assert "Traceback" not in output.err
    assert len(output.err) <= 1_100
    if command == "solve":
        assert result["submission_id"] == str(_Request.submission_id)
        assert 0 < len(result["error"]) <= 1_000
    elif command == "evaluate":
        assert result["submission_id"] == str(_Request.submission_id)
        assert result["task_path"] == _Request.task_path
        assert result["error_category"] == "cli"
        assert result["error_detail"] == "operation exploded"
        assert result["attempt_id"].startswith("cli-")
        assert {"outcome", "score", "rubric_hash", "harbor"}.isdisjoint(result)
    else:
        assert result["authoring_id"] == str(_Request.authoring_id)
        assert result["evaluator_id"] == _Request.evaluator_id
        assert result["error_category"] == "cli"
        assert result["error_detail"] == "operation exploded"
        assert {"evaluator_version", "pull_request_url", "ci_run_id"}.isdisjoint(result)


def test_existing_run_and_list_commands_keep_their_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    async def run_handler(_args) -> int:
        calls.append("run")
        return 7

    def list_handler(_args) -> int:
        calls.append("list")
        return 8

    monkeypatch.setattr(cli, "_cmd_run", run_handler)
    monkeypatch.setattr(cli, "_cmd_list", list_handler)

    assert main(["run", "experiment.yaml"]) == 7
    assert main(["list"]) == 8
    assert calls == ["run", "list"]
