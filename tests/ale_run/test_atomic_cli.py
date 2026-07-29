from __future__ import annotations

import json
import sys
from types import ModuleType

import pytest

from ale_run.cli import main


class _BlockedModule(ModuleType):
    def __getattr__(self, name: str):
        raise AssertionError(f"unexpected import of {self.__name__}.{name}")


class _Request:
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
) -> list[_Request]:
    for name in (
        "ale_run.atomic",
        "ale_run.atomic.contracts",
        "ale_run.atomic.solve",
        "ale_run.atomic.evaluate",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)

    calls: list[_Request] = []

    async def operation(request: _Request) -> _Result:
        calls.append(request)
        return _Result(status)

    contracts = ModuleType("ale_run.atomic.contracts")
    for name in (
        "ArtifactEntry",
        "AtomicInfrastructureError",
        "EvaluateRequest",
        "EvaluationResult",
        "SolveRequest",
        "SolveResult",
        "SubmissionManifest",
    ):
        setattr(contracts, name, _Request)
    implementation = ModuleType(f"ale_run.atomic.{command}")
    setattr(implementation, command, operation)

    other = "evaluate" if command == "solve" else "solve"
    monkeypatch.setitem(sys.modules, "ale_run.atomic.contracts", contracts)
    monkeypatch.setitem(sys.modules, f"ale_run.atomic.{command}", implementation)
    monkeypatch.setitem(
        sys.modules, f"ale_run.atomic.{other}", _BlockedModule(f"ale_run.atomic.{other}")
    )
    return calls


@pytest.mark.parametrize(
    ("command", "status", "expected_exit"),
    [
        ("solve", "submitted", 0),
        ("solve", "failed", 1),
        ("evaluate", "scored", 0),
        ("evaluate", "infra_failed", 1),
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


@pytest.mark.parametrize("command", ["solve", "evaluate"])
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
