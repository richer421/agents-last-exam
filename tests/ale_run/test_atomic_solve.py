from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from ale_run.atomic.contracts import SolveRequest
from ale_run.atomic.runtime import AtomicRuntime


class _FakeProvider:
    def __init__(self, *, metadata: dict[str, str]) -> None:
        self.sandbox = SimpleNamespace(id="sandbox-1", metadata=metadata, os="linux")
        self.acquire_calls: list[object] = []
        self.release_calls: list[str] = []

    async def acquire(self, spec: object) -> SimpleNamespace:
        self.acquire_calls.append(spec)
        return self.sandbox

    def open_session(self, _sandbox: object) -> SimpleNamespace:
        return SimpleNamespace(computer=SimpleNamespace(interface=SimpleNamespace()))

    async def release(self, _sandbox: object, *, mode: str) -> None:
        self.release_calls.append(mode)


@pytest.fixture
def solve_request(tmp_path: Path) -> SolveRequest:
    return _make_solve_request(tmp_path)


def _make_solve_request(tmp_path: Path) -> SolveRequest:
    repo = tmp_path / "task-repo"
    task_dir = repo / "tasks" / "toy"
    task_dir.mkdir(parents=True)
    (task_dir / "main.py").write_text(
        "class Config:\n"
        "    task_description = 'Solve the toy task.'\n"
        "    metadata = {}\n"
        "config = Config()\n",
        encoding="utf-8",
    )
    (task_dir / "task_card.json").write_text('{"vm":{"snapshot":"test-image"}}\n', encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "tasks/toy"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "task",
        ],
        check=True,
    )
    task_commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    return SolveRequest(
        submission_id=uuid4(),
        runtime_spec_path=tmp_path / "runtime.yaml",
        task_repo=repo,
        task_path="toy",
        variant_index=0,
        agent_id="agent-under-test",
        task_commit=task_commit,
        image_id="m-expected",
        submission_root="oss://submissions",
    )


@pytest.mark.asyncio
async def test_open_acquires_one_vm_and_releases_it_when_the_body_raises(
    solve_request: SolveRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_spec = SimpleNamespace(environment=SimpleNamespace(), artifacts=None)
    monkeypatch.setattr("ale_run.atomic.runtime.load_experiment", lambda _path: runtime_spec)
    provider = _FakeProvider(metadata={"image_id": "m-expected"})

    with pytest.raises(RuntimeError, match="body failed"):
        async with AtomicRuntime.open(request=solve_request, provider=provider) as runtime:
            assert runtime.provider is provider
            assert runtime.task_dir == solve_request.task_repo / "tasks" / "toy"
            assert runtime.runtime_spec is runtime_spec
            raise RuntimeError("body failed")

    assert len(provider.acquire_calls) == 1
    assert provider.release_calls == ["delete"]
