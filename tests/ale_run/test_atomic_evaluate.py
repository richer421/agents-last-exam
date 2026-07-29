from __future__ import annotations

from types import SimpleNamespace

import pytest

from ale_run.atomic.contracts import AtomicInfrastructureError, EvaluateRequest
from ale_run.atomic.runtime import AtomicRuntime
from tests.ale_run.test_atomic_solve import _FakeProvider, _make_solve_request


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata", [{}, {"image_id": "m-other"}])
async def test_open_fails_closed_for_missing_or_mismatched_image_identity(
    tmp_path,
    metadata: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solve_request = _make_solve_request(tmp_path)
    request = EvaluateRequest(
        submission_id=solve_request.submission_id,
        runtime_spec_path=solve_request.runtime_spec_path,
        task_repo=solve_request.task_repo,
        task_path=solve_request.task_path,
        variant_index=solve_request.variant_index,
        task_commit=solve_request.task_commit,
        image_id=solve_request.image_id,
        submission_root=solve_request.submission_root,
        evaluator_id="strict-evaluator",
        evaluator_version=solve_request.task_commit,
    )
    monkeypatch.setattr(
        "ale_run.atomic.runtime.load_experiment",
        lambda _path: SimpleNamespace(environment=SimpleNamespace(), artifacts=None),
    )
    provider = _FakeProvider(metadata=metadata)

    with pytest.raises(AtomicInfrastructureError, match="image_id mismatch") as caught:
        async with AtomicRuntime.open(
            request=request,
            run_setup=False,
            provider=provider,
        ):
            pass

    assert caught.value.category == "image_identity"
    assert provider.release_calls == ["delete"]


@pytest.mark.asyncio
async def test_evaluate_open_skips_setup_and_never_resolves_an_agent(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    solve_request = _make_solve_request(tmp_path)
    task_main = solve_request.task_repo / "tasks" / "toy" / "main.py"
    task_main.write_text(
        task_main.read_text(encoding="utf-8")
        + "\ndef setup(_config, _session):\n"
        + "    (__import__('pathlib').Path(__file__).with_name('setup-ran')).write_text('yes')\n",
        encoding="utf-8",
    )
    request = EvaluateRequest(
        submission_id=solve_request.submission_id,
        runtime_spec_path=solve_request.runtime_spec_path,
        task_repo=solve_request.task_repo,
        task_path=solve_request.task_path,
        variant_index=solve_request.variant_index,
        task_commit=solve_request.task_commit,
        image_id=solve_request.image_id,
        submission_root=solve_request.submission_root,
        evaluator_id="strict-evaluator",
        evaluator_version=solve_request.task_commit,
    )
    monkeypatch.setattr(
        "ale_run.atomic.runtime.load_experiment",
        lambda _path: SimpleNamespace(environment=SimpleNamespace(), artifacts=None),
    )
    monkeypatch.setattr(
        "ale_run.orchestration.factory.resolve_agent",
        lambda _spec: (_ for _ in ()).throw(AssertionError("agent resolved")),
    )
    provider = _FakeProvider(metadata={"image_id": "m-expected"})

    async with AtomicRuntime.open(request=request, run_setup=False, provider=provider) as runtime:
        assert runtime.task_driver is None

    assert not (solve_request.task_repo / "tasks" / "toy" / "setup-ran").exists()
    assert provider.release_calls == ["delete"]
