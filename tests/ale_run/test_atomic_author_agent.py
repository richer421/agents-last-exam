import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from uuid import UUID

import pytest

from ale_run.atomic.author_agent import (
    AuthorAgent,
    AuthorInputBundle,
    _stage_inputs,
    _validate_inputs,
)
from ale_run.atomic.author_workspace import AuthorWorkspace
from ale_run.atomic.contracts import (
    AtomicInfrastructureError,
    AuthorEvaluatorRequest,
    ReferenceFileEntry,
    ReferenceManifest,
)


def _rubric_bytes() -> bytes:
    return json.dumps(
        {
            "rubrics": [
                {
                    "id": "format_valid",
                    "weight": 0.25,
                    "score_min": 0,
                    "score_max": 1,
                    "required": True,
                },
                {
                    "id": "visual_quality",
                    "weight": 0.75,
                    "score_min": 0,
                    "score_max": 1,
                    "required": False,
                },
            ]
        },
        separators=(",", ":"),
    ).encode()


def _valid_plan() -> dict[str, object]:
    return {
        "schema_version": 1,
        "rubric_hash": hashlib.sha256(_rubric_bytes()).hexdigest(),
        "items": [
            {
                "rubric_id": "format_valid",
                "implementation_mode": "programmatic",
                "weight": 0.25,
                "score_min": 0,
                "score_max": 1,
                "required": True,
                "rationale": "The file format is deterministic.",
            },
            {
                "rubric_id": "visual_quality",
                "implementation_mode": "llm_judge",
                "weight": 0.75,
                "score_min": 0,
                "score_max": 1,
                "required": False,
                "rationale": "Visual quality requires semantic judgment.",
            },
        ],
    }


def _reference() -> tuple[bytes, dict[str, bytes]]:
    artifact = b"trusted reference"
    manifest = (
        ReferenceManifest(
            files=(
                ReferenceFileEntry(
                    path="reference.txt",
                    size_bytes=len(artifact),
                    sha256=hashlib.sha256(artifact).hexdigest(),
                ),
            )
        )
        .model_dump_json()
        .encode()
    )
    return manifest, {"reference.txt": artifact}


def _request() -> AuthorEvaluatorRequest:
    reference_manifest, _artifacts = _reference()
    return AuthorEvaluatorRequest(
        authoring_id=UUID("00000000-0000-0000-0000-000000000001"),
        task_repository_url="https://github.com/openai/ale-tasks",
        task_path="demo/example",
        variant_index=0,
        task_commit="a" * 40,
        image_id="m-authoring-image",
        rubric_uri="oss://ale-rubrics/example/rubrics.json",
        rubric_hash=hashlib.sha256(_rubric_bytes()).hexdigest(),
        reference_manifest_uri="oss://ale-reference/example/manifest.json",
        reference_manifest_hash=hashlib.sha256(reference_manifest).hexdigest(),
        evaluator_id="rubric",
        evaluator_sdk_version="1.2.3",
    )


def _bundle() -> AuthorInputBundle:
    reference_manifest, artifacts = _reference()
    return AuthorInputBundle(
        task_contract_files={
            "task_card.json": b"{}\n",
            "formats/output.json": b'{"type":"object"}\n',
        },
        rubric=_rubric_bytes(),
        reference_manifest=reference_manifest,
        reference_artifacts=artifacts,
        evaluator_sdk_contract=b'{"version":"1.2.3"}\n',
        image_capability_statement=b'{"image_id":"m-authoring-image"}\n',
    )


def _workspace(tmp_path: Path) -> AuthorWorkspace:
    root = tmp_path / "repo"
    task = root / "tasks" / "demo" / "example"
    task.mkdir(parents=True)
    return AuthorWorkspace(
        root=root,
        task_directory=task,
        branch_name="ale/author-demo-example-deadbeef",
        task_commit="a" * 40,
        _environment={},
        _control_fingerprint="",
    )


def test_author_agent_stages_large_photoshop_reference(tmp_path: Path) -> None:
    artifact = b"x" * (50 * 1024 * 1024 + 1)
    manifest = (
        ReferenceManifest(
            files=(
                ReferenceFileEntry(
                    path="final_result.psd",
                    size_bytes=len(artifact),
                    sha256=hashlib.sha256(artifact).hexdigest(),
                ),
            )
        )
        .model_dump_json()
        .encode()
    )
    request = _request().model_copy(
        update={"reference_manifest_hash": hashlib.sha256(manifest).hexdigest()}
    )
    bundle = AuthorInputBundle(
        task_contract_files={"task_card.json": b"{}\n"},
        rubric=_rubric_bytes(),
        reference_manifest=manifest,
        reference_artifacts={"final_result.psd": artifact},
        evaluator_sdk_contract=b"{}\n",
        image_capability_statement=b"{}\n",
    )

    _items, reference_manifest = _validate_inputs(request, bundle)
    output = _stage_inputs(tmp_path, bundle, reference_manifest)

    staged = output.parent / "inputs" / "reference" / "artifacts" / "final_result.psd"
    assert staged.stat().st_size == len(artifact)


def _author_script(
    tmp_path: Path,
    *,
    plan: dict[str, object] | None = None,
    phase_one_extra: bool = False,
    phase_two_plan: dict[str, object] | None = None,
    provider_key: str | None = None,
) -> tuple[str, ...]:
    script = tmp_path / "author.py"
    phases = tmp_path / "phases.log"
    selected_plan = plan or _valid_plan()
    script.write_text(
        f"""\
import json
import os
from pathlib import Path

assert "FORBIDDEN_SECRET" not in os.environ
{f'assert os.environ["TRUE_SOTA_API_KEY"] == {provider_key!r}' if provider_key else ''}
output = Path(os.environ["ALE_AUTHOR_OUTPUT_DIR"])
phase = os.environ["ALE_AUTHOR_PHASE"]
with Path({str(phases)!r}).open("a", encoding="utf-8") as stream:
    stream.write(phase + "\\n")
if phase == "plan":
    (output / "rubric-plan.json").write_text(json.dumps({selected_plan!r}), encoding="utf-8")
    {"(output / 'checks.py').write_text('EARLY = True\\n', encoding='utf-8')" if phase_one_extra else "pass"}
else:
    {f"(output / 'rubric-plan.json').write_text(json.dumps({phase_two_plan!r}), encoding='utf-8')" if phase_two_plan else "pass"}
    (output / "checks.py").write_text("CHECKS = []\\n", encoding="utf-8")
    (output / "judge.toml").write_text("backend = 'mock'\\n", encoding="utf-8")
""",
        encoding="utf-8",
    )
    return (sys.executable, str(script))


@pytest.mark.asyncio
async def test_author_agent_stages_only_declared_inputs_and_runs_plan_before_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _author_script(tmp_path)
    monkeypatch.setenv("FORBIDDEN_SECRET", "must-not-leak")
    agent = AuthorAgent(command, timeout_seconds=5)
    workspace = _workspace(tmp_path)

    authored = await agent.author(_request(), workspace, _bundle())

    assert authored.rubric_plan.model_dump(mode="json") == _valid_plan()
    assert authored.files == ("checks.py", "judge.toml", "rubric-plan.json")
    assert (workspace.task_directory / "evaluator" / "checks.py").read_text(
        encoding="utf-8"
    ) == "CHECKS = []\n"
    assert (tmp_path / "phases.log").read_text(encoding="utf-8") == "plan\nimplement\n"


@pytest.mark.asyncio
async def test_author_agent_forwards_only_allowlisted_provider_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRUE_SOTA_API_KEY", "true-sota-key")
    monkeypatch.setenv("FORBIDDEN_SECRET", "must-not-leak")
    agent = AuthorAgent(
        _author_script(tmp_path, provider_key="true-sota-key"),
        timeout_seconds=5,
    )

    await agent.author(_request(), _workspace(tmp_path), _bundle())


@pytest.mark.parametrize(
    "mutation",
    ["omitted", "duplicate", "undeclared", "weight", "range"],
)
@pytest.mark.asyncio
async def test_author_agent_rejects_invalid_rubric_plan(
    tmp_path: Path,
    mutation: str,
) -> None:
    plan = _valid_plan()
    items = plan["items"]
    assert isinstance(items, list)
    if mutation == "omitted":
        items.pop()
    elif mutation == "duplicate":
        items[1] = dict(items[0])
    elif mutation == "undeclared":
        items[1] = dict(items[1]) | {"rubric_id": "invented"}
    elif mutation == "weight":
        items[0] = dict(items[0]) | {"weight": 0.5}
    else:
        items[0] = dict(items[0]) | {"score_max": 10}
    agent = AuthorAgent(_author_script(tmp_path, plan=plan), timeout_seconds=5)

    with pytest.raises(AtomicInfrastructureError) as caught:
        await agent.author(_request(), _workspace(tmp_path), _bundle())

    assert caught.value.category == "author_agent"


@pytest.mark.asyncio
async def test_author_agent_reports_invalid_rubric_plan_fields(tmp_path: Path) -> None:
    plan = _valid_plan()
    plan["rubrics"] = plan.pop("items")
    agent = AuthorAgent(_author_script(tmp_path, plan=plan), timeout_seconds=5)

    with pytest.raises(AtomicInfrastructureError) as caught:
        await agent.author(_request(), _workspace(tmp_path), _bundle())

    assert "items: Field required" in caught.value.message
    assert "rubrics: Extra inputs are not permitted" in caught.value.message


@pytest.mark.asyncio
async def test_author_agent_rejects_code_created_before_valid_plan(tmp_path: Path) -> None:
    agent = AuthorAgent(
        _author_script(tmp_path, phase_one_extra=True),
        timeout_seconds=5,
    )

    with pytest.raises(AtomicInfrastructureError, match="before rubric plan"):
        await agent.author(_request(), _workspace(tmp_path), _bundle())

    assert (tmp_path / "phases.log").read_text(encoding="utf-8") == "plan\n"


@pytest.mark.asyncio
async def test_author_agent_revalidates_unchanged_plan_after_generation(
    tmp_path: Path,
) -> None:
    changed = _valid_plan()
    items = changed["items"]
    assert isinstance(items, list)
    items[0] = dict(items[0]) | {"weight": 0.5}
    agent = AuthorAgent(
        _author_script(tmp_path, phase_two_plan=changed),
        timeout_seconds=5,
    )

    with pytest.raises(AtomicInfrastructureError, match="changed after implementation"):
        await agent.author(_request(), _workspace(tmp_path), _bundle())


@pytest.mark.asyncio
async def test_author_agent_validates_reference_and_rubric_hashes_before_command(
    tmp_path: Path,
) -> None:
    command = _author_script(tmp_path)
    bundle = _bundle()
    corrupt = AuthorInputBundle(
        task_contract_files=bundle.task_contract_files,
        rubric=bundle.rubric,
        reference_manifest=bundle.reference_manifest,
        reference_artifacts={"reference.txt": b"corrupt"},
        evaluator_sdk_contract=bundle.evaluator_sdk_contract,
        image_capability_statement=bundle.image_capability_statement,
    )

    with pytest.raises(AtomicInfrastructureError, match="reference artifact"):
        await AuthorAgent(command, timeout_seconds=5).author(
            _request(), _workspace(tmp_path), corrupt
        )

    assert not (tmp_path / "phases.log").exists()


@pytest.mark.asyncio
async def test_author_agent_bounds_command_output(tmp_path: Path) -> None:
    script = tmp_path / "noisy.py"
    script.write_text("print('x' * 1000)\n", encoding="utf-8")
    agent = AuthorAgent(
        (sys.executable, str(script)),
        timeout_seconds=5,
        max_output_bytes=100,
    )

    with pytest.raises(AtomicInfrastructureError, match="output exceeds"):
        await agent.author(_request(), _workspace(tmp_path), _bundle())


@pytest.mark.asyncio
async def test_author_agent_cancellation_kills_process_group(tmp_path: Path) -> None:
    script = tmp_path / "blocking.py"
    pid_path = tmp_path / "child.pid"
    script.write_text(
        f"""\
import os
from pathlib import Path
import subprocess
import time
child = subprocess.Popen(["sleep", "60"])
Path({str(pid_path)!r}).write_text(str(child.pid), encoding="utf-8")
time.sleep(60)
""",
        encoding="utf-8",
    )
    task = asyncio.create_task(
        AuthorAgent((sys.executable, str(script)), timeout_seconds=60).author(
            _request(), _workspace(tmp_path), _bundle()
        )
    )
    for _attempt in range(100):
        if pid_path.exists():
            break
        await asyncio.sleep(0.01)
    assert pid_path.exists()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    child_pid = int(pid_path.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)
