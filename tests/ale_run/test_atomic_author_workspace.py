import os
import subprocess
from pathlib import Path
from uuid import UUID

import pytest

from ale_run.atomic.author_workspace import isolated_author_workspace
from ale_run.atomic.contracts import AtomicInfrastructureError, AuthorEvaluatorRequest


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *arguments],
        text=True,
    ).strip()


def _source_repository(
    tmp_path: Path,
    *,
    include_legacy_evaluator: bool = False,
) -> tuple[Path, str]:
    repo = tmp_path / "source"
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "ALE Test"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "ale@example.com"],
        check=True,
    )
    task = repo / "tasks" / "demo" / "example"
    task.mkdir(parents=True)
    (task / "task_card.json").write_text("{}\n", encoding="utf-8")
    (task / "rubrics.json").write_text('{"rubrics": []}\n', encoding="utf-8")
    (task / "reference").mkdir()
    (task / "reference" / "manifest.json").write_text("{}\n", encoding="utf-8")
    if include_legacy_evaluator:
        (task / "evaluator").mkdir()
        (task / "evaluator" / "checks.py").write_text("LEGACY = True\n", encoding="utf-8")
        (task / "tests").mkdir()
        (task / "tests" / "test_evaluator_legacy.py").write_text(
            "def test_legacy(): pass\n", encoding="utf-8"
        )
    other = repo / "tasks" / "demo" / "other"
    other.mkdir(parents=True)
    (other / "task_card.json").write_text("{}\n", encoding="utf-8")
    (repo / "ale_run").mkdir()
    (repo / "ale_run" / "runtime.py").write_text("SHARED = True\n", encoding="utf-8")
    (repo / ".github" / "workflows").mkdir(parents=True)
    (repo / ".github" / "workflows" / "ci.yml").write_text("name: ci\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "--all"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "baseline"], check=True)
    return repo, _git(repo, "rev-parse", "HEAD")


def _request(commit: str) -> AuthorEvaluatorRequest:
    return AuthorEvaluatorRequest(
        authoring_id=UUID("00000000-0000-0000-0000-000000000001"),
        task_repository_url="https://github.com/openai/ale-tasks",
        task_path="demo/example",
        variant_index=0,
        task_commit=commit,
        image_id="m-authoring-image",
        rubric_uri="oss://ale-rubrics/example/rubrics.json",
        rubric_hash="b" * 64,
        reference_manifest_uri="oss://ale-reference/example/manifest.json",
        reference_manifest_hash="c" * 64,
        evaluator_id="rubric",
        evaluator_sdk_version="1.2.3",
    )


def test_workspace_starts_at_exact_commit_on_deterministic_isolated_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, commit = _source_repository(tmp_path)
    hostile_home = tmp_path / "hostile-home"
    hostile_home.mkdir()
    (hostile_home / ".gitconfig").write_text(
        "[core]\n\thooksPath = /tmp/host-hooks\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(hostile_home))

    with isolated_author_workspace(source, _request(commit)) as workspace:
        assert _git(workspace.root, "rev-parse", "HEAD") == commit
        assert _git(workspace.root, "branch", "--show-current") == workspace.branch_name
        assert workspace.branch_name.startswith("ale/author-demo-example-")
        assert not (workspace.root / ".git" / "hooks").exists()
        hooks_path = subprocess.run(
            [
                "git",
                "-C",
                str(workspace.root),
                "config",
                "--local",
                "--get",
                "core.hooksPath",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert hooks_path.returncode == 1
        assert hooks_path.stdout == ""
        assert (workspace.task_directory / "task_card.json").read_text(encoding="utf-8") == "{}\n"


def test_workspace_scrubs_legacy_evaluator_before_fresh_author_can_read_it(
    tmp_path: Path,
) -> None:
    source, commit = _source_repository(tmp_path, include_legacy_evaluator=True)

    with isolated_author_workspace(source, _request(commit)) as workspace:
        assert not (workspace.task_directory / "evaluator").exists()
        assert not (workspace.task_directory / "tests" / "test_evaluator_legacy.py").exists()
        assert (workspace.task_directory / "rubrics.json").is_file()
        assert (workspace.task_directory / "reference" / "manifest.json").is_file()
        assert workspace.validate_diff() == (
            "tasks/demo/example/evaluator/checks.py",
            "tasks/demo/example/tests/test_evaluator_legacy.py",
        )


def test_workspace_allows_only_selected_task_evaluator_changes(tmp_path: Path) -> None:
    source, commit = _source_repository(tmp_path)

    with isolated_author_workspace(source, _request(commit)) as workspace:
        evaluator = workspace.task_directory / "evaluator"
        evaluator.mkdir()
        (evaluator / "checks.py").write_text("CHECKS = []\n", encoding="utf-8")
        (evaluator / "rubric-plan.json").write_text("{}\n", encoding="utf-8")

        assert workspace.validate_diff() == (
            "tasks/demo/example/evaluator/checks.py",
            "tasks/demo/example/evaluator/rubric-plan.json",
        )


@pytest.mark.parametrize(
    "relative_path",
    [
        "ale_run/runtime.py",
        ".github/workflows/ci.yml",
        "tasks/demo/other/task_card.json",
        "tasks/demo/example/rubrics.json",
        "tasks/demo/example/reference/manifest.json",
        ".gitmodules",
    ],
)
def test_workspace_rejects_changes_outside_allowlist(
    tmp_path: Path,
    relative_path: str,
) -> None:
    source, commit = _source_repository(tmp_path)

    with isolated_author_workspace(source, _request(commit)) as workspace:
        target = workspace.root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("mutated\n", encoding="utf-8")

        with pytest.raises(AtomicInfrastructureError, match="not allowed"):
            workspace.validate_diff()


def test_workspace_rejects_new_symlink_and_nested_git_metadata(tmp_path: Path) -> None:
    source, commit = _source_repository(tmp_path)

    with isolated_author_workspace(source, _request(commit)) as workspace:
        evaluator = workspace.task_directory / "evaluator"
        evaluator.mkdir()
        (evaluator / "linked.py").symlink_to(workspace.root / "ale_run" / "runtime.py")

        with pytest.raises(AtomicInfrastructureError, match="symlink"):
            workspace.validate_diff()

        (evaluator / "linked.py").unlink()
        (evaluator / ".git").mkdir()
        with pytest.raises(AtomicInfrastructureError, match="nested Git"):
            workspace.validate_diff()


def test_workspace_allows_only_explicit_task_dependency_declarations(
    tmp_path: Path,
) -> None:
    source, commit = _source_repository(tmp_path)

    with isolated_author_workspace(source, _request(commit)) as workspace:
        dependency = workspace.task_directory / "requirements-evaluator.txt"
        dependency.write_text("harbor-rewardkit==1.0.0\n", encoding="utf-8")

        with pytest.raises(AtomicInfrastructureError, match="not allowed"):
            workspace.validate_diff()
        assert workspace.validate_diff(
            allowed_task_dependencies=("requirements-evaluator.txt",)
        ) == ("tasks/demo/example/requirements-evaluator.txt",)


def test_workspace_rejects_author_mutation_of_git_control_state(tmp_path: Path) -> None:
    source, commit = _source_repository(tmp_path)

    with isolated_author_workspace(source, _request(commit)) as workspace:
        subprocess.run(
            [
                "git",
                "-C",
                str(workspace.root),
                "config",
                "core.hooksPath",
                str(tmp_path),
            ],
            check=True,
        )

        with pytest.raises(AtomicInfrastructureError, match="Git control state"):
            workspace.validate_diff()


def test_workspace_rejects_ignored_evaluator_output_that_would_not_be_committed(
    tmp_path: Path,
) -> None:
    source, commit = _source_repository(tmp_path)
    (source / ".gitignore").write_text(
        "tasks/demo/example/evaluator/*.cache\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(source), "add", ".gitignore"], check=True)
    subprocess.run(
        ["git", "-C", str(source), "commit", "-qm", "ignore evaluator cache"],
        check=True,
    )
    commit = _git(source, "rev-parse", "HEAD")

    with isolated_author_workspace(source, _request(commit)) as workspace:
        evaluator = workspace.task_directory / "evaluator"
        evaluator.mkdir()
        (evaluator / "generated.cache").write_text("must be committed\n", encoding="utf-8")

        with pytest.raises(AtomicInfrastructureError, match="ignored"):
            workspace.validate_diff()


def test_workspace_is_removed_after_context_exit(tmp_path: Path) -> None:
    source, commit = _source_repository(tmp_path)

    with isolated_author_workspace(source, _request(commit)) as workspace:
        root = workspace.root
        assert root.exists()

    assert not root.exists()
    assert os.path.commonpath((root, tmp_path)) != str(tmp_path)
