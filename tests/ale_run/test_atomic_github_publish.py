from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest

from ale_run.atomic.contracts import AtomicInfrastructureError
from ale_run.atomic.github_publish import (
    CheckRun,
    CommandResult,
    GhGitHubClient,
    GitHubPublisher,
    PullRequest,
)


class FakeWorkspace:
    root = Path("/tmp/author-workspace")
    branch_name = "ale/author-demo-example-deadbeef"

    def validate_diff(self, *, allowed_task_dependencies: Sequence[str] = ()):
        assert tuple(allowed_task_dependencies) == ()
        return ("tasks/demo/example/evaluator/checks.py",)


class FakeGitHubClient:
    def __init__(
        self,
        *,
        pull_requests: tuple[PullRequest, ...] = (),
        check_sequences: tuple[tuple[CheckRun, ...], ...] = (),
        view_sequences: tuple[PullRequest, ...] = (),
    ) -> None:
        self.pull_requests = pull_requests
        self.check_sequences = list(check_sequences)
        self.view_sequences = list(view_sequences)
        self.calls: list[str] = []

    async def push_branch(self, repository_url, workspace, changed_paths, commit_message):
        assert repository_url == "https://github.com/openai/ale-tasks"
        assert workspace.branch_name == FakeWorkspace.branch_name
        assert changed_paths == ("tasks/demo/example/evaluator/checks.py",)
        assert "demo/example" in commit_message
        self.calls.append("push")

    async def find_pull_requests(self, repository_url, branch_name):
        self.calls.append("find")
        return self.pull_requests

    async def create_pull_request(self, repository_url, branch_name, title, body):
        assert title
        assert body
        self.calls.append("create")
        return PullRequest(
            url="https://github.com/openai/ale-tasks/pull/123",
            state="OPEN",
        )

    async def enable_auto_merge(self, pull_request_url):
        self.calls.append("auto_merge")

    async def required_checks(self, pull_request_url):
        self.calls.append("checks")
        if len(self.check_sequences) > 1:
            return self.check_sequences.pop(0)
        return self.check_sequences[0]

    async def view_pull_request(self, pull_request_url):
        self.calls.append("view")
        if len(self.view_sequences) > 1:
            return self.view_sequences.pop(0)
        return self.view_sequences[0]


def _check(bucket: str, *, run_id: str = "987654") -> CheckRun:
    return CheckRun(
        name="evaluator-ci",
        bucket=bucket,
        state=bucket.upper(),
        workflow="evaluator",
        link=f"https://github.com/openai/ale-tasks/actions/runs/{run_id}/job/1",
    )


@pytest.mark.asyncio
async def test_publisher_pushes_creates_pr_and_waits_for_passed_merge() -> None:
    client = FakeGitHubClient(
        check_sequences=((_check("pending"),), (_check("pass"),)),
        view_sequences=(
            PullRequest(
                url="https://github.com/openai/ale-tasks/pull/123",
                state="OPEN",
            ),
            PullRequest(
                url="https://github.com/openai/ale-tasks/pull/123",
                state="MERGED",
                merge_commit_sha="d" * 40,
            ),
        ),
    )
    publisher = GitHubPublisher(client, initial_backoff_seconds=0, max_backoff_seconds=0)

    publication = await publisher.publish(
        repository_url="https://github.com/openai/ale-tasks",
        task_path="demo/example",
        evaluator_id="rubric",
        workspace=FakeWorkspace(),
        timeout_seconds=5,
    )

    assert publication.evaluator_version == "d" * 40
    assert publication.ci_run_id == "987654"
    assert publication.pull_request_url.endswith("/pull/123")
    assert client.calls == [
        "push",
        "find",
        "create",
        "auto_merge",
        "checks",
        "checks",
        "view",
        "view",
    ]


@pytest.mark.asyncio
async def test_publisher_reuses_one_existing_pr() -> None:
    existing = PullRequest(
        url="https://github.com/openai/ale-tasks/pull/123",
        state="OPEN",
    )
    client = FakeGitHubClient(
        pull_requests=(existing,),
        check_sequences=((_check("pass"),),),
        view_sequences=(
            PullRequest(
                url=existing.url,
                state="MERGED",
                merge_commit_sha="d" * 40,
            ),
        ),
    )

    await GitHubPublisher(client, initial_backoff_seconds=0).publish(
        repository_url="https://github.com/openai/ale-tasks",
        task_path="demo/example",
        evaluator_id="rubric",
        workspace=FakeWorkspace(),
        timeout_seconds=5,
    )

    assert "create" not in client.calls
    assert "auto_merge" in client.calls


@pytest.mark.parametrize("bucket", ["fail", "cancel", "skipping", "unknown"])
@pytest.mark.asyncio
async def test_publisher_fails_closed_for_nonpassing_required_check(bucket: str) -> None:
    client = FakeGitHubClient(
        check_sequences=((_check(bucket),),),
        view_sequences=(),
    )

    with pytest.raises(AtomicInfrastructureError) as caught:
        await GitHubPublisher(client, initial_backoff_seconds=0).publish(
            repository_url="https://github.com/openai/ale-tasks",
            task_path="demo/example",
            evaluator_id="rubric",
            workspace=FakeWorkspace(),
            timeout_seconds=5,
        )

    assert caught.value.category == "github_publish"
    assert "view" not in client.calls


@pytest.mark.asyncio
async def test_publisher_rejects_missing_or_ambiguous_required_checks() -> None:
    for checks in ((), (_check("pass", run_id="1"), _check("pass", run_id="2"))):
        client = FakeGitHubClient(
            check_sequences=(checks,),
            view_sequences=(
                PullRequest(
                    url="https://github.com/openai/ale-tasks/pull/123",
                    state="MERGED",
                    merge_commit_sha="d" * 40,
                ),
            ),
        )
        with pytest.raises(AtomicInfrastructureError):
            await GitHubPublisher(client, initial_backoff_seconds=0).publish(
                repository_url="https://github.com/openai/ale-tasks",
                task_path="demo/example",
                evaluator_id="rubric",
                workspace=FakeWorkspace(),
                timeout_seconds=5,
            )


@pytest.mark.asyncio
async def test_publisher_rejects_multiple_or_closed_unmerged_prs() -> None:
    pr = PullRequest(url="https://github.com/openai/ale-tasks/pull/123", state="OPEN")
    for pull_requests in (
        (pr, pr),
        (
            PullRequest(
                url="https://github.com/openai/ale-tasks/pull/123",
                state="CLOSED",
            ),
        ),
    ):
        client = FakeGitHubClient(pull_requests=pull_requests)
        with pytest.raises(AtomicInfrastructureError):
            await GitHubPublisher(client).publish(
                repository_url="https://github.com/openai/ale-tasks",
                task_path="demo/example",
                evaluator_id="rubric",
                workspace=FakeWorkspace(),
                timeout_seconds=5,
            )


@pytest.mark.asyncio
async def test_publisher_times_out_without_merging_pending_checks() -> None:
    client = FakeGitHubClient(check_sequences=((_check("pending"),),))
    ticks = iter((0.0, 0.0, 2.0))
    publisher = GitHubPublisher(
        client,
        initial_backoff_seconds=0,
        monotonic=lambda: next(ticks),
    )

    with pytest.raises(AtomicInfrastructureError, match="timed out"):
        await publisher.publish(
            repository_url="https://github.com/openai/ale-tasks",
            task_path="demo/example",
            evaluator_id="rubric",
            workspace=FakeWorkspace(),
            timeout_seconds=1,
        )

    assert client.calls.count("auto_merge") == 1
    assert "view" not in client.calls


@pytest.mark.asyncio
async def test_gh_client_uses_argv_commands_and_never_puts_token_in_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    calls: list[tuple[tuple[str, ...], dict[str, str]]] = []

    async def runner(arguments, *, cwd, environment, **_limits):
        calls.append((tuple(arguments), dict(environment)))
        if tuple(arguments[:3]) == ("gh", "auth", "token"):
            return CommandResult(0, b"top-secret-token\n", b"")
        if tuple(arguments[:3]) == ("git", "diff", "--cached"):
            return CommandResult(1, b"", b"")
        return CommandResult(0, b"", b"")

    client = GhGitHubClient(command_runner=runner)
    workspace = SimpleNamespace(
        root=tmp_path,
        branch_name="ale/author-demo-example-deadbeef",
        git_environment={"PATH": "/usr/bin"},
    )

    await client.push_branch(
        "https://github.com/openai/ale-tasks",
        workspace,
        ("tasks/demo/example/evaluator/checks.py",),
        "author evaluator",
    )

    arguments = [argument for command, _env in calls for argument in command]
    assert "top-secret-token" not in arguments
    assert any(command[:2] == ("git", "push") for command, _env in calls)
    push_environment = next(
        environment for command, environment in calls if command[:2] == ("git", "push")
    )
    assert push_environment["ALE_GITHUB_PUSH_TOKEN"] == "top-secret-token"


@pytest.mark.asyncio
async def test_gh_client_parses_prs_checks_and_merged_sha() -> None:
    responses = iter(
        (
            b'[{"url":"https://github.com/openai/ale-tasks/pull/123","state":"OPEN","mergeCommit":null}]',
            b"https://github.com/openai/ale-tasks/pull/124\n",
            b'[{"name":"ci","bucket":"pass","state":"SUCCESS","workflow":"eval","link":"https://github.com/openai/ale-tasks/actions/runs/9/job/1"}]',
            b'{"url":"https://github.com/openai/ale-tasks/pull/124","state":"MERGED","mergeCommit":{"oid":"d'
            + b"d" * 39
            + b'"}}',
        )
    )

    async def runner(arguments, **_kwargs):
        if arguments[:3] == ("gh", "pr", "merge"):
            return CommandResult(0, b"", b"")
        return CommandResult(0, next(responses), b"")

    client = GhGitHubClient(command_runner=runner)
    prs = await client.find_pull_requests(
        "https://github.com/openai/ale-tasks",
        "ale/author-demo-example-deadbeef",
    )
    created = await client.create_pull_request(
        "https://github.com/openai/ale-tasks",
        "ale/author-demo-example-deadbeef",
        "title",
        "body",
    )
    checks = await client.required_checks(created.url)
    viewed = await client.view_pull_request(created.url)

    assert prs[0].state == "OPEN"
    assert created.url.endswith("/pull/124")
    assert checks[0].bucket == "pass"
    assert viewed.merge_commit_sha == "d" * 40
