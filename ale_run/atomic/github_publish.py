"""GitHub publication and required-CI auto-merge gate."""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import tempfile
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from .contracts import AtomicInfrastructureError

_MAX_COMMAND_OUTPUT_BYTES = 1024 * 1024
_COMMAND_TIMEOUT_SECONDS = 120
_PROCESS_KILL_WAIT_SECONDS = 1
_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
_RUN_LINK = re.compile(r"/actions/runs/([1-9][0-9]*)(?:/|$)")


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: bytes
    stderr: bytes


@dataclass(frozen=True)
class PullRequest:
    url: str
    state: Literal["OPEN", "CLOSED", "MERGED"]
    merge_commit_sha: str | None = None


@dataclass(frozen=True)
class CheckRun:
    name: str
    bucket: str
    state: str
    workflow: str
    link: str


@dataclass(frozen=True)
class GitHubPublication:
    pull_request_url: str
    ci_run_id: str
    evaluator_version: str


class GitHubClient(Protocol):
    async def push_branch(
        self,
        repository_url: str,
        workspace,
        changed_paths: tuple[str, ...],
        commit_message: str,
    ) -> None: ...

    async def find_pull_requests(
        self,
        repository_url: str,
        branch_name: str,
    ) -> tuple[PullRequest, ...]: ...

    async def create_pull_request(
        self,
        repository_url: str,
        branch_name: str,
        title: str,
        body: str,
    ) -> PullRequest: ...

    async def enable_auto_merge(self, pull_request_url: str) -> None: ...

    async def required_checks(self, pull_request_url: str) -> tuple[CheckRun, ...]: ...

    async def view_pull_request(self, pull_request_url: str) -> PullRequest: ...


class GitHubPublisher:
    """Publish one policy-validated branch and wait for trusted auto-merge."""

    def __init__(
        self,
        client: GitHubClient,
        *,
        initial_backoff_seconds: float = 2,
        max_backoff_seconds: float = 30,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if initial_backoff_seconds < 0 or max_backoff_seconds < 0:
            raise AtomicInfrastructureError(
                "github_publish",
                "CI polling backoff must be non-negative",
            )
        self.client = client
        self.initial_backoff_seconds = initial_backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds
        self.monotonic = monotonic
        self.sleep = sleep

    async def publish(
        self,
        *,
        repository_url: str,
        task_path: str,
        evaluator_id: str,
        workspace,
        timeout_seconds: float,
        allowed_task_dependencies: Sequence[str] = (),
    ) -> GitHubPublication:
        if timeout_seconds <= 0:
            raise AtomicInfrastructureError(
                "github_publish",
                "GitHub publication timeout must be positive",
            )
        changed_paths = workspace.validate_diff(allowed_task_dependencies=allowed_task_dependencies)
        if not changed_paths:
            raise AtomicInfrastructureError(
                "github_publish",
                "authoring produced no task-local changes",
            )
        await self.client.push_branch(
            repository_url,
            workspace,
            changed_paths,
            f"feat(evaluator): author {task_path} {evaluator_id}",
        )
        pull_requests = await self.client.find_pull_requests(
            repository_url,
            workspace.branch_name,
        )
        if len(pull_requests) > 1:
            raise AtomicInfrastructureError(
                "github_publish",
                "multiple pull requests exist for author branch",
            )
        if pull_requests:
            pull_request = pull_requests[0]
            if pull_request.state == "CLOSED":
                raise AtomicInfrastructureError(
                    "github_publish",
                    "author pull request is closed without merge",
                )
        else:
            pull_request = await self.client.create_pull_request(
                repository_url,
                workspace.branch_name,
                f"Author evaluator for {task_path}",
                f"Automated task-local evaluator authoring for `{evaluator_id}`.",
            )
        if pull_request.state == "OPEN":
            await self.client.enable_auto_merge(pull_request.url)

        deadline = self.monotonic() + timeout_seconds
        backoff = self.initial_backoff_seconds
        ci_run_id: str | None = None
        while True:
            if self.monotonic() >= deadline:
                raise AtomicInfrastructureError(
                    "github_publish",
                    "required CI and auto-merge timed out",
                )
            if ci_run_id is None:
                checks = await self.client.required_checks(pull_request.url)
                ci_run_id = _classify_required_checks(checks)
                if ci_run_id is None:
                    await self.sleep(backoff)
                    backoff = min(
                        max(backoff * 2, self.initial_backoff_seconds),
                        self.max_backoff_seconds,
                    )
                    continue
            if pull_request.state == "MERGED":
                return _publication(pull_request, ci_run_id)
            pull_request = await self.client.view_pull_request(pull_request.url)
            if pull_request.state == "CLOSED":
                raise AtomicInfrastructureError(
                    "github_publish",
                    "author pull request closed without merge",
                )
            if pull_request.state == "MERGED":
                return _publication(pull_request, ci_run_id)
            await self.sleep(backoff)
            backoff = min(
                max(backoff * 2, self.initial_backoff_seconds),
                self.max_backoff_seconds,
            )


class GhGitHubClient:
    """Concrete GitHub adapter backed by argv-based ``git`` and ``gh`` calls."""

    def __init__(
        self,
        *,
        command_runner: Callable[..., Awaitable[CommandResult]] | None = None,
    ) -> None:
        self.command_runner = command_runner or _run_command

    async def push_branch(
        self,
        repository_url: str,
        workspace,
        changed_paths: tuple[str, ...],
        commit_message: str,
    ) -> None:
        git_environment = dict(
            getattr(workspace, "git_environment", getattr(workspace, "_environment", {}))
        )
        await self._run_checked(
            ("git", "add", "--all", "--", *changed_paths),
            cwd=workspace.root,
            environment=git_environment,
        )
        diff = await self.command_runner(
            ("git", "diff", "--cached", "--quiet"),
            cwd=workspace.root,
            environment=git_environment,
            timeout_seconds=_COMMAND_TIMEOUT_SECONDS,
            max_output_bytes=_MAX_COMMAND_OUTPUT_BYTES,
        )
        if diff.returncode == 1:
            await self._run_checked(
                (
                    "git",
                    "-c",
                    "user.name=ALE Evaluator Author",
                    "-c",
                    "user.email=ale-evaluator@users.noreply.github.com",
                    "commit",
                    "-m",
                    commit_message,
                ),
                cwd=workspace.root,
                environment=git_environment,
            )
        elif diff.returncode != 0:
            raise AtomicInfrastructureError(
                "github_publish",
                "cannot inspect staged author branch",
            )

        token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
        if token is None:
            token_result = await self._run_checked(
                ("gh", "auth", "token"),
                cwd=workspace.root,
                environment=_gh_environment(),
                max_output_bytes=4096,
            )
            try:
                token = token_result.stdout.decode("utf-8", errors="strict").strip()
            except UnicodeDecodeError as exc:
                raise AtomicInfrastructureError(
                    "github_publish",
                    "gh returned an invalid authentication token",
                ) from exc
        if not token or "\n" in token or "\r" in token:
            raise AtomicInfrastructureError(
                "github_publish",
                "GitHub authentication token is unavailable",
            )
        with tempfile.TemporaryDirectory(prefix="ale-gh-askpass-") as temporary:
            askpass = Path(temporary) / "askpass"
            askpass.write_text(
                "#!/bin/sh\n"
                'case "$1" in\n'
                '  *Username*) printf "%s\\n" "x-access-token" ;;\n'
                '  *) printf "%s\\n" "$ALE_GITHUB_PUSH_TOKEN" ;;\n'
                "esac\n",
                encoding="utf-8",
            )
            askpass.chmod(0o700)
            push_environment = git_environment | {
                "GIT_ASKPASS": str(askpass),
                "GIT_TERMINAL_PROMPT": "0",
                "ALE_GITHUB_PUSH_TOKEN": token,
            }
            await self._run_checked(
                (
                    "git",
                    "push",
                    "--set-upstream",
                    repository_url,
                    f"HEAD:refs/heads/{workspace.branch_name}",
                ),
                cwd=workspace.root,
                environment=push_environment,
            )

    async def find_pull_requests(
        self,
        repository_url: str,
        branch_name: str,
    ) -> tuple[PullRequest, ...]:
        result = await self._run_checked(
            (
                "gh",
                "pr",
                "list",
                "--repo",
                _repository_slug(repository_url),
                "--head",
                branch_name,
                "--state",
                "all",
                "--limit",
                "2",
                "--json",
                "url,state,mergeCommit",
            ),
            environment=_gh_environment(),
        )
        payload = _load_json(result.stdout, "pull request list")
        if not isinstance(payload, list):
            raise AtomicInfrastructureError("github_publish", "invalid pull request list")
        return tuple(_parse_pull_request(item) for item in payload)

    async def create_pull_request(
        self,
        repository_url: str,
        branch_name: str,
        title: str,
        body: str,
    ) -> PullRequest:
        result = await self._run_checked(
            (
                "gh",
                "pr",
                "create",
                "--repo",
                _repository_slug(repository_url),
                "--head",
                branch_name,
                "--base",
                "main",
                "--title",
                title,
                "--body",
                body,
            ),
            environment=_gh_environment(),
        )
        try:
            url = result.stdout.decode("utf-8", errors="strict").strip()
        except UnicodeDecodeError as exc:
            raise AtomicInfrastructureError("github_publish", "invalid gh PR URL") from exc
        if not re.fullmatch(r"https://github\.com/[^/]+/[^/]+/pull/[1-9][0-9]*", url):
            raise AtomicInfrastructureError("github_publish", "invalid gh PR URL")
        return PullRequest(url=url, state="OPEN")

    async def enable_auto_merge(self, pull_request_url: str) -> None:
        await self._run_checked(
            ("gh", "pr", "merge", pull_request_url, "--auto", "--merge"),
            environment=_gh_environment(),
        )

    async def required_checks(self, pull_request_url: str) -> tuple[CheckRun, ...]:
        result = await self.command_runner(
            (
                "gh",
                "pr",
                "checks",
                pull_request_url,
                "--required",
                "--json",
                "name,bucket,state,workflow,link",
            ),
            cwd=None,
            environment=_gh_environment(),
            timeout_seconds=_COMMAND_TIMEOUT_SECONDS,
            max_output_bytes=_MAX_COMMAND_OUTPUT_BYTES,
        )
        if result.returncode not in {0, 1, 8}:
            raise AtomicInfrastructureError(
                "github_publish",
                "gh required-check query failed",
            )
        payload = _load_json(result.stdout, "required checks")
        if not isinstance(payload, list):
            raise AtomicInfrastructureError("github_publish", "invalid required checks")
        checks: list[CheckRun] = []
        for item in payload:
            if not isinstance(item, dict) or any(
                not isinstance(item.get(field), str)
                for field in ("name", "bucket", "state", "workflow", "link")
            ):
                raise AtomicInfrastructureError("github_publish", "invalid required check")
            checks.append(
                CheckRun(
                    name=item["name"],
                    bucket=item["bucket"],
                    state=item["state"],
                    workflow=item["workflow"],
                    link=item["link"],
                )
            )
        return tuple(checks)

    async def view_pull_request(self, pull_request_url: str) -> PullRequest:
        result = await self._run_checked(
            (
                "gh",
                "pr",
                "view",
                pull_request_url,
                "--json",
                "url,state,mergeCommit",
            ),
            environment=_gh_environment(),
        )
        return _parse_pull_request(_load_json(result.stdout, "pull request"))

    async def _run_checked(
        self,
        arguments: Sequence[str],
        *,
        cwd: Path | None = None,
        environment: Mapping[str, str],
        max_output_bytes: int = _MAX_COMMAND_OUTPUT_BYTES,
    ) -> CommandResult:
        result = await self.command_runner(
            tuple(arguments),
            cwd=cwd,
            environment=dict(environment),
            timeout_seconds=_COMMAND_TIMEOUT_SECONDS,
            max_output_bytes=max_output_bytes,
        )
        if result.returncode != 0:
            raise AtomicInfrastructureError(
                "github_publish",
                f"{arguments[0]} {arguments[1]} failed",
            )
        return result


def _classify_required_checks(checks: tuple[CheckRun, ...]) -> str | None:
    if not checks:
        raise AtomicInfrastructureError(
            "github_publish",
            "no required CI checks were reported",
        )
    buckets = {check.bucket for check in checks}
    if buckets - {"pass", "pending", "fail", "cancel", "skipping"}:
        raise AtomicInfrastructureError(
            "github_publish",
            "required CI check returned an ambiguous state",
        )
    if buckets & {"fail", "cancel", "skipping"}:
        raise AtomicInfrastructureError(
            "github_publish",
            "required CI check did not pass",
        )
    if "pending" in buckets:
        return None
    run_ids = {
        match.group(1) for check in checks if (match := _RUN_LINK.search(check.link)) is not None
    }
    if len(run_ids) != 1:
        raise AtomicInfrastructureError(
            "github_publish",
            "required CI run identity is missing or ambiguous",
        )
    return next(iter(run_ids))


def _publication(pull_request: PullRequest, ci_run_id: str) -> GitHubPublication:
    if pull_request.merge_commit_sha is None or not _FULL_SHA.fullmatch(
        pull_request.merge_commit_sha
    ):
        raise AtomicInfrastructureError(
            "github_publish",
            "merged pull request has no full merge commit SHA",
        )
    return GitHubPublication(
        pull_request_url=pull_request.url,
        ci_run_id=ci_run_id,
        evaluator_version=pull_request.merge_commit_sha,
    )


def _repository_slug(repository_url: str) -> str:
    match = re.fullmatch(
        r"https://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+?)(?:\.git)?",
        repository_url,
    )
    if match is None:
        raise AtomicInfrastructureError("github_publish", "invalid GitHub repository URL")
    return f"{match.group(1)}/{match.group(2)}"


def _parse_pull_request(payload: object) -> PullRequest:
    if not isinstance(payload, dict):
        raise AtomicInfrastructureError("github_publish", "invalid pull request response")
    url = payload.get("url")
    state = payload.get("state")
    merge = payload.get("mergeCommit")
    if (
        not isinstance(url, str)
        or not re.fullmatch(r"https://github\.com/[^/]+/[^/]+/pull/[1-9][0-9]*", url)
        or state not in {"OPEN", "CLOSED", "MERGED"}
        or (merge is not None and not isinstance(merge, dict))
    ):
        raise AtomicInfrastructureError("github_publish", "invalid pull request response")
    merge_commit = merge.get("oid") if isinstance(merge, dict) else None
    if merge_commit is not None and (
        not isinstance(merge_commit, str) or not _FULL_SHA.fullmatch(merge_commit)
    ):
        raise AtomicInfrastructureError("github_publish", "invalid pull request merge SHA")
    return PullRequest(url=url, state=state, merge_commit_sha=merge_commit)


def _load_json(payload: bytes, label: str) -> object:
    try:
        return json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AtomicInfrastructureError(
            "github_publish",
            f"invalid gh {label} response",
        ) from exc


def _gh_environment() -> dict[str, str]:
    return {
        key: os.environ[key]
        for key in (
            "PATH",
            "HOME",
            "GH_CONFIG_DIR",
            "GH_HOST",
            "GH_TOKEN",
            "GITHUB_TOKEN",
        )
        if key in os.environ
    }


async def _run_command(
    arguments: Sequence[str],
    *,
    cwd: Path | None,
    environment: Mapping[str, str],
    timeout_seconds: float,
    max_output_bytes: int,
) -> CommandResult:
    try:
        process = await asyncio.create_subprocess_exec(
            *arguments,
            cwd=cwd,
            env=dict(environment),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise AtomicInfrastructureError(
            "github_publish",
            f"cannot start {arguments[0]}",
        ) from exc
    assert process.stdout is not None
    assert process.stderr is not None
    readers = (
        asyncio.create_task(_read_bounded(process.stdout, max_output_bytes)),
        asyncio.create_task(_read_bounded(process.stderr, max_output_bytes)),
    )
    try:
        async with asyncio.timeout(timeout_seconds):
            stdout, stderr, returncode = await asyncio.gather(
                *readers,
                process.wait(),
            )
        return CommandResult(returncode=returncode, stdout=stdout, stderr=stderr)
    except TimeoutError as exc:
        raise AtomicInfrastructureError(
            "github_publish",
            f"{arguments[0]} command timed out",
        ) from exc
    finally:
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for reader in readers:
            if not reader.done():
                reader.cancel()
        await asyncio.gather(*readers, return_exceptions=True)
        if process.returncode is None:
            try:
                await asyncio.wait_for(process.wait(), timeout=_PROCESS_KILL_WAIT_SECONDS)
            except (TimeoutError, ProcessLookupError):
                pass


async def _read_bounded(reader: asyncio.StreamReader, limit: int) -> bytes:
    output = bytearray()
    while True:
        chunk = await reader.read(min(64 * 1024, limit - len(output) + 1))
        if not chunk:
            return bytes(output)
        output.extend(chunk)
        if len(output) > limit:
            raise AtomicInfrastructureError(
                "github_publish",
                "command output exceeds its limit",
            )
