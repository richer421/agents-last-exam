"""Independent atomic evaluator-authoring operation."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
import tempfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from pydantic import ValidationError

from .author_agent import AuthorAgent, AuthoredEvaluator, AuthorInputBundle
from .author_registry import AuthorRegistry, FilesystemAuthorRegistry
from .author_workspace import (
    AuthorWorkspace,
    _isolated_git_environment,
    isolated_author_workspace,
)
from .contracts import (
    AtomicInfrastructureError,
    AuthorEvaluatorRegistryIdentity,
    AuthorEvaluatorRequest,
    AuthorEvaluatorResult,
    EvaluatorRegistryRecord,
    ReferenceManifest,
)
from .github_publish import (
    GhGitHubClient,
    GitHubPublication,
    GitHubPublisher,
    _gh_environment,
    _run_command,
)
from .host_oss import read_host_oss_object

_MAX_TASK_CONTRACT_BYTES = 1024 * 1024
_MAX_RUBRIC_BYTES = 8 * 1024 * 1024
_MAX_REFERENCE_MANIFEST_BYTES = 1024 * 1024
_MAX_REFERENCE_FILE_BYTES = 128 * 1024 * 1024
_MAX_REFERENCE_TOTAL_BYTES = 256 * 1024 * 1024
_MAX_TEST_OUTPUT_BYTES = 1024 * 1024
_LOCAL_TEST_TIMEOUT_SECONDS = 1800


class AuthorAgentAdapter(Protocol):
    async def author(
        self,
        request: AuthorEvaluatorRequest,
        workspace: AuthorWorkspace,
        inputs: AuthorInputBundle,
    ) -> AuthoredEvaluator: ...


class GitHubPublisherAdapter(Protocol):
    async def publish(self, **kwargs: Any) -> GitHubPublication: ...


@dataclass(frozen=True)
class AuthorEvaluatorDependencies:
    registry: AuthorRegistry
    workspace_factory: Callable[
        [AuthorEvaluatorRequest],
        AbstractAsyncContextManager[AuthorWorkspace],
    ]
    input_loader: Callable[
        [AuthorEvaluatorRequest, AuthorWorkspace],
        Awaitable[AuthorInputBundle],
    ]
    author_agent: AuthorAgentAdapter
    publisher: GitHubPublisherAdapter
    local_test_runner: Callable[[AuthorWorkspace, float], Awaitable[None]]
    harbor_version: str
    rewardkit_version: str
    now: Callable[[], datetime]


async def author_evaluator(
    request: AuthorEvaluatorRequest,
    *,
    dependencies: AuthorEvaluatorDependencies | None = None,
) -> AuthorEvaluatorResult:
    """Author, validate, merge, and register one immutable evaluator version."""
    try:
        selected = dependencies or _default_dependencies(request)
        async with asyncio.timeout(request.timeout_seconds):
            return await _invoke_author_evaluator(request, selected)
    except TimeoutError:
        return _failure_result(request, "timeout", "author evaluator capability timed out")
    except Exception as exc:  # noqa: BLE001 - atomic boundary returns a typed failure.
        if isinstance(exc, AtomicInfrastructureError):
            return _failure_result(request, exc.category, exc.message)
        return _failure_result(
            request,
            "runtime",
            f"{type(exc).__name__} during author evaluator operation",
        )


async def _invoke_author_evaluator(
    request: AuthorEvaluatorRequest,
    dependencies: AuthorEvaluatorDependencies,
) -> AuthorEvaluatorResult:
    identity = _registry_identity(request)
    async with dependencies.registry.claim(identity) as transaction:
        existing = transaction.get_ready()
        if existing is not None:
            _validate_ready_record(request, existing)
            return _ready_result(request, existing)

        async with dependencies.workspace_factory(request) as workspace:
            inputs = await dependencies.input_loader(request, workspace)
            last_authoring_failure: AtomicInfrastructureError | None = None
            for attempt in range(request.max_retries + 1):
                try:
                    await dependencies.author_agent.author(request, workspace, inputs)
                    last_authoring_failure = None
                    break
                except AtomicInfrastructureError as exc:
                    last_authoring_failure = exc
                    if attempt == request.max_retries:
                        raise
            assert last_authoring_failure is None
            await dependencies.local_test_runner(workspace, float(request.timeout_seconds))
            publication = await dependencies.publisher.publish(
                repository_url=request.task_repository_url,
                task_path=request.task_path,
                evaluator_id=request.evaluator_id,
                workspace=workspace,
                timeout_seconds=float(request.timeout_seconds),
            )

            record = EvaluatorRegistryRecord(
                status="ready",
                task_path=request.task_path,
                variant_index=request.variant_index,
                task_commit=request.task_commit,
                evaluator_id=request.evaluator_id,
                evaluator_version=publication.evaluator_version,
                rubric_hash=request.rubric_hash,
                reference_manifest_uri=request.reference_manifest_uri,
                reference_manifest_hash=request.reference_manifest_hash,
                evaluator_sdk_version=request.evaluator_sdk_version,
                harbor_version=dependencies.harbor_version,
                rewardkit_version=dependencies.rewardkit_version,
                image_id=request.image_id,
                pull_request_url=publication.pull_request_url,
                ci_run_id=publication.ci_run_id,
                ready_at=dependencies.now(),
            )
            published = transaction.publish_ready(record)
            _validate_ready_record(request, published)
            return _ready_result(request, published)


def _default_dependencies(request: AuthorEvaluatorRequest) -> AuthorEvaluatorDependencies:
    registry_root = os.environ.get("ALE_AUTHOR_REGISTRY_ROOT")
    if not registry_root:
        raise AtomicInfrastructureError(
            "author_registry",
            "ALE_AUTHOR_REGISTRY_ROOT is not configured",
        )
    raw_command = os.environ.get("ALE_AUTHOR_COMMAND_JSON")
    if not raw_command:
        raise AtomicInfrastructureError(
            "author_agent",
            "ALE_AUTHOR_COMMAND_JSON is not configured",
        )
    try:
        command = json.loads(raw_command)
    except json.JSONDecodeError as exc:
        raise AtomicInfrastructureError(
            "author_agent",
            "ALE_AUTHOR_COMMAND_JSON is invalid",
        ) from exc
    if not isinstance(command, list) or any(not isinstance(item, str) for item in command):
        raise AtomicInfrastructureError(
            "author_agent",
            "ALE_AUTHOR_COMMAND_JSON must be a JSON argv array",
        )
    harbor_version = os.environ.get("ALE_HARBOR_VERSION")
    rewardkit_version = os.environ.get("ALE_REWARDKIT_VERSION")
    if not harbor_version or not rewardkit_version:
        raise AtomicInfrastructureError(
            "author_agent",
            "ALE_HARBOR_VERSION and ALE_REWARDKIT_VERSION must be configured",
        )
    return AuthorEvaluatorDependencies(
        registry=FilesystemAuthorRegistry(registry_root),
        workspace_factory=_default_workspace_factory,
        input_loader=_default_input_loader,
        author_agent=AuthorAgent(command, timeout_seconds=request.timeout_seconds),
        publisher=GitHubPublisher(GhGitHubClient()),
        local_test_runner=_default_local_test_runner,
        harbor_version=harbor_version,
        rewardkit_version=rewardkit_version,
        now=lambda: datetime.now(UTC),
    )


@asynccontextmanager
async def _default_workspace_factory(
    request: AuthorEvaluatorRequest,
):
    with tempfile.TemporaryDirectory(prefix="ale-author-source-") as temporary:
        temporary_root = Path(temporary)
        source = temporary_root / "source"
        empty_template = temporary_root / "empty-git-template"
        empty_template.mkdir(mode=0o700)
        private_home = temporary_root / "home"
        private_home.mkdir(mode=0o700)
        environment = _isolated_git_environment(private_home)
        askpass: Path | None = None
        token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
        if token is None:
            token_result = await _run_command(
                ("gh", "auth", "token"),
                cwd=None,
                environment=_gh_environment(),
                timeout_seconds=120,
                max_output_bytes=4096,
            )
            if token_result.returncode != 0:
                raise AtomicInfrastructureError(
                    "author_workspace",
                    "cannot resolve GitHub authentication",
                )
            try:
                token = token_result.stdout.decode("utf-8", errors="strict").strip()
            except UnicodeDecodeError as exc:
                raise AtomicInfrastructureError(
                    "author_workspace",
                    "gh returned invalid authentication data",
                ) from exc
        if token:
            askpass = temporary_root / "askpass"
            askpass.write_text(
                "#!/bin/sh\n"
                'case "$1" in\n'
                '  *Username*) printf "%s\\n" "x-access-token" ;;\n'
                '  *) printf "%s\\n" "$ALE_GITHUB_FETCH_TOKEN" ;;\n'
                "esac\n",
                encoding="utf-8",
            )
            askpass.chmod(0o700)
            environment.update(
                {
                    "GIT_ASKPASS": str(askpass),
                    "GIT_TERMINAL_PROMPT": "0",
                    "ALE_GITHUB_FETCH_TOKEN": token,
                }
            )
        commands = (
            ("git", "init", "--quiet", f"--template={empty_template}", str(source)),
            (
                "git",
                "-C",
                str(source),
                "fetch",
                "--quiet",
                "--no-tags",
                "--no-recurse-submodules",
                "--depth=1",
                request.task_repository_url,
                request.task_commit,
            ),
            (
                "git",
                "-C",
                str(source),
                "update-ref",
                "refs/heads/main",
                request.task_commit,
            ),
            ("git", "-C", str(source), "symbolic-ref", "HEAD", "refs/heads/main"),
            ("git", "-C", str(source), "read-tree", request.task_commit),
        )
        for command in commands:
            try:
                result = await _run_command(
                    command,
                    cwd=temporary_root,
                    environment=environment,
                    timeout_seconds=120,
                    max_output_bytes=1024 * 1024,
                )
            except AtomicInfrastructureError as exc:
                raise AtomicInfrastructureError(
                    "author_workspace",
                    "bounded Git source materialization failed",
                ) from exc
            if result.returncode != 0:
                raise AtomicInfrastructureError(
                    "author_workspace",
                    "Git source materialization failed",
                )
        del askpass
        with isolated_author_workspace(source, request) as workspace:
            yield workspace


async def _default_input_loader(
    request: AuthorEvaluatorRequest,
    workspace: AuthorWorkspace,
) -> AuthorInputBundle:
    rubric = await read_host_oss_object(
        request.rubric_uri,
        limit=_MAX_RUBRIC_BYTES,
        missing_ok=False,
        integrity_category="author_input",
    )
    reference_manifest_bytes = await read_host_oss_object(
        request.reference_manifest_uri,
        limit=_MAX_REFERENCE_MANIFEST_BYTES,
        missing_ok=False,
        integrity_category="author_input",
    )
    assert rubric is not None
    assert reference_manifest_bytes is not None
    try:
        reference_manifest = ReferenceManifest.model_validate_json(reference_manifest_bytes)
    except ValidationError as exc:
        raise AtomicInfrastructureError(
            "author_input",
            "invalid reference manifest",
        ) from exc
    reference_root = request.reference_manifest_uri.rsplit("/", 1)[0]
    artifacts: dict[str, bytes] = {}
    total_bytes = 0
    for entry in reference_manifest.files:
        path = _canonical_reference_path(entry.path)
        payload = await read_host_oss_object(
            f"{reference_root}/{path.as_posix()}",
            limit=_MAX_REFERENCE_FILE_BYTES,
            missing_ok=False,
            integrity_category="author_input",
        )
        assert payload is not None
        total_bytes += len(payload)
        if total_bytes > _MAX_REFERENCE_TOTAL_BYTES:
            raise AtomicInfrastructureError(
                "author_input",
                "reference artifacts exceed 256 MiB",
            )
        artifacts[entry.path] = payload

    task_card = workspace.task_directory / "task_card.json"
    try:
        status_result = task_card.lstat()
        if not stat.S_ISREG(status_result.st_mode):
            raise AtomicInfrastructureError(
                "author_input",
                "task_card.json is not a regular file",
            )
        if status_result.st_size > _MAX_TASK_CONTRACT_BYTES:
            raise AtomicInfrastructureError(
                "author_input",
                "task_card.json exceeds 1 MiB",
            )
        task_card_bytes = task_card.read_bytes()
    except AtomicInfrastructureError:
        raise
    except OSError as exc:
        raise AtomicInfrastructureError(
            "author_input",
            "cannot read task_card.json",
        ) from exc
    sdk_contract = json.dumps(
        {
            "evaluator_sdk_version": request.evaluator_sdk_version,
            "implementation_modes": ["programmatic", "llm_judge", "hybrid"],
            "reward_path": "/logs/verifier/reward.json",
            "details_path": "/logs/verifier/reward-details.json",
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    image_capability = json.dumps(
        {"image_id": request.image_id},
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return AuthorInputBundle(
        task_contract_files={"task_card.json": task_card_bytes},
        rubric=rubric,
        reference_manifest=reference_manifest_bytes,
        reference_artifacts=artifacts,
        evaluator_sdk_contract=sdk_contract,
        image_capability_statement=image_capability,
    )


async def _default_local_test_runner(
    workspace: AuthorWorkspace,
    timeout_seconds: float,
) -> None:
    evaluator = workspace.task_directory / "evaluator"
    test_script = evaluator / "test.sh"
    tests = evaluator / "tests"
    if test_script.is_file() and not test_script.is_symlink():
        command: Sequence[str] = ("bash", str(test_script))
    elif tests.is_dir() and not tests.is_symlink():
        command = (sys.executable, "-m", "pytest", str(tests))
    else:
        raise AtomicInfrastructureError(
            "author_tests",
            "authored evaluator has no local test entrypoint",
        )
    environment = {
        key: os.environ[key] for key in ("PATH", "TMPDIR", "LANG", "LC_ALL") if key in os.environ
    }
    environment["PYTHONPATH"] = str(workspace.root)
    try:
        result = await _run_command(
            command,
            cwd=evaluator,
            environment=environment,
            timeout_seconds=min(timeout_seconds, _LOCAL_TEST_TIMEOUT_SECONDS),
            max_output_bytes=_MAX_TEST_OUTPUT_BYTES,
        )
    except AtomicInfrastructureError as exc:
        raise AtomicInfrastructureError(
            "author_tests",
            "local evaluator tests could not complete",
        ) from exc
    if result.returncode != 0:
        raise AtomicInfrastructureError(
            "author_tests",
            "local evaluator tests failed",
        )


def _canonical_reference_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not path.parts
        or path.is_absolute()
        or path.as_posix() == "."
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise AtomicInfrastructureError(
            "author_input",
            f"unsafe reference artifact path: {value}",
        )
    return path


def _registry_identity(request: AuthorEvaluatorRequest) -> AuthorEvaluatorRegistryIdentity:
    return AuthorEvaluatorRegistryIdentity(
        task_commit=request.task_commit,
        rubric_hash=request.rubric_hash,
        reference_manifest_hash=request.reference_manifest_hash,
        evaluator_sdk_version=request.evaluator_sdk_version,
        image_id=request.image_id,
    )


def _validate_ready_record(
    request: AuthorEvaluatorRequest,
    record: EvaluatorRegistryRecord,
) -> None:
    expected: Mapping[str, object] = {
        "task_path": request.task_path,
        "variant_index": request.variant_index,
        "task_commit": request.task_commit,
        "evaluator_id": request.evaluator_id,
        "rubric_hash": request.rubric_hash,
        "reference_manifest_uri": request.reference_manifest_uri,
        "reference_manifest_hash": request.reference_manifest_hash,
        "evaluator_sdk_version": request.evaluator_sdk_version,
        "image_id": request.image_id,
    }
    mismatches = [field for field, value in expected.items() if getattr(record, field) != value]
    if mismatches:
        raise AtomicInfrastructureError(
            "author_registry",
            f"ready evaluator record mismatch: {', '.join(mismatches)}",
        )


def _ready_result(
    request: AuthorEvaluatorRequest,
    record: EvaluatorRegistryRecord,
) -> AuthorEvaluatorResult:
    return AuthorEvaluatorResult(
        authoring_id=request.authoring_id,
        status="ready",
        evaluator_id=request.evaluator_id,
        evaluator_version=record.evaluator_version,
        pull_request_url=record.pull_request_url,
        ci_run_id=record.ci_run_id,
    )


def _failure_result(
    request: AuthorEvaluatorRequest,
    category: str,
    detail: str,
) -> AuthorEvaluatorResult:
    safe_category = category if category and len(category) <= 64 else "runtime"
    safe_detail = detail.strip() or "author evaluator operation failed"
    secret_values = sorted(
        {
            value
            for key, value in os.environ.items()
            if value
            and len(value) >= 4
            and any(marker in key.upper() for marker in ("TOKEN", "KEY", "SECRET", "PASSWORD"))
        },
        key=len,
        reverse=True,
    )
    for secret in secret_values:
        safe_detail = safe_detail.replace(secret, "[redacted]")
    return AuthorEvaluatorResult(
        authoring_id=request.authoring_id,
        status="authoring_failed",
        evaluator_id=request.evaluator_id,
        error_category=safe_category,
        error_detail=safe_detail[:4000],
    )
