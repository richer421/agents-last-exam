"""Two-phase, input-isolated AI evaluator author adapter."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import shutil
import signal
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pydantic import ValidationError

from .author_workspace import AuthorWorkspace
from .contracts import (
    AtomicInfrastructureError,
    AuthorEvaluatorRequest,
    ReferenceManifest,
    RubricPlan,
)

_MAX_INPUT_FILE_BYTES = 128 * 1024 * 1024
_MAX_INPUT_BYTES = 256 * 1024 * 1024
_MAX_OUTPUT_FILE_BYTES = 8 * 1024 * 1024
_MAX_OUTPUT_BYTES = 32 * 1024 * 1024
_MAX_STAGED_ENTRIES = 4096
_DEFAULT_MAX_OUTPUT_BYTES = 1024 * 1024
_PROCESS_KILL_WAIT_SECONDS = 1


@dataclass(frozen=True)
class AuthorInputBundle:
    task_contract_files: Mapping[str, bytes]
    rubric: bytes
    reference_manifest: bytes
    reference_artifacts: Mapping[str, bytes]
    evaluator_sdk_contract: bytes
    image_capability_statement: bytes


@dataclass(frozen=True)
class AuthoredEvaluator:
    rubric_plan: RubricPlan
    files: tuple[str, ...]


@dataclass(frozen=True)
class _ExpertRubricItem:
    rubric_id: str
    weight: float
    score_min: float
    score_max: float
    required: bool


class AuthorAgent:
    """Invoke one configured argv command for planning and implementation."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: float,
        max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES,
    ) -> None:
        self.command = tuple(command)
        if not self.command or any(
            not isinstance(argument, str) or not argument or "\0" in argument
            for argument in self.command
        ):
            raise AtomicInfrastructureError(
                "author_agent",
                "author command must be a non-empty argv sequence",
            )
        if timeout_seconds <= 0:
            raise AtomicInfrastructureError(
                "author_agent",
                "author command timeout must be positive",
            )
        if max_output_bytes <= 0:
            raise AtomicInfrastructureError(
                "author_agent",
                "author command output limit must be positive",
            )
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes

    async def author(
        self,
        request: AuthorEvaluatorRequest,
        workspace: AuthorWorkspace,
        inputs: AuthorInputBundle,
    ) -> AuthoredEvaluator:
        expert_items, reference_manifest = _validate_inputs(request, inputs)
        with tempfile.TemporaryDirectory(prefix="ale-author-input-") as temporary:
            stage = Path(temporary)
            output = _stage_inputs(stage, inputs, reference_manifest)
            input_fingerprint = _input_fingerprint(stage / "inputs")

            await self._invoke(stage, request, output, phase="plan")
            _require_unchanged_inputs(stage / "inputs", input_fingerprint)
            phase_one_files = _validate_output_tree(output)
            if phase_one_files != ("rubric-plan.json",):
                raise AtomicInfrastructureError(
                    "author_agent",
                    "author created implementation files before rubric plan validation",
                )
            plan_before = _load_rubric_plan(output / "rubric-plan.json")
            _validate_rubric_coverage(plan_before, request, expert_items)

            await self._invoke(stage, request, output, phase="implement")
            _require_unchanged_inputs(stage / "inputs", input_fingerprint)
            files = _validate_output_tree(output)
            plan_after = _load_rubric_plan(output / "rubric-plan.json")
            if plan_after != plan_before:
                raise AtomicInfrastructureError(
                    "author_agent",
                    "rubric plan changed after implementation",
                )
            _validate_rubric_coverage(plan_after, request, expert_items)
            _publish_to_workspace(output, workspace.task_directory / "evaluator", files)
            return AuthoredEvaluator(rubric_plan=plan_after, files=files)

    async def _invoke(
        self,
        stage: Path,
        request: AuthorEvaluatorRequest,
        output: Path,
        *,
        phase: str,
    ) -> None:
        environment = {
            key: os.environ[key]
            for key in (
                "PATH",
                "TMPDIR",
                "LANG",
                "LC_ALL",
                "SSL_CERT_FILE",
                "TRUE_SOTA_API_KEY",
                "ALE_AUTHOR_MODEL",
                "ALE_AUTHOR_REASONING_EFFORT",
            )
            if key in os.environ
        }
        private_home = stage / "home"
        private_home.mkdir(mode=0o700, exist_ok=True)
        inputs = stage / "inputs"
        environment.update(
            {
                "HOME": str(private_home),
                "XDG_CONFIG_HOME": str(private_home / ".config"),
                "ALE_AUTHOR_PHASE": phase,
                "ALE_AUTHOR_TASK_PATH": request.task_path,
                "ALE_AUTHOR_TASK_COMMIT": request.task_commit,
                "ALE_AUTHOR_IMAGE_ID": request.image_id,
                "ALE_AUTHOR_TASK_CONTRACT_DIR": str(inputs / "task"),
                "ALE_AUTHOR_RUBRIC_PATH": str(inputs / "rubric" / "rubrics.json"),
                "ALE_AUTHOR_REFERENCE_DIR": str(inputs / "reference"),
                "ALE_AUTHOR_SDK_CONTRACT_PATH": str(inputs / "sdk" / "contract.json"),
                "ALE_AUTHOR_IMAGE_CAPABILITY_PATH": str(inputs / "image" / "capabilities.json"),
                "ALE_AUTHOR_OUTPUT_DIR": str(output),
            }
        )
        try:
            process = await asyncio.create_subprocess_exec(
                *self.command,
                cwd=stage,
                env=environment,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            raise AtomicInfrastructureError(
                "author_agent",
                f"cannot start author command: {exc}",
            ) from exc
        assert process.stdout is not None
        assert process.stderr is not None
        readers = (
            asyncio.create_task(_read_bounded(process.stdout, self.max_output_bytes)),
            asyncio.create_task(_read_bounded(process.stderr, self.max_output_bytes)),
        )
        try:
            async with asyncio.timeout(self.timeout_seconds):
                stdout, stderr, returncode = await asyncio.gather(
                    *readers,
                    process.wait(),
                )
            if returncode != 0:
                del stdout, stderr
                raise AtomicInfrastructureError(
                    "author_agent",
                    f"author command failed with exit code {returncode}",
                )
        except TimeoutError as exc:
            raise AtomicInfrastructureError(
                "author_agent",
                "author command timed out",
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
                    await asyncio.wait_for(
                        process.wait(),
                        timeout=_PROCESS_KILL_WAIT_SECONDS,
                    )
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
                "author_agent",
                "author command output exceeds its limit",
            )


def _validate_inputs(
    request: AuthorEvaluatorRequest,
    inputs: AuthorInputBundle,
) -> tuple[tuple[_ExpertRubricItem, ...], ReferenceManifest]:
    if hashlib.sha256(inputs.rubric).hexdigest() != request.rubric_hash:
        raise AtomicInfrastructureError("author_agent", "rubric hash mismatch")
    if hashlib.sha256(inputs.reference_manifest).hexdigest() != request.reference_manifest_hash:
        raise AtomicInfrastructureError(
            "author_agent",
            "reference manifest hash mismatch",
        )
    try:
        reference_manifest = ReferenceManifest.model_validate_json(inputs.reference_manifest)
    except ValidationError as exc:
        raise AtomicInfrastructureError(
            "author_agent",
            "invalid reference manifest",
        ) from exc
    expected_paths = {entry.path for entry in reference_manifest.files}
    if set(inputs.reference_artifacts) != expected_paths:
        raise AtomicInfrastructureError(
            "author_agent",
            "reference artifact set does not match manifest",
        )
    for entry in reference_manifest.files:
        artifact = inputs.reference_artifacts[entry.path]
        if (
            len(artifact) != entry.size_bytes
            or hashlib.sha256(artifact).hexdigest() != entry.sha256
        ):
            raise AtomicInfrastructureError(
                "author_agent",
                f"reference artifact verification failed: {entry.path}",
            )
    if "task_card.json" not in inputs.task_contract_files:
        raise AtomicInfrastructureError(
            "author_agent",
            "task contract is missing task_card.json",
        )
    if not inputs.evaluator_sdk_contract or not inputs.image_capability_statement:
        raise AtomicInfrastructureError(
            "author_agent",
            "SDK and image capability contracts must be non-empty",
        )
    return _parse_expert_rubric(inputs.rubric), reference_manifest


def _parse_expert_rubric(payload: bytes) -> tuple[_ExpertRubricItem, ...]:
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AtomicInfrastructureError("author_agent", "invalid expert rubric JSON") from exc
    raw_items = document.get("rubrics") if isinstance(document, dict) else None
    if not isinstance(raw_items, list) or not raw_items:
        raise AtomicInfrastructureError(
            "author_agent",
            "expert rubric must contain a non-empty rubrics list",
        )
    items: list[_ExpertRubricItem] = []
    identifiers: set[str] = set()
    for raw in raw_items:
        if not isinstance(raw, dict):
            raise AtomicInfrastructureError("author_agent", "invalid expert rubric item")
        rubric_id = raw.get("id")
        values = (raw.get("weight"), raw.get("score_min"), raw.get("score_max"))
        required = raw.get("required")
        if (
            not isinstance(rubric_id, str)
            or not rubric_id
            or rubric_id in identifiers
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) for value in values
            )
            or any(not math.isfinite(float(value)) for value in values)
            or not isinstance(required, bool)
        ):
            raise AtomicInfrastructureError("author_agent", "invalid expert rubric item")
        weight, score_min, score_max = (float(value) for value in values)
        if weight < 0 or score_min > score_max:
            raise AtomicInfrastructureError("author_agent", "invalid expert rubric item")
        identifiers.add(rubric_id)
        items.append(
            _ExpertRubricItem(
                rubric_id=rubric_id,
                weight=weight,
                score_min=score_min,
                score_max=score_max,
                required=required,
            )
        )
    return tuple(items)


def _stage_inputs(
    stage: Path,
    inputs: AuthorInputBundle,
    reference_manifest: ReferenceManifest,
) -> Path:
    input_root = stage / "inputs"
    output = stage / "output"
    input_root.mkdir(mode=0o700)
    output.mkdir(mode=0o700)
    files: dict[PurePosixPath, bytes] = {
        PurePosixPath("rubric/rubrics.json"): inputs.rubric,
        PurePosixPath("reference/manifest.json"): inputs.reference_manifest,
        PurePosixPath("sdk/contract.json"): inputs.evaluator_sdk_contract,
        PurePosixPath("image/capabilities.json"): inputs.image_capability_statement,
    }
    for path, payload in inputs.task_contract_files.items():
        files[PurePosixPath("task") / _validate_input_path(path)] = payload
    for entry in reference_manifest.files:
        files[PurePosixPath("reference/artifacts") / _validate_input_path(entry.path)] = (
            inputs.reference_artifacts[entry.path]
        )
    total = 0
    if len(files) > _MAX_STAGED_ENTRIES:
        raise AtomicInfrastructureError("author_agent", "too many staged author inputs")
    for relative, payload in files.items():
        if not isinstance(payload, bytes):
            raise AtomicInfrastructureError("author_agent", "author inputs must be bytes")
        if len(payload) > _MAX_INPUT_FILE_BYTES:
            raise AtomicInfrastructureError("author_agent", "staged author input exceeds 128 MiB")
        total += len(payload)
        if total > _MAX_INPUT_BYTES:
            raise AtomicInfrastructureError("author_agent", "staged author inputs exceed 256 MiB")
        destination = input_root.joinpath(*relative.parts)
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o400,
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
    for directory, directory_names, _file_names in os.walk(input_root, topdown=False):
        for name in directory_names:
            (Path(directory) / name).chmod(0o500)
        Path(directory).chmod(0o500)
    return output


def _validate_input_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not path.parts
        or path.is_absolute()
        or path.as_posix() == "."
        or any(part in {"", ".", "..", ".git"} for part in path.parts)
    ):
        raise AtomicInfrastructureError("author_agent", f"unsafe author input path: {value}")
    return path


def _input_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix().encode()
        status_result = path.lstat()
        digest.update(relative)
        digest.update(stat.S_IFMT(status_result.st_mode).to_bytes(4, "big"))
        if stat.S_ISREG(status_result.st_mode):
            payload = path.read_bytes()
            if len(payload) > _MAX_INPUT_FILE_BYTES:
                raise AtomicInfrastructureError("author_agent", "author input exceeds 128 MiB")
            digest.update(payload)
        elif not stat.S_ISDIR(status_result.st_mode):
            raise AtomicInfrastructureError("author_agent", "unsafe staged author input")
    return digest.hexdigest()


def _require_unchanged_inputs(root: Path, expected: str) -> None:
    if _input_fingerprint(root) != expected:
        raise AtomicInfrastructureError("author_agent", "author modified authoritative inputs")


def _validate_output_tree(root: Path) -> tuple[str, ...]:
    files: list[str] = []
    total = 0
    for entries, path in enumerate(sorted(root.rglob("*")), start=1):
        relative = path.relative_to(root)
        status_result = path.lstat()
        if entries > _MAX_STAGED_ENTRIES:
            raise AtomicInfrastructureError("author_agent", "too many authored output entries")
        if ".git" in relative.parts or stat.S_ISLNK(status_result.st_mode):
            raise AtomicInfrastructureError("author_agent", "unsafe authored output entry")
        if stat.S_ISREG(status_result.st_mode):
            if status_result.st_size > _MAX_OUTPUT_FILE_BYTES:
                raise AtomicInfrastructureError("author_agent", "authored output exceeds 8 MiB")
            total += status_result.st_size
            if total > _MAX_OUTPUT_BYTES:
                raise AtomicInfrastructureError("author_agent", "authored outputs exceed 32 MiB")
            files.append(relative.as_posix())
        elif not stat.S_ISDIR(status_result.st_mode):
            raise AtomicInfrastructureError("author_agent", "unsafe authored output entry")
    if "rubric-plan.json" not in files:
        raise AtomicInfrastructureError("author_agent", "rubric-plan.json is required")
    return tuple(files)


def _load_rubric_plan(path: Path) -> RubricPlan:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise AtomicInfrastructureError("author_agent", "cannot read rubric-plan.json") from exc
    if len(payload) > _MAX_OUTPUT_FILE_BYTES:
        raise AtomicInfrastructureError("author_agent", "rubric-plan.json exceeds 8 MiB")
    try:
        return RubricPlan.model_validate_json(payload)
    except ValidationError as exc:
        raise AtomicInfrastructureError("author_agent", "invalid rubric-plan.json") from exc


def _validate_rubric_coverage(
    plan: RubricPlan,
    request: AuthorEvaluatorRequest,
    expert_items: tuple[_ExpertRubricItem, ...],
) -> None:
    if plan.rubric_hash != request.rubric_hash:
        raise AtomicInfrastructureError("author_agent", "rubric plan hash mismatch")
    planned = {item.rubric_id: item for item in plan.items}
    expected = {item.rubric_id: item for item in expert_items}
    if set(planned) != set(expected):
        raise AtomicInfrastructureError("author_agent", "rubric plan coverage mismatch")
    for rubric_id, expert in expected.items():
        item = planned[rubric_id]
        if (
            item.weight != expert.weight
            or item.score_min != expert.score_min
            or item.score_max != expert.score_max
            or item.required != expert.required
        ):
            raise AtomicInfrastructureError(
                "author_agent",
                f"rubric plan values changed for {rubric_id}",
            )


def _publish_to_workspace(
    output: Path,
    destination: Path,
    files: tuple[str, ...],
) -> None:
    if destination.exists():
        raise AtomicInfrastructureError(
            "author_agent",
            "evaluator destination was not scrubbed before authoring",
        )
    destination.mkdir(mode=0o755)
    try:
        for relative_value in files:
            relative = _validate_input_path(relative_value)
            source = output.joinpath(*relative.parts)
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
            descriptor = os.open(
                target,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                stat.S_IMODE(source.stat(follow_symlinks=False).st_mode) & 0o755 or 0o644,
            )
            with os.fdopen(descriptor, "wb") as stream, source.open("rb") as source_stream:
                shutil.copyfileobj(source_stream, stream)
    except BaseException:
        shutil.rmtree(destination)
        raise
