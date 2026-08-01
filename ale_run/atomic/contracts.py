"""Immutable versioned data contracts for solve and evaluate capabilities."""

from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    model_serializer,
    model_validator,
)


def _validate_task_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or value == "."
        or "\x00" in value
        or "\\" in value
        or path.is_absolute()
        or PureWindowsPath(value).is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise ValueError("task_path must be a canonical relative POSIX path")
    return value


TaskPath = Annotated[
    str,
    Field(min_length=1, max_length=1_000),
    AfterValidator(_validate_task_path),
]
CommitSha = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
AliyunImageId = Annotated[str, Field(pattern=r"^m-[A-Za-z0-9-]+$")]
OssRoot = Annotated[str, Field(pattern=r"^oss://")]
GitHubRepositoryUrl = Annotated[
    str,
    Field(
        pattern=r"^https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?$",
        max_length=1_000,
    ),
]
GitHubPullRequestUrl = Annotated[
    str,
    Field(pattern=r"^https://github\.com/[^/]+/[^/]+/pull/[1-9][0-9]*$"),
]
Score = Annotated[float, Field(ge=0, le=1)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
MediaType = Annotated[str, Field(pattern=r"^[^\s/]+/[^\s/]+$", max_length=255)]
NonEmptyString = Annotated[str, Field(min_length=1, max_length=1_000)]
ErrorCategory = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
ErrorDetail = Annotated[str, Field(min_length=1, max_length=4_000)]
AttemptId = Annotated[str, Field(min_length=1, max_length=255)]


class SolveRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    submission_id: UUID
    runtime_spec_path: Path
    task_repo: Path
    task_path: TaskPath
    variant_index: int = Field(ge=0)
    agent_id: str
    task_commit: CommitSha
    image_id: AliyunImageId
    submission_root: OssRoot


class EvaluateRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    submission_id: UUID
    runtime_spec_path: Path
    task_repo: Path
    task_path: TaskPath
    variant_index: int = Field(ge=0)
    task_commit: CommitSha
    image_id: AliyunImageId
    submission_root: OssRoot
    evaluator_id: str
    evaluator_version: CommitSha


class ArtifactEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    size_bytes: int = Field(ge=0, strict=True)
    sha256: Sha256
    media_type: MediaType


class SubmissionManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    status: Literal["submitted"] = "submitted"
    submission_id: UUID
    task_path: str
    variant_index: int = Field(ge=0)
    task_commit: CommitSha
    image_id: AliyunImageId
    ale_run_id: str
    agent_id: str
    model_id: str
    config_digest: str
    started_at: datetime
    completed_at: datetime
    artifacts: tuple[ArtifactEntry, ...] = Field(min_length=1)


class EvaluatorRegistryRecord(BaseModel):
    """Trusted control-plane authorization for one immutable evaluator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    status: Literal["ready"]
    task_path: NonEmptyString
    variant_index: int = Field(ge=0)
    task_commit: CommitSha
    evaluator_id: NonEmptyString
    evaluator_version: CommitSha
    rubric_hash: Sha256
    reference_manifest_uri: OssRoot
    reference_manifest_hash: Sha256
    evaluator_sdk_version: NonEmptyString
    harbor_version: NonEmptyString
    rewardkit_version: NonEmptyString
    image_id: AliyunImageId
    pull_request_url: GitHubPullRequestUrl
    ci_run_id: NonEmptyString
    ready_at: datetime


class ReferenceFileEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: NonEmptyString
    size_bytes: int = Field(ge=0, strict=True)
    sha256: Sha256


class ReferenceManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    files: tuple[ReferenceFileEntry, ...]


class RubricPlanEntry(BaseModel):
    """One expert rubric item's selected evaluator implementation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rubric_id: Annotated[str, Field(pattern=r"^[A-Za-z0-9_.-]+$", max_length=255)]
    implementation_mode: Literal["programmatic", "llm_judge", "hybrid"]
    weight: float = Field(ge=0, allow_inf_nan=False)
    score_min: float = Field(allow_inf_nan=False)
    score_max: float = Field(allow_inf_nan=False)
    required: bool
    rationale: NonEmptyString

    @model_validator(mode="after")
    def require_ordered_score_range(self) -> "RubricPlanEntry":
        if self.score_min > self.score_max:
            raise ValueError("rubric plan score_min must not exceed score_max")
        return self


class RubricPlan(BaseModel):
    """Complete one-to-one implementation plan for an immutable rubric."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    rubric_hash: Sha256
    items: tuple[RubricPlanEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def require_unique_rubric_ids(self) -> "RubricPlan":
        rubric_ids = [item.rubric_id for item in self.items]
        if len(rubric_ids) != len(set(rubric_ids)):
            raise ValueError("rubric plan contains duplicate rubric IDs")
        return self


class AuthorEvaluatorRegistryIdentity(BaseModel):
    """The complete immutable identity of an authored evaluator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    task_commit: CommitSha
    rubric_hash: Sha256
    reference_manifest_hash: Sha256
    evaluator_sdk_version: NonEmptyString
    image_id: AliyunImageId


class AuthorEvaluatorRequest(BaseModel):
    """Authoring inputs, deliberately isolated from candidate evaluation data."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    authoring_id: UUID
    task_repository_url: GitHubRepositoryUrl
    task_path: TaskPath
    variant_index: int = Field(ge=0)
    task_commit: CommitSha
    image_id: AliyunImageId
    rubric_uri: OssRoot
    rubric_hash: Sha256
    reference_manifest_uri: OssRoot
    reference_manifest_hash: Sha256
    evaluator_id: NonEmptyString
    evaluator_sdk_version: NonEmptyString
    max_retries: int = Field(default=2, ge=0, le=5, strict=True)
    timeout_seconds: int = Field(default=18_000, ge=1, le=18_000, strict=True)


class AuthorEvaluatorResult(BaseModel):
    """Terminal outcome of authoring a versioned evaluator identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    authoring_id: UUID
    status: Literal["ready", "authoring_failed"]
    evaluator_id: NonEmptyString
    evaluator_version: CommitSha | None = None
    pull_request_url: GitHubPullRequestUrl | None = None
    ci_run_id: NonEmptyString | None = None
    error_category: ErrorCategory | None = None
    error_detail: ErrorDetail | None = None

    @model_validator(mode="after")
    def require_consistent_authoring_fields(self) -> "AuthorEvaluatorResult":
        ready_fields = (
            self.evaluator_version,
            self.pull_request_url,
            self.ci_run_id,
        )
        error_fields = (self.error_category, self.error_detail)
        if self.status == "ready":
            if any(value is None for value in ready_fields) or any(
                value is not None for value in error_fields
            ):
                raise ValueError("ready results require publication fields and no error")
            return self
        if any(value is not None for value in ready_fields) or any(
            value is None for value in error_fields
        ):
            raise ValueError(
                "authoring_failed results require structured errors and no publication fields"
            )
        return self

    @model_serializer(mode="wrap")
    def omit_absent_fields(self, handler):
        return {key: value for key, value in handler(self).items() if value is not None}


class HarborProvenance(BaseModel):
    """Harbor RewardKit output recorded in every trusted score."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reward: dict[str, Any] = Field(min_length=1)
    reward_path: Literal["evidence/reward.json"]
    reward_size_bytes: int = Field(ge=0, strict=True)
    reward_sha256: Sha256
    details_path: Literal["evidence/reward-details.json"]
    details_size_bytes: int = Field(ge=0, strict=True)
    details_sha256: Sha256


class SolveResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    status: Literal["submitted", "failed"]
    submission_id: UUID
    manifest: SubmissionManifest | None = None
    error: str | None = None

    @model_validator(mode="after")
    def require_consistent_solve_fields(self) -> "SolveResult":
        if self.status == "submitted" and (self.manifest is None or self.error is not None):
            raise ValueError("submitted results require a manifest and no error")
        if self.status == "submitted" and self.manifest.submission_id != self.submission_id:
            raise ValueError(
                "submitted result manifest submission_id must match result submission_id"
            )
        if self.status == "failed" and (self.manifest is not None or not self.error):
            raise ValueError("failed results require an error and no manifest")
        return self


class EvaluationResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    status: Literal["scored", "infra_failed"]
    submission_id: UUID
    task_path: NonEmptyString
    variant_index: int = Field(ge=0)
    task_commit: CommitSha
    image_id: AliyunImageId
    evaluator_id: NonEmptyString
    evaluator_version: CommitSha
    outcome: Literal["valid", "invalid_output"] | None = None
    score: Score | None = None
    rubric_hash: Sha256 | None = None
    harbor: HarborProvenance | None = None
    error_category: ErrorCategory | None = None
    error_detail: ErrorDetail | None = None
    attempt_id: AttemptId | None = None

    @model_validator(mode="after")
    def require_consistent_evaluation_fields(self) -> "EvaluationResult":
        error_fields = (self.error_category, self.error_detail, self.attempt_id)
        if self.status == "infra_failed":
            if self.outcome is not None or self.score is not None:
                raise ValueError("infra_failed results must not include outcome or score")
            if self.rubric_hash is not None or self.harbor is not None:
                raise ValueError("infra_failed results must not include scored provenance")
            if any(value is None for value in error_fields):
                raise ValueError("infra_failed results require category, detail, and attempt_id")
            return self
        if self.outcome is None or self.score is None:
            raise ValueError("scored results must include outcome and score")
        if self.rubric_hash is None or self.harbor is None:
            raise ValueError("scored results require rubric and Harbor provenance")
        if any(value is not None for value in error_fields):
            raise ValueError("scored results must not include infrastructure errors")
        if self.outcome == "invalid_output" and self.score != 0.0:
            raise ValueError("invalid_output requires an exact 0.0 score")
        return self

    @model_serializer(mode="wrap")
    def omit_absent_fields(self, handler):
        return {key: value for key, value in handler(self).items() if value is not None}


class AtomicInfrastructureError(RuntimeError):
    """Infrastructure failure that prevents an atomic capability from running."""

    def __init__(self, category: str, message: str) -> None:
        self.category = category
        self.message = message
        super().__init__(f"{category}: {message}")
