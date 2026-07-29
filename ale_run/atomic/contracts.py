"""Immutable versioned data contracts for solve and evaluate capabilities."""

from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

CommitSha = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
AliyunImageId = Annotated[str, Field(pattern=r"^m-[A-Za-z0-9-]+$")]
OssRoot = Annotated[str, Field(pattern=r"^oss://")]
Score = Annotated[float, Field(ge=0, le=1)]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class SolveRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    submission_id: UUID
    runtime_spec_path: Path
    task_repo: Path
    task_path: str
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
    task_path: str
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


class SubmissionManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
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
        if self.status == "failed" and (self.manifest is not None or not self.error):
            raise ValueError("failed results require an error and no manifest")
        return self


class EvaluationResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    status: Literal["scored", "infra_failed"]
    outcome: Literal["valid", "invalid_output"] | None = None
    score: Score | None = None

    @model_validator(mode="after")
    def require_consistent_evaluation_fields(self) -> "EvaluationResult":
        if self.status == "infra_failed" and (self.outcome is not None or self.score is not None):
            raise ValueError("infra_failed results must not include outcome or score")
        if self.status == "scored" and (self.outcome is None or self.score is None):
            raise ValueError("scored results must include outcome and score")
        return self


class AtomicInfrastructureError(RuntimeError):
    """Infrastructure failure that prevents an atomic capability from running."""

    def __init__(self, category: str, message: str) -> None:
        self.category = category
        self.message = message
        super().__init__(f"{category}: {message}")
