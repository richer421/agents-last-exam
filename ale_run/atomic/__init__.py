"""Versioned contracts for independent atomic ALE capabilities."""

from ale_run.atomic.contracts import (
    ArtifactEntry,
    AtomicInfrastructureError,
    EvaluateRequest,
    EvaluationResult,
    SolveRequest,
    SolveResult,
    SubmissionManifest,
)
from ale_run.atomic.solve import solve

__all__ = [
    "ArtifactEntry",
    "AtomicInfrastructureError",
    "EvaluateRequest",
    "EvaluationResult",
    "SolveRequest",
    "SolveResult",
    "SubmissionManifest",
    "solve",
]
