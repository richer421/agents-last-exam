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

__all__ = [
    "ArtifactEntry",
    "AtomicInfrastructureError",
    "EvaluateRequest",
    "EvaluationResult",
    "SolveRequest",
    "SolveResult",
    "SubmissionManifest",
    "evaluate",
    "solve",
]


def __getattr__(name: str):
    if name == "evaluate":
        from ale_run.atomic.evaluate import evaluate

        return evaluate
    if name == "solve":
        from ale_run.atomic.solve import solve

        return solve
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
