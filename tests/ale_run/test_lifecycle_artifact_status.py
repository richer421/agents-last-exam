from ale_run.orchestration.lifecycle import (
    _promote_artifact_failure,
    _promote_cleanup_failure,
)


def test_output_upload_failure_invalidates_score() -> None:
    result = _promote_artifact_failure(
        status="completed",
        score=0.91,
        gather_report={
            "status": "failed",
            "transport": "oss",
            "error": "upload failed",
        },
    )

    assert result == (
        "infra_error",
        None,
        "artifact output upload failed: upload failed",
    )


def test_successful_output_upload_preserves_score() -> None:
    result = _promote_artifact_failure(
        status="completed",
        score=0.91,
        gather_report={
            "status": "success",
            "transport": "oss",
            "oss_path": "oss://bucket/task/runs/run/output/",
        },
    )

    assert result == ("completed", 0.91, None)


def test_partial_output_download_invalidates_score() -> None:
    result = _promote_artifact_failure(
        status="completed",
        score=0.91,
        gather_report={
            "status": "success",
            "transport": "cua",
            "errors": [{"path": "final_result.psd", "error": "download_failed"}],
        },
    )

    assert result == (
        "infra_error",
        None,
        "artifact output upload failed: partial artifact transfer failed",
    )


def test_artifact_failure_clears_timeout_score_without_hiding_timeout() -> None:
    result = _promote_artifact_failure(
        status="timeout",
        score=0.91,
        gather_report={
            "status": "failed",
            "transport": "oss",
            "error": "upload failed",
        },
    )

    assert result == ("timeout", None, None)


def test_cleanup_failure_promotes_completed_run_to_infra_error() -> None:
    result = _promote_cleanup_failure(
        status="completed",
        score=0.91,
        message="sandbox instance-1 release failed: delete failed",
    )

    assert result == (
        "infra_error",
        None,
        "sandbox instance-1 release failed: delete failed",
    )


def test_cleanup_failure_preserves_timeout_but_clears_score() -> None:
    result = _promote_cleanup_failure(
        status="timeout",
        score=0.91,
        message="sandbox instance-1 release failed: delete failed",
    )

    assert result == ("timeout", None, None)
