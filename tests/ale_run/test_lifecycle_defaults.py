from ale_run.orchestration.lifecycle import _DEFAULT_TIMEOUT_S


def test_default_agent_wall_time_is_five_hours() -> None:
    assert _DEFAULT_TIMEOUT_S == 18_000
