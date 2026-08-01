from ale_run.orchestration.run_writer import RunWriter


def test_run_ids_are_unique_within_the_same_second(tmp_path) -> None:
    first = RunWriter(
        output_root=tmp_path,
        agent_id="agent",
        model="model",
        task_path="task",
        variant_index=0,
    )
    second = RunWriter(
        output_root=tmp_path,
        agent_id="agent",
        model="model",
        task_path="task",
        variant_index=0,
    )

    try:
        assert first.run_id != second.run_id
        assert first.run_dir != second.run_dir
    finally:
        first.close()
        second.close()
