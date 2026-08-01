import json

from ale_run.agents.codex.config import _sanitise_catalog_for_fork


def test_sanitise_catalog_adds_reasoning_summary_capability_for_new_catalogs() -> None:
    raw = json.dumps(
        {
            "models": [
                {
                    "slug": "gpt-5.6-sol",
                    "default_reasoning_summary": "none",
                    "supported_reasoning_levels": [{"effort": "low"}],
                }
            ]
        }
    )

    catalog = json.loads(_sanitise_catalog_for_fork(raw))

    assert catalog["models"][0]["supports_reasoning_summaries"] is False
