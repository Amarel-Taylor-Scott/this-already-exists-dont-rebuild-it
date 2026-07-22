from __future__ import annotations

import json
from pathlib import Path

from existing_code_reuse.cli import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPACE_PATH = PROJECT_ROOT / "configs" / "experiment_space.json"


def test_design_cli_writes_a_bounded_schedule(tmp_path) -> None:
    output = tmp_path / "design.json"

    result = main(
        [
            "design-experiments",
            "--strategy",
            "mixed_screen",
            "--max-experiments",
            "5",
            "--max-resource-tier",
            "cpu_small",
            "--budget",
            "smoke",
            "--output",
            str(output),
        ]
    )

    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert result == 0
    assert manifest["requested_configuration_count"] == 5
    assert manifest["scheduled_configuration_count"] == 5
    assert manifest["underfilled"] is False
    assert manifest["runner_coverage"]["full_configuration_materializer"] is False
    assert all(
        "stage_configuration_keys" in item for item in manifest["configurations"]
    )


def test_custom_registry_controls_budget_and_resource_names(tmp_path) -> None:
    document = json.loads(SPACE_PATH.read_text(encoding="utf-8"))
    document["resource_tiers"].append("custom_accelerator")
    document["budgets"].append(
        {
            "budget_id": "custom_smoke",
            "task_limit": 1,
            "package_limit": 1,
            "query_repeats": 1,
            "learned_seeds": 1,
            "route_execution_limit": 0,
        }
    )
    registry = tmp_path / "custom-space.json"
    registry.write_text(json.dumps(document), encoding="utf-8")
    output = tmp_path / "custom-design.json"

    result = main(
        [
            "design-experiments",
            "--space",
            str(registry),
            "--strategy",
            "baseline",
            "--max-experiments",
            "1",
            "--max-resource-tier",
            "custom_accelerator",
            "--budget",
            "custom_smoke",
            "--output",
            str(output),
        ]
    )

    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert result == 0
    assert manifest["max_resource_tier"] == "custom_accelerator"
    assert manifest["trials"][0]["budget_id"] == "custom_smoke"
