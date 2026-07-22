from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pandas")

from existing_code_reuse.capabilities import seed_capabilities
from existing_code_reuse.execution import (
    ExecutionTask,
    RouteStep,
    execute_route,
    plan_route,
    validate_route,
)


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "seed" / "transactions.csv"


def _composition_steps(output: Path, *, residual: bool = False) -> tuple[RouteStep, ...]:
    steps = [
        RouteStep("pandas.read_csv", {}),
        RouteStep("pandas.DataFrame.fillna", {"value": {"amount": 0}}),
        RouteStep("pandas.DataFrame.groupby", {"by": "category"}),
        RouteStep("pandas.core.groupby.DataFrameGroupBy.sum", {"column": "amount"}),
        RouteStep("pandas.Series.reset_index", {"name": "amount"}),
    ]
    if residual:
        steps.append(
            RouteStep(
                "residual.category_adjustment",
                {
                    "category": "B",
                    "category_column": "category",
                    "value_column": "amount",
                    "multiplier": 1.1,
                },
            )
        )
    steps.append(RouteStep("pandas.DataFrame.to_json", {"path": str(output)}))
    return tuple(steps)


def test_route_planner_finds_required_typed_stages() -> None:
    route = plan_route(
        "file/csv",
        "file/json-records",
        seed_capabilities(),
        required_stages=("load", "missing-values", "group", "aggregate", "adapt", "write"),
    )
    assert route.status == "route_found"
    assert route.steps[0] == "pandas.read_csv"
    assert route.steps[-1] == "pandas.DataFrame.to_json"
    assert "pandas.core.groupby.DataFrameGroupBy.sum" in route.steps


def test_incompatible_reader_is_rejected_before_execution() -> None:
    valid, reason = validate_route(
        ("pandas.read_json",),
        input_artifact_type="file/csv",
        required_output_type="table/pandas-dataframe",
    )
    assert valid is False
    assert "expects file/json" in reason

    task = ExecutionTask(
        task_id="hard-negative",
        prompt="Read CSV, not JSON",
        input_artifact_type="file/csv",
        required_output_type="table/pandas-dataframe",
        required_stages=("load",),
        input_path=str(INPUT),
        expected_rows=None,
    )
    receipt = execute_route(task, (RouteStep("pandas.read_json", {}),))
    assert receipt["status"] == "rejected_before_execution"
    assert receipt["nodes"] == []


def test_no_reuse_is_a_valid_planning_result() -> None:
    route = plan_route(
        "file/csv",
        "file/company-ledger-v7",
        seed_capabilities(),
        required_stages=("company-policy-v7",),
    )
    assert route.status == "no_valid_route"
    assert route.steps == ()


def test_direct_reuse_executes_and_checks_table_contract() -> None:
    task = ExecutionTask(
        task_id="direct",
        prompt="Load CSV",
        input_artifact_type="file/csv",
        required_output_type="table/pandas-dataframe",
        required_stages=("load",),
        input_path=str(INPUT),
        expected_rows=None,
        expected_columns=("category", "amount"),
        expected_row_count=5,
    )
    receipt = execute_route(task, (RouteStep("pandas.read_csv", {}),))
    assert receipt["accepted"] is True
    assert receipt["nodes"][0]["contribution_state"] == "accepted_output"
    assert receipt["model_usage"]["calls"] == 0


def test_composed_route_executes_and_traces_consumption(tmp_path: Path) -> None:
    output = tmp_path / "grouped.json"
    task = ExecutionTask(
        task_id="composition",
        prompt="Aggregate transactions",
        input_artifact_type="file/csv",
        required_output_type="file/json-records",
        required_stages=("load", "missing-values", "group", "aggregate", "adapt", "write"),
        input_path=str(INPUT),
        expected_rows=(
            {"category": "A", "amount": 10.0},
            {"category": "B", "amount": 12.0},
            {"category": "C", "amount": 2.0},
        ),
    )
    receipt = execute_route(task, _composition_steps(output))
    assert receipt["accepted"] is True
    assert json.loads(output.read_text()) == list(task.expected_rows or ())
    assert len(receipt["nodes"]) == 6
    assert all(node["consumed_by"] for node in receipt["nodes"])
    assert receipt["nodes"][-1]["contribution_state"] == "accepted_output"
    assert receipt["residual_authored_lines"] == 0


def test_residual_route_counts_only_custom_step(tmp_path: Path) -> None:
    output = tmp_path / "adjusted.json"
    task = ExecutionTask(
        task_id="residual",
        prompt="Apply custom B adjustment after standard aggregation",
        input_artifact_type="file/csv",
        required_output_type="file/json-records",
        required_stages=("aggregate", "residual-custom", "write"),
        input_path=str(INPUT),
        expected_rows=(
            {"category": "A", "amount": 10.0},
            {"category": "B", "amount": 13.2},
            {"category": "C", "amount": 2.0},
        ),
        allow_residual=True,
    )
    receipt = execute_route(task, _composition_steps(output, residual=True))
    assert receipt["accepted"] is True
    assert receipt["residual_authored_lines"] == 7
    assert sum(node["capability_id"].startswith("residual.") for node in receipt["nodes"]) == 1

