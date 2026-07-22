"""Typed planning, allowlisted execution, contribution tracing, and independent evaluation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time
from typing import Any, Callable, Iterable

from .capabilities import VerifiedCapability, seed_capabilities
from .models import digest_value


@dataclass(frozen=True, slots=True)
class RouteStep:
    capability_id: str
    bindings: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ExecutionTask:
    task_id: str
    prompt: str
    input_artifact_type: str
    required_output_type: str
    required_stages: tuple[str, ...]
    input_path: str
    expected_rows: tuple[dict[str, Any], ...] | None
    expected_columns: tuple[str, ...] | None = None
    expected_row_count: int | None = None
    allow_residual: bool = False


@dataclass(frozen=True, slots=True)
class PlannedRoute:
    status: str
    steps: tuple[str, ...]
    reason: str


def plan_route(
    input_artifact_type: str,
    required_output_type: str,
    capabilities: Iterable[VerifiedCapability],
    *,
    required_stages: Iterable[str] = (),
    allow_residual: bool = False,
    max_steps: int = 8,
) -> PlannedRoute:
    """Find a short type-correct route while tracking required workflow stages.

    This is intentionally a small breadth-first baseline, not a claim that general software
    composition is a simple shortest-path problem.  Its state and limits are explicit so it can be
    compared with graph, constraint, and learned planners later.
    """

    required = frozenset(required_stages)
    ordered = sorted(capabilities, key=lambda item: item.capability_id)
    start = (input_artifact_type, frozenset())
    queue: deque[tuple[str, frozenset[str], tuple[str, ...]]] = deque(
        [(start[0], start[1], ())]
    )
    seen: set[tuple[str, frozenset[str]]] = {start}

    while queue:
        artifact_type, stages, steps = queue.popleft()
        if artifact_type == required_output_type and required.issubset(stages):
            return PlannedRoute("route_found", steps, "type and required-stage constraints satisfied")
        if len(steps) >= max_steps:
            continue
        for capability in ordered:
            if capability.capability_id == "residual.category_adjustment" and not allow_residual:
                continue
            if not capability.inputs or not capability.outputs:
                continue
            if capability.inputs[0].artifact_type != artifact_type:
                continue
            next_stages = stages | {capability.workflow_stage}
            state = (capability.outputs[0].artifact_type, next_stages)
            if state in seen:
                continue
            seen.add(state)
            queue.append((state[0], state[1], steps + (capability.capability_id,)))

    return PlannedRoute(
        "no_valid_route",
        (),
        "no type-correct route satisfies the required output and workflow stages",
    )


def validate_route(
    capability_ids: Iterable[str],
    *,
    input_artifact_type: str,
    required_output_type: str,
    capabilities: Iterable[VerifiedCapability] | None = None,
) -> tuple[bool, str]:
    catalog = {item.capability_id: item for item in (capabilities or seed_capabilities())}
    current = input_artifact_type
    for capability_id in capability_ids:
        capability = catalog.get(capability_id)
        if capability is None:
            return False, f"unknown capability: {capability_id}"
        expected = capability.inputs[0].artifact_type
        if expected != current:
            return False, f"{capability_id} expects {expected}, received {current}"
        current = capability.outputs[0].artifact_type
    if current != required_output_type:
        return False, f"route produces {current}, required {required_output_type}"
    return True, "compatible"


def _digest_runtime_value(value: Any) -> str:
    if isinstance(value, Path):
        if value.exists() and value.is_file():
            return "sha256:" + hashlib.sha256(value.read_bytes()).hexdigest()
        return digest_value({"path": str(value), "exists": value.exists()})
    if hasattr(value, "to_json"):
        try:
            return digest_value(json.loads(value.to_json(orient="split", date_format="iso")))
        except (TypeError, ValueError, AttributeError):
            pass
    # A DataFrameGroupBy is not directly serializable.  Its selected object and grouping names are
    # sufficient for seed contribution lineage and avoid memory addresses in repr().
    if hasattr(value, "obj") and hasattr(value, "grouper"):
        try:
            return digest_value(
                {
                    "object": json.loads(value.obj.to_json(orient="split", date_format="iso")),
                    "group_names": list(value.grouper.names),
                }
            )
        except (TypeError, ValueError, AttributeError):
            pass
    return digest_value({"type": type(value).__qualname__, "value": str(value)})


def _read_csv(value: Any, bindings: dict[str, Any]) -> Any:
    import pandas as pd

    return pd.read_csv(Path(value), **bindings)


def _read_json(value: Any, bindings: dict[str, Any]) -> Any:
    import pandas as pd

    return pd.read_json(Path(value), **bindings)


def _fillna(value: Any, bindings: dict[str, Any]) -> Any:
    return value.fillna(bindings.get("value", 0))


def _groupby(value: Any, bindings: dict[str, Any]) -> Any:
    return value.groupby(bindings["by"], sort=True)


def _group_sum(value: Any, bindings: dict[str, Any]) -> Any:
    return value[bindings["column"]].sum()


def _reset_index(value: Any, bindings: dict[str, Any]) -> Any:
    return value.reset_index(name=bindings.get("name", value.name or "value"))


def _write_json(value: Any, bindings: dict[str, Any]) -> Any:
    path = Path(bindings["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    value.to_json(path, orient="records", indent=2)
    return path


def _write_csv(value: Any, bindings: dict[str, Any]) -> Any:
    path = Path(bindings["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    value.to_csv(path, index=False)
    return path


def category_adjustment(value: Any, bindings: dict[str, Any]) -> Any:
    """Deliberately task-specific residual used to test residual-code accounting."""

    result = value.copy()
    category = bindings["category"]
    multiplier = float(bindings["multiplier"])
    mask = result[bindings["category_column"]] == category
    result.loc[mask, bindings["value_column"]] = (
        result.loc[mask, bindings["value_column"]] * multiplier
    ).round(2)
    return result


_EXECUTORS: dict[str, Callable[[Any, dict[str, Any]], Any]] = {
    "pandas.read_csv": _read_csv,
    "pandas.read_json": _read_json,
    "pandas.DataFrame.fillna": _fillna,
    "pandas.DataFrame.groupby": _groupby,
    "pandas.core.groupby.DataFrameGroupBy.sum": _group_sum,
    "pandas.Series.reset_index": _reset_index,
    "pandas.DataFrame.to_json": _write_json,
    "pandas.DataFrame.to_csv": _write_csv,
    "residual.category_adjustment": category_adjustment,
}


def _evaluate_output(value: Any, task: ExecutionTask) -> dict[str, Any]:
    if isinstance(value, Path):
        if not value.exists():
            return {"accepted": False, "reason": "expected output file does not exist"}
        if task.expected_rows is None:
            return {"accepted": True, "reason": "artifact exists"}
        actual = json.loads(value.read_text(encoding="utf-8"))
    elif hasattr(value, "to_dict"):
        actual = value.to_dict(orient="records")
        if task.expected_rows is None:
            columns = tuple(str(item) for item in value.columns)
            accepted = (
                (task.expected_columns is None or columns == task.expected_columns)
                and (task.expected_row_count is None or len(value) == task.expected_row_count)
            )
            return {
                "accepted": accepted,
                "reason": "table shape and columns match" if accepted else "table contract differs",
                "actual_columns": columns,
                "actual_row_count": len(value),
                "expected_columns": task.expected_columns,
                "expected_row_count": task.expected_row_count,
            }
    else:
        return {"accepted": task.expected_rows is None, "reason": "runtime artifact was produced"}
    expected = list(task.expected_rows or ())
    accepted = actual == expected
    return {
        "accepted": accepted,
        "reason": "exact JSON rows match" if accepted else "JSON rows differ",
        "actual": actual,
        "expected": expected,
    }


def execute_route(
    task: ExecutionTask,
    steps: Iterable[RouteStep],
    *,
    capabilities: Iterable[VerifiedCapability] | None = None,
) -> dict[str, Any]:
    """Execute only contract-tested, explicitly allowlisted seed operations."""

    catalog = {item.capability_id: item for item in (capabilities or seed_capabilities())}
    steps = tuple(steps)
    valid, reason = validate_route(
        (step.capability_id for step in steps),
        input_artifact_type=task.input_artifact_type,
        required_output_type=task.required_output_type,
        capabilities=catalog.values(),
    )
    if not valid:
        return {
            "schema_version": "1",
            "task_id": task.task_id,
            "status": "rejected_before_execution",
            "accepted": False,
            "reason": reason,
            "nodes": [],
        }

    value: Any = Path(task.input_path)
    input_digest = _digest_runtime_value(value)
    node_receipts: list[dict[str, Any]] = []
    started = datetime.now(UTC)
    residual_lines = 0

    for position, step in enumerate(steps):
        capability = catalog[step.capability_id]
        executor = _EXECUTORS[capability.executor_id]
        node_input_digest = _digest_runtime_value(value)
        before = time.perf_counter()
        value = executor(value, step.bindings)
        elapsed_ms = (time.perf_counter() - before) * 1000
        node_output_digest = _digest_runtime_value(value)
        consumed_by = steps[position + 1].capability_id if position + 1 < len(steps) else "evaluator"
        if capability.capability_id.startswith("residual."):
            # Count the executable statements in the deliberately custom function, not its wrapper
            # record or surrounding reusable route.
            residual_lines += 7
        node_receipts.append(
            {
                "position": position,
                "capability_id": capability.capability_id,
                "package": capability.package_name,
                "version": capability.package_version,
                "input_digest": node_input_digest,
                "output_digest": node_output_digest,
                "consumed_by": consumed_by,
                "contribution_state": "consumed" if consumed_by != "evaluator" else "evaluated",
                "elapsed_ms": round(elapsed_ms, 3),
                "bindings": step.bindings,
            }
        )

    evaluator = _evaluate_output(value, task)
    if evaluator["accepted"] and node_receipts:
        node_receipts[-1]["contribution_state"] = "accepted_output"
    finished = datetime.now(UTC)
    receipt_core = {
        "schema_version": "1",
        "task_id": task.task_id,
        "route": [step.capability_id for step in steps],
        "input_digest": input_digest,
        "final_artifact_digest": _digest_runtime_value(value),
        "environment": {
            "python": platform.python_version(),
            "pandas": version("pandas"),
            "platform": platform.platform(),
        },
    }
    return {
        **receipt_core,
        "run_id": digest_value(receipt_core),
        "status": "accepted" if evaluator["accepted"] else "failed_acceptance",
        "accepted": evaluator["accepted"],
        "evaluator": evaluator,
        "nodes": node_receipts,
        "residual_authored_lines": residual_lines,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "wall_time_ms": round((finished - started).total_seconds() * 1000, 3),
        "model_usage": {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": None},
    }


def receipt_json(receipt: dict[str, Any]) -> str:
    """Render a receipt with stable key ordering for files and comparisons."""

    return json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def route_steps_from_json(items: Iterable[dict[str, Any]]) -> tuple[RouteStep, ...]:
    return tuple(RouteStep(item["capability_id"], dict(item.get("bindings", {}))) for item in items)
