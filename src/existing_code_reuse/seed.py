"""Load the reviewable seed tasks and project verified capabilities into search records."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .capabilities import VerifiedCapability, seed_capabilities
from .execution import ExecutionTask, RouteStep
from .models import DerivedSignal, OperationRecord, digest_value


@dataclass(frozen=True, slots=True)
class SeedExecutionCase:
    track: str
    task: ExecutionTask
    route: tuple[RouteStep, ...]
    candidate_route: tuple[RouteStep, ...]
    expected_decision: str | None


def _signal(
    capability: VerifiedCapability,
    signal_kind: str,
    namespace: str,
    value: str,
    confidence: float,
    evidence_fields: tuple[str, ...],
) -> DerivedSignal:
    identity = {
        "capability_id": capability.capability_id,
        "signal_kind": signal_kind,
        "namespace": namespace,
        "value": value,
        "generator": "verified_seed_projection",
        "generator_version": "1.0.0",
    }
    return DerivedSignal(
        signal_id="sig:" + digest_value(identity).removeprefix("sha256:"),
        operation_id=capability.capability_id,
        signal_kind=signal_kind,  # type: ignore[arg-type]
        namespace=namespace,
        value=value,
        confidence=confidence,
        evidence_fields=evidence_fields,
        generator="verified_seed_projection",
        generator_version="1.0.0",
    )


def project_verified_capabilities_to_search(
    capabilities: tuple[VerifiedCapability, ...] | None = None,
) -> tuple[tuple[OperationRecord, ...], tuple[DerivedSignal, ...]]:
    """Project contract-tested seed records into the generic retrieval interface."""

    capabilities = capabilities or seed_capabilities()
    operations: list[OperationRecord] = []
    signals: list[DerivedSignal] = []
    for capability in capabilities:
        module, _, qualified_leaf = capability.qualified_name.rpartition(".")
        package_id = f"pypi:{capability.package_name}@{capability.package_version}"
        source_digest = digest_value(capability.to_dict())
        contract_text = " ".join(
            [
                capability.purpose,
                *capability.aliases,
                *(f"input {port.artifact_type}" for port in capability.inputs),
                *(f"output {port.artifact_type}" for port in capability.outputs),
                *capability.limitations,
            ]
        )
        operations.append(
            OperationRecord(
                operation_id=capability.capability_id,
                package_id=package_id,
                package_name=capability.package_name,
                package_version=capability.package_version,
                module=module,
                qualified_name=qualified_leaf or capability.qualified_name,
                kind="function",
                signature=(
                    "("
                    + ", ".join(
                        f"{port.name}: {port.artifact_type}" for port in capability.inputs
                    )
                    + ") -> "
                    + ", ".join(port.artifact_type for port in capability.outputs)
                ),
                docstring=contract_text,
                relative_path="verified-seed-contract",
                line_start=1,
                line_end=1,
                source_digest=source_digest,
                visibility="public",
                extraction_method="verified_seed_projection_v1",
                evidence_level="contract_tested",
            )
        )
        for alias in capability.aliases:
            signals.append(
                _signal(capability, "label", "alias", alias, 1.0, ("aliases",))
            )
        signals.append(
            _signal(
                capability,
                "blocking_key",
                "workflow_stage",
                capability.workflow_stage,
                1.0,
                ("workflow_stage",),
            )
        )
        for port in capability.inputs:
            signals.append(
                _signal(
                    capability,
                    "blocking_key",
                    "input_artifact",
                    port.artifact_type,
                    1.0,
                    ("inputs",),
                )
            )
        for port in capability.outputs:
            signals.append(
                _signal(
                    capability,
                    "blocking_key",
                    "output_artifact",
                    port.artifact_type,
                    1.0,
                    ("outputs",),
                )
            )
    return (
        tuple(sorted(operations, key=lambda item: item.operation_id)),
        tuple(sorted(signals, key=lambda item: item.signal_id)),
    )


def load_seed_execution_cases(
    path: str | Path,
    *,
    project_root: str | Path,
    output_root: str | Path | None = None,
) -> tuple[SeedExecutionCase, ...]:
    """Load execution task templates and resolve their local artifact paths."""

    path = Path(path)
    project_root = Path(project_root)
    output_root = Path(output_root) if output_root is not None else project_root
    document = json.loads(path.read_text(encoding="utf-8"))
    cases: list[SeedExecutionCase] = []
    for item in document["tasks"]:
        expected_rows = item.get("expected_rows")
        task = ExecutionTask(
            task_id=item["task_id"],
            prompt=item["prompt"],
            input_artifact_type=item["input_artifact_type"],
            required_output_type=item["required_output_type"],
            required_stages=tuple(item.get("required_stages", ())),
            input_path=str(project_root / item["input_path"]),
            expected_rows=(
                tuple(dict(row) for row in expected_rows) if expected_rows is not None else None
            ),
            expected_columns=(
                tuple(item["expected_columns"]) if "expected_columns" in item else None
            ),
            expected_row_count=item.get("expected_row_count"),
            allow_residual=bool(item.get("allow_residual", False)),
        )

        def route_steps(key: str) -> tuple[RouteStep, ...]:
            result: list[RouteStep] = []
            for raw_step in item.get(key, ()):
                bindings = dict(raw_step.get("bindings", {}))
                if "path" in bindings:
                    bindings["path"] = str(output_root / bindings["path"])
                result.append(RouteStep(raw_step["capability_id"], bindings))
            return tuple(result)

        cases.append(
            SeedExecutionCase(
                track=item["track"],
                task=task,
                route=route_steps("route"),
                candidate_route=route_steps("candidate_route"),
                expected_decision=item.get("expected_decision"),
            )
        )
    return tuple(cases)


def load_jsonl(path: str | Path) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return tuple(records)
