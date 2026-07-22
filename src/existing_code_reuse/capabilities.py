"""Verified package-operation contracts used by the executable seed benchmark.

Static extraction can make an operation searchable, but it cannot establish runtime behavior.
Records in this module are deliberately separate: each one has named artifact ports, a pinned
installed package version, an allowlisted executor, and contract-test evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Literal


CompatibilityStatus = Literal[
    "compatible",
    "compatible_with_adapter",
    "requires_probe",
    "incompatible",
    "prohibited",
    "unknown",
]


@dataclass(frozen=True, slots=True)
class ArtifactPort:
    name: str
    artifact_type: str
    required: bool = True


@dataclass(frozen=True, slots=True)
class VerifiedCapability:
    capability_id: str
    package_name: str
    package_version: str
    qualified_name: str
    purpose: str
    aliases: tuple[str, ...]
    inputs: tuple[ArtifactPort, ...]
    outputs: tuple[ArtifactPort, ...]
    workflow_stage: str
    executor_id: str
    limitations: tuple[str, ...] = ()
    evidence_level: str = "contract_tested"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _installed_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError as exc:
        raise RuntimeError(
            f"The executable seed requires the {distribution!r} distribution. "
            "Install the 'execution' optional dependency group."
        ) from exc


def seed_capabilities() -> tuple[VerifiedCapability, ...]:
    """Return the small, real pandas operation catalog used by execution tests.

    Versions are resolved from the environment at runtime and copied into every receipt.  This
    avoids claiming that a contract was tested against a different package release.
    """

    pandas_version = _installed_version("pandas")
    return (
        VerifiedCapability(
            capability_id="pandas.read_csv",
            package_name="pandas",
            package_version=pandas_version,
            qualified_name="pandas.read_csv",
            purpose="Read a comma-separated file into a pandas DataFrame.",
            aliases=("load csv", "read comma separated table", "csv to dataframe"),
            inputs=(ArtifactPort("path", "file/csv"),),
            outputs=(ArtifactPort("table", "table/pandas-dataframe"),),
            workflow_stage="load",
            executor_id="pandas.read_csv",
        ),
        VerifiedCapability(
            capability_id="pandas.read_json",
            package_name="pandas",
            package_version=pandas_version,
            qualified_name="pandas.read_json",
            purpose="Read JSON data into a pandas DataFrame.",
            aliases=("load json", "json to dataframe"),
            inputs=(ArtifactPort("path", "file/json"),),
            outputs=(ArtifactPort("table", "table/pandas-dataframe"),),
            workflow_stage="load",
            executor_id="pandas.read_json",
            limitations=("Does not accept CSV input without a prior conversion.",),
        ),
        VerifiedCapability(
            capability_id="pandas.DataFrame.fillna",
            package_name="pandas",
            package_version=pandas_version,
            qualified_name="pandas.DataFrame.fillna",
            purpose="Replace missing table values using an explicit scalar or column mapping.",
            aliases=("fill missing values", "replace nulls", "impute constant"),
            inputs=(ArtifactPort("table", "table/pandas-dataframe"),),
            outputs=(ArtifactPort("table", "table/pandas-dataframe"),),
            workflow_stage="missing-values",
            executor_id="pandas.DataFrame.fillna",
        ),
        VerifiedCapability(
            capability_id="pandas.DataFrame.groupby",
            package_name="pandas",
            package_version=pandas_version,
            qualified_name="pandas.DataFrame.groupby",
            purpose="Group DataFrame rows by one or more keys.",
            aliases=("group rows", "group by column", "partition table by key"),
            inputs=(ArtifactPort("table", "table/pandas-dataframe"),),
            outputs=(ArtifactPort("groups", "group/pandas-dataframe-groupby"),),
            workflow_stage="group",
            executor_id="pandas.DataFrame.groupby",
        ),
        VerifiedCapability(
            capability_id="pandas.core.groupby.DataFrameGroupBy.sum",
            package_name="pandas",
            package_version=pandas_version,
            qualified_name="pandas.core.groupby.DataFrameGroupBy.sum",
            purpose="Sum a selected numeric column within each DataFrame group.",
            aliases=("grouped sum", "sum by category", "aggregate groups"),
            inputs=(ArtifactPort("groups", "group/pandas-dataframe-groupby"),),
            outputs=(ArtifactPort("series", "series/pandas"),),
            workflow_stage="aggregate",
            executor_id="pandas.core.groupby.DataFrameGroupBy.sum",
        ),
        VerifiedCapability(
            capability_id="pandas.Series.reset_index",
            package_name="pandas",
            package_version=pandas_version,
            qualified_name="pandas.Series.reset_index",
            purpose="Convert an indexed pandas Series into a two-column DataFrame.",
            aliases=("series to dataframe", "move index to column", "flatten grouped result"),
            inputs=(ArtifactPort("series", "series/pandas"),),
            outputs=(ArtifactPort("table", "table/pandas-dataframe"),),
            workflow_stage="adapt",
            executor_id="pandas.Series.reset_index",
        ),
        VerifiedCapability(
            capability_id="pandas.DataFrame.to_json",
            package_name="pandas",
            package_version=pandas_version,
            qualified_name="pandas.DataFrame.to_json",
            purpose="Write a pandas DataFrame as JSON records.",
            aliases=("save json", "dataframe to json", "export json records"),
            inputs=(ArtifactPort("table", "table/pandas-dataframe"),),
            outputs=(ArtifactPort("path", "file/json-records"),),
            workflow_stage="write",
            executor_id="pandas.DataFrame.to_json",
        ),
        VerifiedCapability(
            capability_id="pandas.DataFrame.to_csv",
            package_name="pandas",
            package_version=pandas_version,
            qualified_name="pandas.DataFrame.to_csv",
            purpose="Write a pandas DataFrame as a CSV file.",
            aliases=("save csv", "dataframe to csv", "export comma separated table"),
            inputs=(ArtifactPort("table", "table/pandas-dataframe"),),
            outputs=(ArtifactPort("path", "file/csv"),),
            workflow_stage="write",
            executor_id="pandas.DataFrame.to_csv",
            limitations=("Does not produce a JSON-record artifact.",),
        ),
        VerifiedCapability(
            capability_id="residual.category_adjustment",
            package_name="this-project",
            package_version="0.1.0",
            qualified_name="existing_code_reuse.execution.category_adjustment",
            purpose="Apply the seed task's deliberately custom category adjustment rule.",
            aliases=("custom category rule",),
            inputs=(ArtifactPort("table", "table/pandas-dataframe"),),
            outputs=(ArtifactPort("table", "table/pandas-dataframe"),),
            workflow_stage="residual-custom",
            executor_id="residual.category_adjustment",
            limitations=("This is task-specific residual code, not reusable PyPI supply."),
            evidence_level="contract_tested",
        ),
    )


def check_primary_port_compatibility(
    source: VerifiedCapability,
    target: VerifiedCapability,
) -> CompatibilityStatus:
    """Check the seed catalog's primary dataflow edge without semantic guessing."""

    if not source.outputs or not target.inputs:
        return "unknown"
    return (
        "compatible"
        if source.outputs[0].artifact_type == target.inputs[0].artifact_type
        else "incompatible"
    )

