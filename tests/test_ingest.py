from __future__ import annotations

import importlib
from pathlib import Path
import sys
from typing import Any

from existing_code_reuse.ingest import (
    IngestConfig,
    ingest_distribution,
    ingest_installed_distributions,
    parse_dependencies,
)


def test_installed_packaging_is_cataloged_statically(monkeypatch: Any) -> None:
    target_module = "packaging._elffile"
    loaded_before = target_module in sys.modules

    def reject_dynamic_import(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("ingestion must not dynamically import a target module")

    monkeypatch.setattr(importlib, "import_module", reject_dynamic_import)
    catalog = ingest_installed_distributions(["packaging"])

    assert catalog.counts()["packages"] == 1
    package = catalog.packages[0]
    assert package.normalized_name == "packaging"
    assert package.source_locator.startswith("installed:packaging==")
    assert package.source_digest.startswith("sha256:")
    assert package.evidence_level == "static_source"
    assert (target_module in sys.modules) is loaded_before

    operations = {(item.module, item.qualified_name): item for item in catalog.operations}
    version_parse = operations[("packaging.version", "parse")]
    requirement_class = operations[("packaging.requirements", "Requirement")]
    requirement_init = operations[("packaging.requirements", "Requirement.__init__")]
    assert version_parse.kind == "function"
    assert "version: str" in version_parse.signature
    assert "Parse the given version string" in version_parse.docstring
    assert requirement_class.kind == "class"
    assert requirement_init.kind == "method"
    assert requirement_init.visibility == "private"
    assert version_parse.operation_id.startswith("op:")
    assert version_parse.source_digest.startswith("sha256:")

    parse_planes = {
        item.plane
        for item in catalog.representations
        if item.operation_id == version_parse.operation_id
    }
    assert {"identity", "signature", "documentation", "package_summary"} <= parse_planes
    assert all(item.generator_version == "1.0.0" for item in catalog.representations)

    parse_signals = {
        (item.signal_kind, item.namespace, item.value)
        for item in catalog.signals
        if item.operation_id == version_parse.operation_id
    }
    assert ("label", "action", "parse") in parse_signals
    assert ("blocking_key", "action", "parse") in parse_signals
    assert ("label", "artifact", "version") in parse_signals
    assert ("blocking_key", "package", "packaging") in parse_signals


def test_installed_catalog_is_deterministic_and_ordered() -> None:
    config = IngestConfig(max_files_per_distribution=100)
    first = ingest_installed_distributions(["packaging", "packaging"], config=config)
    second = ingest_installed_distributions(["packaging"], config=config)

    assert first == second
    assert first.operations == sorted(
        first.operations,
        key=lambda item: (
            item.package_id,
            item.module,
            item.qualified_name,
            item.kind,
            item.operation_id,
        ),
    )
    assert len({item.operation_id for item in first.operations}) == len(first.operations)
    assert len({item.representation_id for item in first.representations}) == len(
        first.representations
    )
    assert len({item.signal_id for item in first.signals}) == len(first.signals)


def test_distribution_and_byte_limits_are_evidenced() -> None:
    no_files = ingest_installed_distributions(
        ["packaging"], config=IngestConfig(max_files_per_distribution=0)
    )
    assert no_files.packages
    assert not no_files.operations
    assert any(error["error_type"] == "file_count_limit_exceeded" for error in no_files.errors)

    no_bytes = ingest_installed_distributions(
        ["packaging"],
        config=IngestConfig(
            max_files_per_distribution=100,
            max_bytes_per_file=0,
            max_total_bytes_per_distribution=0,
        ),
    )
    assert not no_bytes.operations
    assert any(error["error_type"] == "file_byte_limit_exceeded" for error in no_bytes.errors)


def test_requirements_are_parsed_and_invalid_metadata_is_preserved() -> None:
    records, errors = parse_dependencies(
        "pypi:example@1",
        [
            "Requests[security]>=2; python_version > '3.10'",
            "not a valid requirement ???",
        ],
        package_name="example",
    )

    parsed = next(item for item in records if item.parse_status == "parsed")
    unparsed = next(item for item in records if item.parse_status == "unparsed")
    assert parsed.target_normalized_name == "requests"
    assert parsed.specifier == ">=2"
    assert parsed.extras == ("security",)
    assert "python_version" in parsed.marker
    assert unparsed.requirement == "not a valid requirement ???"
    assert errors[0]["stage"] == "requirement_parse"
    assert errors[0]["requirement"] == unparsed.requirement


class _FixtureMetadata(dict[str, str]):
    def get_all(self, key: str) -> list[str] | None:
        return None


class _FixtureDistribution:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.metadata = _FixtureMetadata(
            Name="fixture-dist",
            Summary="Parse CSV data without executing imported package code.",
        )
        self.version = "1.2.3"
        self.requires = ["packaging>=24", "broken requirement ???"]
        self.files = ["fixture_dist/api.py", "fixture_dist/broken.py"]

    def locate_file(self, item: str) -> Path:
        return self.root / item


def test_synthetic_distribution_preserves_ast_errors_and_extracts_methods(
    tmp_path: Path,
) -> None:
    package_root = tmp_path / "fixture_dist"
    package_root.mkdir()
    (package_root / "api.py").write_text(
        '''"""Fixture module."""

def parse_csv(path: str, /, *, strict: bool = True) -> list[str]:
    """Read and parse a CSV file."""
    return []

async def _load_bytes(path):
    return b""

class Reader:
    """A CSV reader."""

    def read(self, path: str) -> str:
        return path

    async def _probe(self) -> bool:
        return True
''',
        encoding="utf-8",
    )
    (package_root / "broken.py").write_text("def broken(:\n", encoding="utf-8")

    catalog = ingest_distribution(_FixtureDistribution(tmp_path))
    kinds = {(item.qualified_name, item.kind, item.visibility) for item in catalog.operations}
    assert ("parse_csv", "function", "public") in kinds
    assert ("_load_bytes", "async_function", "private") in kinds
    assert ("Reader", "class", "public") in kinds
    assert ("Reader.read", "method", "public") in kinds
    assert ("Reader._probe", "async_method", "private") in kinds
    parse_csv = next(item for item in catalog.operations if item.qualified_name == "parse_csv")
    assert "/" in parse_csv.signature
    assert "*" in parse_csv.signature
    assert any(error["stage"] == "ast_parse" for error in catalog.errors)
    assert any(item.parse_status == "unparsed" for item in catalog.dependencies)


def test_missing_distribution_is_an_explicit_error() -> None:
    catalog = ingest_installed_distributions(
        ["this-distribution-does-not-exist-for-reuse-bench-tests"]
    )
    assert not catalog.packages
    assert catalog.errors[0]["stage"] == "distribution_lookup"
    assert catalog.errors[0]["error_type"] == "PackageNotFoundError"
