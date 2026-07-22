from __future__ import annotations

from pathlib import Path
import sqlite3

import pytest

from existing_code_reuse.ingest import IngestConfig, ingest_installed_distributions, parse_dependencies
from existing_code_reuse.models import CatalogBuild
from existing_code_reuse.storage import (
    read_catalog,
    search_fts,
    write_catalog,
    write_catalog_batches,
)


@pytest.fixture()
def packaging_catalog():
    catalog = ingest_installed_distributions(
        ["packaging"], config=IngestConfig(max_files_per_distribution=100)
    )
    dependencies, errors = parse_dependencies(
        catalog.packages[0].package_id,
        ["typing-extensions[tests]>=4; python_version < '3.13'"],
        package_name="packaging",
    )
    catalog.dependencies.extend(dependencies)
    catalog.errors.extend(errors)
    return catalog


def test_sqlite_round_trip_restores_tuple_fields(
    tmp_path: Path, packaging_catalog
) -> None:
    database = write_catalog(tmp_path / "catalog.sqlite", packaging_catalog)
    restored = read_catalog(database)

    assert restored == packaging_catalog
    package = restored.packages[0]
    assert isinstance(package.classifiers, tuple)
    assert isinstance(package.requirements, tuple)
    assert isinstance(package.project_urls, tuple)
    assert all(isinstance(item, tuple) for item in package.project_urls)
    assert isinstance(restored.representations[0].source_fields, tuple)
    assert isinstance(restored.signals[0].evidence_fields, tuple)
    assert restored.dependencies[0].extras == ("tests",)


def test_sqlite_has_normalized_tables_and_fts_search(
    tmp_path: Path, packaging_catalog
) -> None:
    database = write_catalog(tmp_path / "catalog.sqlite", packaging_catalog)
    with sqlite3.connect(database) as connection:
        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            )
        }
        operation_count = connection.execute("SELECT count(*) FROM operations").fetchone()[0]
        representation_count = connection.execute(
            "SELECT count(*) FROM representations"
        ).fetchone()[0]

    assert {
        "packages",
        "operations",
        "representations",
        "signals",
        "dependencies",
        "errors",
        "representations_fts",
    } <= table_names
    assert operation_count == len(packaging_catalog.operations)
    assert representation_count == len(packaging_catalog.representations)

    results = search_fts(
        database,
        '"Parse the given version string"',
        plane="documentation",
        limit=10,
    )
    parse_operation = next(
        item
        for item in packaging_catalog.operations
        if item.module == "packaging.version" and item.qualified_name == "parse"
    )
    assert results
    assert results[0].operation_id == parse_operation.operation_id
    assert results[0].plane == "documentation"
    assert "Parse the given version string" in results[0].text


def test_write_is_repeatable_and_replace_can_be_refused(
    tmp_path: Path, packaging_catalog
) -> None:
    first_path = write_catalog(tmp_path / "first.sqlite", packaging_catalog)
    second_path = write_catalog(tmp_path / "second.sqlite", packaging_catalog)

    assert read_catalog(first_path) == read_catalog(second_path)
    with pytest.raises(FileExistsError):
        write_catalog(first_path, packaging_catalog, replace=False)


def test_empty_fts_query_and_nonpositive_limit_return_no_results(
    tmp_path: Path, packaging_catalog
) -> None:
    database = write_catalog(tmp_path / "catalog.sqlite", packaging_catalog)
    assert search_fts(database, "") == []
    assert search_fts(database, "version", limit=0) == []


def test_batched_write_bounds_memory_to_one_catalog_shard(
    tmp_path: Path, packaging_catalog
) -> None:
    midpoint = len(packaging_catalog.operations) // 2
    first_operation_ids = {
        item.operation_id for item in packaging_catalog.operations[:midpoint]
    }
    first = CatalogBuild(
        packages=list(packaging_catalog.packages),
        operations=list(packaging_catalog.operations[:midpoint]),
        representations=[
            item
            for item in packaging_catalog.representations
            if item.operation_id in first_operation_ids
        ],
        signals=[
            item for item in packaging_catalog.signals if item.operation_id in first_operation_ids
        ],
        dependencies=list(packaging_catalog.dependencies),
        errors=list(packaging_catalog.errors[:1]),
    )
    second = CatalogBuild(
        operations=list(packaging_catalog.operations[midpoint:]),
        representations=[
            item
            for item in packaging_catalog.representations
            if item.operation_id not in first_operation_ids
        ],
        signals=[
            item for item in packaging_catalog.signals if item.operation_id not in first_operation_ids
        ],
        errors=list(packaging_catalog.errors[1:]),
    )

    database = write_catalog_batches(tmp_path / "batched.sqlite", (first, second))

    assert read_catalog(database) == packaging_catalog
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
