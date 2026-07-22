"""Normalized SQLite persistence and full-text search for static capability catalogs."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any, Iterable

from .models import (
    CatalogBuild,
    DependencyRecord,
    DerivedSignal,
    OperationRecord,
    PackageReleaseRecord,
    RepresentationRecord,
    canonical_json,
    digest_value,
)


SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class FTSResult:
    representation_id: str
    operation_id: str
    plane: str
    text: str
    rank: float


_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE catalog_metadata (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL
) STRICT;

CREATE TABLE packages (
    package_id TEXT PRIMARY KEY,
    ecosystem TEXT NOT NULL,
    name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    version TEXT NOT NULL,
    summary TEXT NOT NULL,
    requires_python TEXT,
    license_expression TEXT,
    classifiers_json TEXT NOT NULL,
    requirements_json TEXT NOT NULL,
    project_urls_json TEXT NOT NULL,
    installed_file_count INTEGER NOT NULL,
    source_kind TEXT NOT NULL,
    source_locator TEXT NOT NULL,
    source_digest TEXT NOT NULL,
    evidence_level TEXT NOT NULL
) STRICT;

CREATE TABLE operations (
    operation_id TEXT PRIMARY KEY,
    package_id TEXT NOT NULL REFERENCES packages(package_id),
    package_name TEXT NOT NULL,
    package_version TEXT NOT NULL,
    module TEXT NOT NULL,
    qualified_name TEXT NOT NULL,
    kind TEXT NOT NULL,
    signature TEXT NOT NULL,
    docstring TEXT NOT NULL,
    relative_path TEXT NOT NULL,
    line_start INTEGER NOT NULL,
    line_end INTEGER NOT NULL,
    source_digest TEXT NOT NULL,
    visibility TEXT NOT NULL,
    extraction_method TEXT NOT NULL,
    evidence_level TEXT NOT NULL
) STRICT;

CREATE TABLE representations (
    representation_id TEXT PRIMARY KEY,
    operation_id TEXT NOT NULL REFERENCES operations(operation_id),
    plane TEXT NOT NULL,
    text TEXT NOT NULL,
    source_fields_json TEXT NOT NULL,
    generator TEXT NOT NULL,
    generator_version TEXT NOT NULL,
    source_digest TEXT NOT NULL,
    review_status TEXT NOT NULL
) STRICT;

CREATE TABLE signals (
    signal_id TEXT PRIMARY KEY,
    operation_id TEXT NOT NULL REFERENCES operations(operation_id),
    signal_kind TEXT NOT NULL,
    namespace TEXT NOT NULL,
    value TEXT NOT NULL,
    confidence REAL NOT NULL,
    evidence_fields_json TEXT NOT NULL,
    generator TEXT NOT NULL,
    generator_version TEXT NOT NULL
) STRICT;

CREATE TABLE dependencies (
    dependency_id TEXT PRIMARY KEY,
    package_id TEXT NOT NULL REFERENCES packages(package_id),
    requirement TEXT NOT NULL,
    target_normalized_name TEXT NOT NULL,
    specifier TEXT NOT NULL,
    marker TEXT NOT NULL,
    extras_json TEXT NOT NULL,
    parse_status TEXT NOT NULL
) STRICT;

CREATE TABLE errors (
    error_id TEXT PRIMARY KEY,
    sort_order INTEGER NOT NULL UNIQUE,
    error_json TEXT NOT NULL
) STRICT;

CREATE INDEX operations_package_idx
    ON operations(package_id, module, qualified_name);
CREATE INDEX representations_operation_idx
    ON representations(operation_id, plane);
CREATE INDEX signals_lookup_idx
    ON signals(signal_kind, namespace, value, operation_id);
CREATE INDEX dependencies_target_idx
    ON dependencies(target_normalized_name, package_id);

CREATE VIRTUAL TABLE representations_fts USING fts5(
    representation_id UNINDEXED,
    operation_id UNINDEXED,
    plane UNINDEXED,
    text,
    tokenize = 'unicode61'
);
"""


def _tuple_json(values: tuple[Any, ...]) -> str:
    return canonical_json(list(values))


def _nested_tuple_json(values: tuple[tuple[Any, ...], ...]) -> str:
    return canonical_json([list(value) for value in values])


def _json_tuple(value: str) -> tuple[Any, ...]:
    decoded = json.loads(value)
    if not isinstance(decoded, list):
        raise ValueError("stored tuple value is not a JSON array")
    return tuple(decoded)


def _json_pair_tuple(value: str) -> tuple[tuple[str, str], ...]:
    decoded = json.loads(value)
    if not isinstance(decoded, list):
        raise ValueError("stored pair tuple is not a JSON array")
    pairs: list[tuple[str, str]] = []
    for item in decoded:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("stored project URL is not a two-item JSON array")
        pairs.append((str(item[0]), str(item[1])))
    return tuple(pairs)


def _connect(path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _create_schema(connection: sqlite3.Connection) -> None:
    try:
        connection.executescript(_SCHEMA_SQL)
    except sqlite3.OperationalError as exc:
        if "fts5" in str(exc).lower():
            raise RuntimeError("this SQLite build does not provide required FTS5 support") from exc
        raise
    connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


def _insert_catalog(
    connection: sqlite3.Connection,
    catalog: CatalogBuild,
    *,
    error_sort_start: int = 0,
) -> int:
    connection.executemany(
        """
        INSERT INTO packages VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (
            (
                item.package_id,
                item.ecosystem,
                item.name,
                item.normalized_name,
                item.version,
                item.summary,
                item.requires_python,
                item.license_expression,
                _tuple_json(item.classifiers),
                _tuple_json(item.requirements),
                _nested_tuple_json(item.project_urls),
                item.installed_file_count,
                item.source_kind,
                item.source_locator,
                item.source_digest,
                item.evidence_level,
            )
            for item in catalog.packages
        ),
    )
    connection.executemany(
        """
        INSERT INTO operations VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (
            (
                item.operation_id,
                item.package_id,
                item.package_name,
                item.package_version,
                item.module,
                item.qualified_name,
                item.kind,
                item.signature,
                item.docstring,
                item.relative_path,
                item.line_start,
                item.line_end,
                item.source_digest,
                item.visibility,
                item.extraction_method,
                item.evidence_level,
            )
            for item in catalog.operations
        ),
    )
    representations = catalog.representations
    connection.executemany(
        """
        INSERT INTO representations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                item.representation_id,
                item.operation_id,
                item.plane,
                item.text,
                _tuple_json(item.source_fields),
                item.generator,
                item.generator_version,
                item.source_digest,
                item.review_status,
            )
            for item in representations
        ),
    )
    connection.executemany(
        """
        INSERT INTO representations_fts(
            representation_id, operation_id, plane, text
        ) VALUES (?, ?, ?, ?)
        """,
        (
            (item.representation_id, item.operation_id, item.plane, item.text)
            for item in representations
        ),
    )
    connection.executemany(
        """
        INSERT INTO signals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                item.signal_id,
                item.operation_id,
                item.signal_kind,
                item.namespace,
                item.value,
                item.confidence,
                _tuple_json(item.evidence_fields),
                item.generator,
                item.generator_version,
            )
            for item in catalog.signals
        ),
    )
    connection.executemany(
        """
        INSERT INTO dependencies VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            (
                item.dependency_id,
                item.package_id,
                item.requirement,
                item.target_normalized_name,
                item.specifier,
                item.marker,
                _tuple_json(item.extras),
                item.parse_status,
            )
            for item in catalog.dependencies
        ),
    )
    ordered_errors = catalog.errors
    connection.executemany(
        "INSERT INTO errors(error_id, sort_order, error_json) VALUES (?, ?, ?)",
        (
            (
                digest_value(
                    {"schema": "catalog-error-v1", "sort_order": index, "error": error}
                ),
                index,
                canonical_json(error),
            )
            for index, error in enumerate(ordered_errors, start=error_sort_start)
        ),
    )
    return error_sort_start + len(ordered_errors)


def _insert_metadata(connection: sqlite3.Connection, counts: dict[str, int]) -> None:
    metadata = {"schema_version": SCHEMA_VERSION, "counts": counts}
    connection.executemany(
        "INSERT INTO catalog_metadata(key, value_json) VALUES (?, ?)",
        [(key, canonical_json(value)) for key, value in sorted(metadata.items())],
    )


def _integrity_check(connection: sqlite3.Connection) -> None:
    integrity = tuple(str(row[0]) for row in connection.execute("PRAGMA integrity_check"))
    if integrity != ("ok",):
        detail = "; ".join(integrity[:10])
        raise RuntimeError(f"refusing to publish malformed catalog: {detail}")


def write_catalog(
    path: str | Path, catalog: CatalogBuild, *, replace: bool = True
) -> Path:
    """Atomically write a normalized SQLite catalog and deterministic FTS projection."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not replace:
        raise FileExistsError(destination)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        # A sqlite3 connection context commits or rolls back but does not close.  Close before
        # replacing the temporary pathname so large FTS databases are fully finalized first.
        with closing(_connect(temporary_path)) as connection:
            with connection:
                _create_schema(connection)
                _insert_catalog(connection, catalog)
                _insert_metadata(connection, catalog.counts())
            _integrity_check(connection)
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def write_catalog_batches(
    path: str | Path,
    catalogs: Iterable[CatalogBuild],
    *,
    replace: bool = True,
) -> Path:
    """Write catalog shards incrementally so memory is bounded by one shard.

    The caller controls shard order. Each shard must use globally stable record identifiers; the
    database constraints reject duplicate or dangling records. The destination is replaced only
    after all shards commit and the complete database passes ``PRAGMA integrity_check``.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not replace:
        raise FileExistsError(destination)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        counts = {
            "packages": 0,
            "operations": 0,
            "representations": 0,
            "signals": 0,
            "dependencies": 0,
            "errors": 0,
        }
        error_sort_start = 0
        with closing(_connect(temporary_path)) as connection:
            with connection:
                _create_schema(connection)
                for catalog in catalogs:
                    error_sort_start = _insert_catalog(
                        connection,
                        catalog,
                        error_sort_start=error_sort_start,
                    )
                    for name, value in catalog.counts().items():
                        counts[name] += value
                _insert_metadata(connection, counts)
            _integrity_check(connection)
        os.replace(temporary_path, destination)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination


def _ensure_schema(connection: sqlite3.Connection) -> None:
    version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported catalog schema version {version}; expected {SCHEMA_VERSION}"
        )


def read_catalog(path: str | Path) -> CatalogBuild:
    """Load a catalog while restoring immutable tuple fields from JSON arrays."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with closing(_connect(source)) as connection:
        _ensure_schema(connection)
        packages = [
            PackageReleaseRecord(
                package_id=row["package_id"],
                ecosystem=row["ecosystem"],
                name=row["name"],
                normalized_name=row["normalized_name"],
                version=row["version"],
                summary=row["summary"],
                requires_python=row["requires_python"],
                license_expression=row["license_expression"],
                classifiers=tuple(str(value) for value in _json_tuple(row["classifiers_json"])),
                requirements=tuple(
                    str(value) for value in _json_tuple(row["requirements_json"])
                ),
                project_urls=_json_pair_tuple(row["project_urls_json"]),
                installed_file_count=row["installed_file_count"],
                source_kind=row["source_kind"],
                source_locator=row["source_locator"],
                source_digest=row["source_digest"],
                evidence_level=row["evidence_level"],
            )
            for row in connection.execute("SELECT * FROM packages ORDER BY package_id")
        ]
        operations = [
            OperationRecord(
                operation_id=row["operation_id"],
                package_id=row["package_id"],
                package_name=row["package_name"],
                package_version=row["package_version"],
                module=row["module"],
                qualified_name=row["qualified_name"],
                kind=row["kind"],
                signature=row["signature"],
                docstring=row["docstring"],
                relative_path=row["relative_path"],
                line_start=row["line_start"],
                line_end=row["line_end"],
                source_digest=row["source_digest"],
                visibility=row["visibility"],
                extraction_method=row["extraction_method"],
                evidence_level=row["evidence_level"],
            )
            for row in connection.execute(
                """
                SELECT * FROM operations
                ORDER BY package_id, module, qualified_name, kind, operation_id
                """
            )
        ]
        representations = [
            RepresentationRecord(
                representation_id=row["representation_id"],
                operation_id=row["operation_id"],
                plane=row["plane"],
                text=row["text"],
                source_fields=tuple(
                    str(value) for value in _json_tuple(row["source_fields_json"])
                ),
                generator=row["generator"],
                generator_version=row["generator_version"],
                source_digest=row["source_digest"],
                review_status=row["review_status"],
            )
            for row in connection.execute(
                """
                SELECT * FROM representations
                ORDER BY operation_id, plane, representation_id
                """
            )
        ]
        signals = [
            DerivedSignal(
                signal_id=row["signal_id"],
                operation_id=row["operation_id"],
                signal_kind=row["signal_kind"],
                namespace=row["namespace"],
                value=row["value"],
                confidence=row["confidence"],
                evidence_fields=tuple(
                    str(value) for value in _json_tuple(row["evidence_fields_json"])
                ),
                generator=row["generator"],
                generator_version=row["generator_version"],
            )
            for row in connection.execute(
                """
                SELECT * FROM signals
                ORDER BY operation_id, signal_kind, namespace, value, signal_id
                """
            )
        ]
        dependencies = [
            DependencyRecord(
                dependency_id=row["dependency_id"],
                package_id=row["package_id"],
                requirement=row["requirement"],
                target_normalized_name=row["target_normalized_name"],
                specifier=row["specifier"],
                marker=row["marker"],
                extras=tuple(str(value) for value in _json_tuple(row["extras_json"])),
                parse_status=row["parse_status"],
            )
            for row in connection.execute(
                "SELECT * FROM dependencies ORDER BY package_id, requirement"
            )
        ]
        errors = [
            json.loads(row["error_json"])
            for row in connection.execute("SELECT error_json FROM errors ORDER BY sort_order")
        ]

    return CatalogBuild(
        packages=packages,
        operations=operations,
        representations=representations,
        signals=signals,
        dependencies=dependencies,
        errors=errors,
    )


def search_fts(
    path: str | Path,
    query: str,
    *,
    limit: int = 20,
    plane: str | None = None,
) -> list[FTSResult]:
    """Search observed representation text using SQLite FTS5/BM25 ordering."""

    if limit <= 0 or not query.strip():
        return []
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    sql = """
        SELECT representation_id, operation_id, plane, text,
               bm25(representations_fts) AS rank
        FROM representations_fts
        WHERE representations_fts MATCH ?
    """
    parameters: list[Any] = [query]
    if plane is not None:
        sql += " AND plane = ?"
        parameters.append(plane)
    sql += " ORDER BY rank, representation_id LIMIT ?"
    parameters.append(limit)

    with closing(_connect(source)) as connection:
        _ensure_schema(connection)
        rows = connection.execute(sql, parameters).fetchall()
    return [
        FTSResult(
            representation_id=row["representation_id"],
            operation_id=row["operation_id"],
            plane=row["plane"],
            text=row["text"],
            rank=float(row["rank"]),
        )
        for row in rows
    ]


# Explicit aliases make call sites readable without multiplying implementations.
write_catalog_sqlite = write_catalog
load_catalog = read_catalog


__all__ = [
    "FTSResult",
    "SCHEMA_VERSION",
    "load_catalog",
    "read_catalog",
    "search_fts",
    "write_catalog",
    "write_catalog_batches",
    "write_catalog_sqlite",
]
