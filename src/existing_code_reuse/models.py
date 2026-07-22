"""Canonical records shared by ingestion, indexing, retrieval, and evaluation.

The catalog deliberately separates observed facts from derived retrieval signals.  A docstring,
signature, file digest, or package requirement is evidence.  A label, blocking key, generated
description, or embedding is a versioned derivation and must never silently replace that evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import re
from typing import Any, Literal


EvidenceLevel = Literal[
    "metadata_only",
    "static_source",
    "smoke_tested",
    "contract_tested",
    "accepted_route",
]


def normalize_project_name(name: str) -> str:
    """Apply the PyPA project-name normalization rule used by package indexes."""

    return re.sub(r"[-_.]+", "-", name).lower()


def canonical_json(value: Any) -> str:
    """Serialize a JSON-compatible value deterministically."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest_value(value: Any) -> str:
    """Return a SHA-256 digest for a JSON-compatible value."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PackageReleaseRecord:
    package_id: str
    ecosystem: str
    name: str
    normalized_name: str
    version: str
    summary: str = ""
    requires_python: str | None = None
    license_expression: str | None = None
    classifiers: tuple[str, ...] = ()
    requirements: tuple[str, ...] = ()
    project_urls: tuple[tuple[str, str], ...] = ()
    installed_file_count: int = 0
    source_kind: str = "installed_distribution"
    source_locator: str = ""
    source_digest: str = ""
    evidence_level: EvidenceLevel = "metadata_only"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class OperationRecord:
    operation_id: str
    package_id: str
    package_name: str
    package_version: str
    module: str
    qualified_name: str
    kind: Literal["function", "async_function", "class", "method", "async_method"]
    signature: str
    docstring: str
    relative_path: str
    line_start: int
    line_end: int
    source_digest: str
    visibility: Literal["public", "private"]
    extraction_method: str = "python_ast_v1"
    evidence_level: EvidenceLevel = "static_source"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RepresentationRecord:
    representation_id: str
    operation_id: str
    plane: Literal["identity", "signature", "documentation", "package_summary"]
    text: str
    source_fields: tuple[str, ...]
    generator: str
    generator_version: str
    source_digest: str
    review_status: Literal["observed", "derived", "reviewed"] = "observed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DerivedSignal:
    signal_id: str
    operation_id: str
    signal_kind: Literal["label", "blocking_key"]
    namespace: str
    value: str
    confidence: float
    evidence_fields: tuple[str, ...]
    generator: str
    generator_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DependencyRecord:
    dependency_id: str
    package_id: str
    requirement: str
    target_normalized_name: str
    specifier: str = ""
    marker: str = ""
    extras: tuple[str, ...] = ()
    parse_status: Literal["parsed", "unparsed"] = "parsed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class CatalogBuild:
    packages: list[PackageReleaseRecord] = field(default_factory=list)
    operations: list[OperationRecord] = field(default_factory=list)
    representations: list[RepresentationRecord] = field(default_factory=list)
    signals: list[DerivedSignal] = field(default_factory=list)
    dependencies: list[DependencyRecord] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {
            "packages": len(self.packages),
            "operations": len(self.operations),
            "representations": len(self.representations),
            "signals": len(self.signals),
            "dependencies": len(self.dependencies),
            "errors": len(self.errors),
        }

