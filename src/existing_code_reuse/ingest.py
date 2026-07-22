"""Static ingestion of explicitly selected installed Python distributions.

The ingester deliberately uses :mod:`importlib.metadata` plus AST parsing.  It never imports a
module from a distribution being cataloged, calls an extracted object, or evaluates an annotation
or default expression.  Observed source facts and deterministic retrieval derivations remain
separate records.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
from io import BytesIO
from pathlib import Path, PurePosixPath
import re
import tokenize
from typing import Any, Iterable, Sequence

from packaging.requirements import InvalidRequirement, Requirement

from .models import (
    CatalogBuild,
    DependencyRecord,
    DerivedSignal,
    OperationRecord,
    PackageReleaseRecord,
    RepresentationRecord,
    canonical_json,
    digest_value,
    normalize_project_name,
)


EXTRACTION_METHOD = "python_ast_v1"
REPRESENTATION_GENERATOR = "static_ast_representations"
REPRESENTATION_GENERATOR_VERSION = "1.0.0"
SIGNAL_GENERATOR = "deterministic_identity_action_artifact_hints"
SIGNAL_GENERATOR_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class IngestConfig:
    """Resource and visibility limits for one installed distribution."""

    max_files_per_distribution: int = 1_000
    max_bytes_per_file: int = 2_000_000
    max_total_bytes_per_distribution: int = 50_000_000
    include_private: bool = True

    def __post_init__(self) -> None:
        for field_name in (
            "max_files_per_distribution",
            "max_bytes_per_file",
            "max_total_bytes_per_distribution",
        ):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative")


_ACTION_HINTS: dict[str, frozenset[str]] = {
    "aggregate": frozenset({"aggregate", "count", "group", "reduce", "sum"}),
    "compare": frozenset({"compare", "diff", "equal", "match"}),
    "convert": frozenset({"cast", "convert", "decode", "encode", "serialize"}),
    "create": frozenset({"build", "construct", "create", "generate", "make"}),
    "load": frozenset({"fetch", "load", "open", "read"}),
    "normalize": frozenset({"canonicalize", "clean", "normalize", "standardize"}),
    "parse": frozenset({"parse", "scan", "tokenize"}),
    "predict": frozenset({"classify", "infer", "predict", "score"}),
    "retrieve": frozenset({"find", "get", "lookup", "query", "search"}),
    "train": frozenset({"fit", "learn", "train"}),
    "transform": frozenset({"apply", "map", "process", "transform"}),
    "validate": frozenset({"check", "ensure", "validate", "verify"}),
    "write": frozenset({"dump", "export", "save", "write"}),
}

_ARTIFACT_HINTS: dict[str, frozenset[str]] = {
    "array": frozenset({"array", "matrix", "ndarray", "tensor", "vector"}),
    "binary": frozenset({"binary", "bytes", "bytearray"}),
    "csv": frozenset({"csv"}),
    "dataframe": frozenset({"dataframe", "frame", "series", "table", "tabular"}),
    "document": frozenset({"document", "markdown", "pdf"}),
    "email": frozenset({"email", "eml", "mime"}),
    "file": frozenset({"directory", "file", "folder", "path"}),
    "html": frozenset({"html"}),
    "image": frozenset({"bitmap", "image", "jpeg", "jpg", "png"}),
    "json": frozenset({"json", "jsonl"}),
    "model": frozenset({"classifier", "estimator", "model"}),
    "parquet": frozenset({"arrow", "parquet"}),
    "requirement": frozenset({"dependency", "requirement", "specifier"}),
    "schema": frozenset({"contract", "schema"}),
    "text": frozenset({"string", "text", "unicode"}),
    "version": frozenset({"release", "version"}),
    "xml": frozenset({"xml"}),
    "yaml": frozenset({"yaml", "yml"}),
}


def _stable_id(prefix: str, value: Any) -> str:
    return f"{prefix}:{digest_value(value).removeprefix('sha256:')}"


def _source_digest(source: str | bytes) -> str:
    data = source.encode("utf-8") if isinstance(source, str) else source
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _error(
    *,
    package_name: str,
    stage: str,
    error_type: str,
    message: str,
    package_id: str = "",
    path: str = "",
    **details: Any,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "package_name": package_name,
        "package_id": package_id,
        "stage": stage,
        "path": path,
        "error_type": error_type,
        "message": message,
    }
    record.update(details)
    return record


def parse_dependencies(
    package_id: str,
    raw_requirements: Iterable[str],
    *,
    package_name: str = "",
) -> tuple[list[DependencyRecord], list[dict[str, Any]]]:
    """Parse ``Requires-Dist`` values without dropping invalid source metadata."""

    dependencies: list[DependencyRecord] = []
    errors: list[dict[str, Any]] = []
    for raw in sorted({str(item).strip() for item in raw_requirements if str(item).strip()}):
        dependency_id = _stable_id(
            "dep", {"schema": "dependency-id-v1", "package_id": package_id, "requirement": raw}
        )
        try:
            requirement = Requirement(raw)
        except InvalidRequirement as exc:
            dependencies.append(
                DependencyRecord(
                    dependency_id=dependency_id,
                    package_id=package_id,
                    requirement=raw,
                    target_normalized_name="",
                    parse_status="unparsed",
                )
            )
            errors.append(
                _error(
                    package_name=package_name,
                    package_id=package_id,
                    stage="requirement_parse",
                    error_type=type(exc).__name__,
                    message=str(exc),
                    requirement=raw,
                )
            )
            continue

        dependencies.append(
            DependencyRecord(
                dependency_id=dependency_id,
                package_id=package_id,
                requirement=raw,
                target_normalized_name=normalize_project_name(requirement.name),
                specifier=str(requirement.specifier),
                marker=str(requirement.marker) if requirement.marker is not None else "",
                extras=tuple(sorted(requirement.extras)),
            )
        )

    return dependencies, errors


def _metadata_values(package_metadata: Any, key: str) -> tuple[str, ...]:
    values = package_metadata.get_all(key) if hasattr(package_metadata, "get_all") else None
    return tuple(str(value).strip() for value in (values or ()) if str(value).strip())


def _project_urls(package_metadata: Any) -> tuple[tuple[str, str], ...]:
    parsed: list[tuple[str, str]] = []
    for raw in _metadata_values(package_metadata, "Project-URL"):
        if "," in raw:
            label, url = raw.split(",", 1)
            parsed.append((label.strip(), url.strip()))
        else:
            parsed.append(("", raw))
    return tuple(sorted(set(parsed)))


def _file_inventory(files: Sequence[Any]) -> tuple[tuple[Any, ...], ...]:
    inventory: list[tuple[Any, ...]] = []
    for item in files:
        file_hash = getattr(item, "hash", None)
        hash_value = ""
        if file_hash is not None:
            mode = getattr(file_hash, "mode", "")
            value = getattr(file_hash, "value", "")
            hash_value = f"{mode}:{value}" if mode else str(value)
        inventory.append((str(item), hash_value, getattr(item, "size", None)))
    return tuple(sorted(inventory, key=lambda row: row[0]))


def _module_from_relative_path(relative_path: str) -> str:
    path = PurePosixPath(relative_path)
    parts = list(path.parts)
    if not parts:
        return ""
    parts[-1] = parts[-1][:-3]
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(part for part in parts if part)


def _unparse(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - defensive for future AST node types
        return "?"


def _format_arg(argument: ast.arg) -> str:
    annotation = f": {_unparse(argument.annotation)}" if argument.annotation is not None else ""
    return f"{argument.arg}{annotation}"


def _format_type_params(node: ast.AST) -> str:
    type_params = getattr(node, "type_params", ())
    if not type_params:
        return ""
    return "[" + ", ".join(_unparse(item) for item in type_params) + "]"


def _function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    arguments = node.args
    positional = [*arguments.posonlyargs, *arguments.args]
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(arguments.defaults))
    defaults.extend(arguments.defaults)

    rendered: list[str] = []
    for index, (argument, default) in enumerate(zip(positional, defaults, strict=True)):
        value = _format_arg(argument)
        if default is not None:
            value += f" = {_unparse(default)}"
        rendered.append(value)
        if arguments.posonlyargs and index + 1 == len(arguments.posonlyargs):
            rendered.append("/")

    if arguments.vararg is not None:
        rendered.append("*" + _format_arg(arguments.vararg))
    elif arguments.kwonlyargs:
        rendered.append("*")

    for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True):
        value = _format_arg(argument)
        if default is not None:
            value += f" = {_unparse(default)}"
        rendered.append(value)

    if arguments.kwarg is not None:
        rendered.append("**" + _format_arg(arguments.kwarg))

    returns = f" -> {_unparse(node.returns)}" if node.returns is not None else ""
    return f"{_format_type_params(node)}({', '.join(rendered)}){returns}"


def _class_signature(node: ast.ClassDef) -> str:
    bases = [_unparse(base) for base in node.bases]
    for keyword in node.keywords:
        if keyword.arg is None:
            bases.append("**" + _unparse(keyword.value))
        else:
            bases.append(f"{keyword.arg}={_unparse(keyword.value)}")
    return f"{_format_type_params(node)}({', '.join(bases)})"


def _visibility(name: str, parent_name: str = "") -> str:
    return "private" if name.startswith("_") or parent_name.startswith("_") else "public"


def _operation(
    *,
    node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef,
    package_id: str,
    package_name: str,
    package_version: str,
    module: str,
    relative_path: str,
    source: str,
    parent_name: str = "",
) -> OperationRecord:
    if isinstance(node, ast.ClassDef):
        kind = "class"
        signature = _class_signature(node)
    elif isinstance(node, ast.AsyncFunctionDef):
        kind = "async_method" if parent_name else "async_function"
        signature = _function_signature(node)
    else:
        kind = "method" if parent_name else "function"
        signature = _function_signature(node)

    qualified_name = f"{parent_name}.{node.name}" if parent_name else node.name
    source_segment = ast.get_source_segment(source, node) or _unparse(node)
    source_digest = _source_digest(source_segment)
    operation_id = _stable_id(
        "op",
        {
            "schema": "operation-id-v1",
            "package_id": package_id,
            "module": module,
            "qualified_name": qualified_name,
            "kind": kind,
            "signature": signature,
            "source_digest": source_digest,
        },
    )
    return OperationRecord(
        operation_id=operation_id,
        package_id=package_id,
        package_name=package_name,
        package_version=package_version,
        module=module,
        qualified_name=qualified_name,
        kind=kind,  # type: ignore[arg-type]
        signature=signature,
        docstring=ast.get_docstring(node, clean=True) or "",
        relative_path=relative_path,
        line_start=node.lineno,
        line_end=getattr(node, "end_lineno", node.lineno) or node.lineno,
        source_digest=source_digest,
        visibility=_visibility(node.name, parent_name),  # type: ignore[arg-type]
        extraction_method=EXTRACTION_METHOD,
    )


def _representation(
    operation: OperationRecord,
    *,
    plane: str,
    text: str,
    source_fields: tuple[str, ...],
    review_status: str = "observed",
) -> RepresentationRecord:
    source_digest = digest_value(
        {
            "operation_source_digest": operation.source_digest,
            "source_fields": source_fields,
            "text": text,
        }
    )
    representation_id = _stable_id(
        "rep",
        {
            "schema": "representation-id-v1",
            "operation_id": operation.operation_id,
            "plane": plane,
            "text": text,
            "source_digest": source_digest,
            "generator": REPRESENTATION_GENERATOR,
            "generator_version": REPRESENTATION_GENERATOR_VERSION,
        },
    )
    return RepresentationRecord(
        representation_id=representation_id,
        operation_id=operation.operation_id,
        plane=plane,  # type: ignore[arg-type]
        text=text,
        source_fields=source_fields,
        generator=REPRESENTATION_GENERATOR,
        generator_version=REPRESENTATION_GENERATOR_VERSION,
        source_digest=source_digest,
        review_status=review_status,  # type: ignore[arg-type]
    )


def derive_representations(
    operation: OperationRecord, package_summary: str
) -> list[RepresentationRecord]:
    """Create independently addressable representation planes from observed facts."""

    identity = f"{operation.package_name} {operation.module}.{operation.qualified_name} {operation.kind}"
    records = [
        _representation(
            operation,
            plane="identity",
            text=identity,
            source_fields=("package_name", "module", "qualified_name", "kind"),
        ),
        _representation(
            operation,
            plane="signature",
            text=f"{operation.module}.{operation.qualified_name}{operation.signature}",
            source_fields=("module", "qualified_name", "signature"),
        ),
    ]
    if operation.docstring:
        records.append(
            _representation(
                operation,
                plane="documentation",
                text=operation.docstring,
                source_fields=("docstring",),
            )
        )
    if package_summary:
        records.append(
            _representation(
                operation,
                plane="package_summary",
                text=package_summary,
                source_fields=("package_summary",),
            )
        )
    return records


def _tokens(value: str) -> frozenset[str]:
    de_camel = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    return frozenset(re.findall(r"[a-z0-9]+", de_camel.lower().replace("_", " ")))


def _signal(
    operation: OperationRecord,
    *,
    signal_kind: str,
    namespace: str,
    value: str,
    confidence: float,
    evidence_fields: tuple[str, ...],
) -> DerivedSignal:
    signal_id = _stable_id(
        "sig",
        {
            "schema": "derived-signal-id-v1",
            "operation_id": operation.operation_id,
            "signal_kind": signal_kind,
            "namespace": namespace,
            "value": value,
            "generator": SIGNAL_GENERATOR,
            "generator_version": SIGNAL_GENERATOR_VERSION,
        },
    )
    return DerivedSignal(
        signal_id=signal_id,
        operation_id=operation.operation_id,
        signal_kind=signal_kind,  # type: ignore[arg-type]
        namespace=namespace,
        value=value,
        confidence=confidence,
        evidence_fields=evidence_fields,
        generator=SIGNAL_GENERATOR,
        generator_version=SIGNAL_GENERATOR_VERSION,
    )


def derive_signals(operation: OperationRecord) -> list[DerivedSignal]:
    """Derive deterministic identity, action, and artifact signals with field evidence."""

    package = normalize_project_name(operation.package_name)
    operation_name = operation.qualified_name.rsplit(".", 1)[-1]
    signals = [
        _signal(
            operation,
            signal_kind="label",
            namespace="identity.kind",
            value=operation.kind,
            confidence=1.0,
            evidence_fields=("kind",),
        ),
        _signal(
            operation,
            signal_kind="label",
            namespace="identity.visibility",
            value=operation.visibility,
            confidence=1.0,
            evidence_fields=("visibility",),
        ),
        _signal(
            operation,
            signal_kind="blocking_key",
            namespace="ecosystem",
            value="python",
            confidence=1.0,
            evidence_fields=("module",),
        ),
        _signal(
            operation,
            signal_kind="blocking_key",
            namespace="package",
            value=package,
            confidence=1.0,
            evidence_fields=("package_name",),
        ),
        _signal(
            operation,
            signal_kind="blocking_key",
            namespace="module",
            value=operation.module,
            confidence=1.0,
            evidence_fields=("module",),
        ),
        _signal(
            operation,
            signal_kind="blocking_key",
            namespace="operation_name",
            value=operation_name.lower(),
            confidence=1.0,
            evidence_fields=("qualified_name",),
        ),
    ]

    evidence_text = {
        "qualified_name": operation.qualified_name,
        "module": operation.module,
        "docstring": operation.docstring,
    }
    evidence_tokens = {field: _tokens(text) for field, text in evidence_text.items()}
    identity_fields = ("qualified_name", "module")

    def add_hints(namespace: str, hints: dict[str, frozenset[str]]) -> None:
        for value, aliases in sorted(hints.items()):
            matched_fields = tuple(
                field for field in evidence_text if aliases.intersection(evidence_tokens[field])
            )
            if not matched_fields:
                continue
            confidence = 0.95 if any(field in identity_fields for field in matched_fields) else 0.7
            signals.append(
                _signal(
                    operation,
                    signal_kind="label",
                    namespace=namespace,
                    value=value,
                    confidence=confidence,
                    evidence_fields=matched_fields,
                )
            )
            signals.append(
                _signal(
                    operation,
                    signal_kind="blocking_key",
                    namespace=namespace,
                    value=value,
                    confidence=confidence,
                    evidence_fields=matched_fields,
                )
            )

    add_hints("action", _ACTION_HINTS)
    add_hints("artifact", _ARTIFACT_HINTS)

    for token in sorted(_tokens(operation_name)):
        signals.append(
            _signal(
                operation,
                signal_kind="blocking_key",
                namespace="name_token",
                value=token,
                confidence=0.9,
                evidence_fields=("qualified_name",),
            )
        )

    unique = {signal.signal_id: signal for signal in signals}
    return sorted(
        unique.values(),
        key=lambda item: (item.signal_kind, item.namespace, item.value, item.signal_id),
    )


def _decode_source(data: bytes) -> str:
    encoding, _ = tokenize.detect_encoding(BytesIO(data).readline)
    return data.decode(encoding)


def _operations_from_source(
    *,
    source: str,
    relative_path: str,
    package_id: str,
    package_name: str,
    package_version: str,
    include_private: bool,
) -> list[OperationRecord]:
    tree = ast.parse(source, filename=relative_path, type_comments=True)
    module = _module_from_relative_path(relative_path)
    operations: list[OperationRecord] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if include_private or not node.name.startswith("_"):
            operations.append(
                _operation(
                    node=node,
                    package_id=package_id,
                    package_name=package_name,
                    package_version=package_version,
                    module=module,
                    relative_path=relative_path,
                    source=source,
                )
            )
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if include_private or (
                    not node.name.startswith("_") and not child.name.startswith("_")
                ):
                    operations.append(
                        _operation(
                            node=child,
                            package_id=package_id,
                            package_name=package_name,
                            package_version=package_version,
                            module=module,
                            relative_path=relative_path,
                            source=source,
                            parent_name=node.name,
                        )
                    )
    return operations


def _partition_python_files(
    files: Sequence[Any],
) -> tuple[list[tuple[str, Any]], list[str]]:
    candidates: list[tuple[str, Any]] = []
    rejected: list[str] = []
    for item in files:
        relative = str(item).replace("\\", "/")
        path = PurePosixPath(relative)
        if not relative.endswith(".py"):
            continue
        is_windows_absolute = bool(re.match(r"^[A-Za-z]:/", relative))
        if path.is_absolute() or is_windows_absolute or ".." in path.parts:
            rejected.append(relative)
        else:
            candidates.append((relative, item))
    return sorted(candidates, key=lambda pair: pair[0]), sorted(rejected)


def ingest_distribution(
    distribution: Any,
    *,
    requested_name: str = "",
    config: IngestConfig | None = None,
) -> CatalogBuild:
    """Ingest one distribution object returned by :mod:`importlib.metadata`."""

    config = config or IngestConfig()
    build = CatalogBuild()
    package_metadata = distribution.metadata
    name = str(package_metadata.get("Name") or requested_name).strip()
    version = str(distribution.version)
    normalized_name = normalize_project_name(name)
    package_id = f"pypi:{normalized_name}@{version}"
    summary = str(package_metadata.get("Summary") or "").strip()
    requires_python = str(package_metadata.get("Requires-Python") or "").strip() or None
    license_expression = (
        str(
            package_metadata.get("License-Expression")
            or package_metadata.get("License")
            or ""
        ).strip()
        or None
    )
    classifiers = tuple(sorted(set(_metadata_values(package_metadata, "Classifier"))))
    project_urls = _project_urls(package_metadata)

    try:
        raw_requirements = tuple(distribution.requires or ())
    except Exception as exc:  # metadata providers may be third-party implementations
        raw_requirements = _metadata_values(package_metadata, "Requires-Dist")
        build.errors.append(
            _error(
                package_name=name,
                package_id=package_id,
                stage="metadata_read",
                error_type=type(exc).__name__,
                message=str(exc),
                field="Requires-Dist",
            )
        )
    requirements = tuple(sorted(set(str(item).strip() for item in raw_requirements if item)))
    dependencies, dependency_errors = parse_dependencies(
        package_id, requirements, package_name=name
    )
    build.dependencies.extend(dependencies)
    build.errors.extend(dependency_errors)

    try:
        files = tuple(distribution.files or ())
    except Exception as exc:
        files = ()
        build.errors.append(
            _error(
                package_name=name,
                package_id=package_id,
                stage="metadata_read",
                error_type=type(exc).__name__,
                message=str(exc),
                field="files",
            )
        )

    candidates, unsafe_paths = _partition_python_files(files)
    for unsafe_path in unsafe_paths:
        build.errors.append(
            _error(
                package_name=name,
                package_id=package_id,
                stage="source_path",
                path=unsafe_path,
                error_type="unsafe_relative_path",
                message="distribution file path is absolute or contains parent traversal",
            )
        )
    selected = candidates[: config.max_files_per_distribution]
    if len(candidates) > len(selected):
        build.errors.append(
            _error(
                package_name=name,
                package_id=package_id,
                stage="source_limit",
                error_type="file_count_limit_exceeded",
                message=(
                    f"selected {len(selected)} of {len(candidates)} Python files; "
                    f"limit={config.max_files_per_distribution}"
                ),
                selected_file_count=len(selected),
                candidate_file_count=len(candidates),
                limit=config.max_files_per_distribution,
            )
        )

    total_bytes = 0
    parsed_file_count = 0
    for relative_path, file_entry in selected:
        try:
            located = Path(distribution.locate_file(file_entry))
            size = located.stat().st_size
        except (OSError, TypeError, ValueError) as exc:
            build.errors.append(
                _error(
                    package_name=name,
                    package_id=package_id,
                    stage="source_read",
                    path=relative_path,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )
            continue

        if size > config.max_bytes_per_file:
            build.errors.append(
                _error(
                    package_name=name,
                    package_id=package_id,
                    stage="source_limit",
                    path=relative_path,
                    error_type="file_byte_limit_exceeded",
                    message=f"file size {size} exceeds limit {config.max_bytes_per_file}",
                    observed_bytes=size,
                    limit=config.max_bytes_per_file,
                )
            )
            continue
        if total_bytes + size > config.max_total_bytes_per_distribution:
            build.errors.append(
                _error(
                    package_name=name,
                    package_id=package_id,
                    stage="source_limit",
                    path=relative_path,
                    error_type="distribution_byte_limit_exceeded",
                    message=(
                        f"reading {size} bytes would exceed distribution limit "
                        f"{config.max_total_bytes_per_distribution}"
                    ),
                    observed_total_bytes=total_bytes,
                    next_file_bytes=size,
                    limit=config.max_total_bytes_per_distribution,
                )
            )
            continue

        try:
            with located.open("rb") as source_file:
                data = source_file.read(config.max_bytes_per_file + 1)
        except OSError as exc:
            build.errors.append(
                _error(
                    package_name=name,
                    package_id=package_id,
                    stage="source_read",
                    path=relative_path,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )
            continue
        if len(data) > config.max_bytes_per_file:
            build.errors.append(
                _error(
                    package_name=name,
                    package_id=package_id,
                    stage="source_limit",
                    path=relative_path,
                    error_type="file_byte_limit_exceeded",
                    message=(
                        f"read exceeded limit {config.max_bytes_per_file}; "
                        "file may have changed after stat"
                    ),
                    observed_bytes=len(data),
                    limit=config.max_bytes_per_file,
                )
            )
            continue
        total_bytes += len(data)

        try:
            source = _decode_source(data)
        except (SyntaxError, UnicodeError) as exc:
            build.errors.append(
                _error(
                    package_name=name,
                    package_id=package_id,
                    stage="source_decode",
                    path=relative_path,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )
            continue

        try:
            operations = _operations_from_source(
                source=source,
                relative_path=relative_path,
                package_id=package_id,
                package_name=name,
                package_version=version,
                include_private=config.include_private,
            )
        except SyntaxError as exc:
            build.errors.append(
                _error(
                    package_name=name,
                    package_id=package_id,
                    stage="ast_parse",
                    path=relative_path,
                    error_type=type(exc).__name__,
                    message=exc.msg,
                    line=exc.lineno,
                    offset=exc.offset,
                )
            )
            continue

        parsed_file_count += 1
        build.operations.extend(operations)
        for operation in operations:
            build.representations.extend(derive_representations(operation, summary))
            build.signals.extend(derive_signals(operation))

    package_source_digest = digest_value(
        {
            "schema": "installed-distribution-source-v1",
            "name": name,
            "version": version,
            "summary": summary,
            "requires_python": requires_python,
            "license_expression": license_expression,
            "classifiers": classifiers,
            "requirements": requirements,
            "project_urls": project_urls,
            "file_inventory": _file_inventory(files),
        }
    )
    build.packages.append(
        PackageReleaseRecord(
            package_id=package_id,
            ecosystem="pypi",
            name=name,
            normalized_name=normalized_name,
            version=version,
            summary=summary,
            requires_python=requires_python,
            license_expression=license_expression,
            classifiers=classifiers,
            requirements=requirements,
            project_urls=project_urls,
            installed_file_count=len(files),
            source_locator=f"installed:{normalized_name}=={version}",
            source_digest=package_source_digest,
            evidence_level="static_source" if parsed_file_count else "metadata_only",
        )
    )
    _sort_build(build)
    return build


def _sort_build(build: CatalogBuild) -> None:
    build.packages.sort(key=lambda item: item.package_id)
    build.operations.sort(
        key=lambda item: (
            item.package_id,
            item.module,
            item.qualified_name,
            item.kind,
            item.operation_id,
        )
    )
    build.representations.sort(
        key=lambda item: (item.operation_id, item.plane, item.representation_id)
    )
    build.signals.sort(
        key=lambda item: (
            item.operation_id,
            item.signal_kind,
            item.namespace,
            item.value,
            item.signal_id,
        )
    )
    build.dependencies.sort(key=lambda item: (item.package_id, item.requirement))
    build.errors.sort(key=canonical_json)


def ingest_installed_distributions(
    distribution_names: Sequence[str], *, config: IngestConfig | None = None
) -> CatalogBuild:
    """Catalog explicitly named installed distributions without importing their packages."""

    config = config or IngestConfig()
    combined = CatalogBuild()
    seen_package_ids: set[str] = set()
    requested_names = sorted(
        {name.strip() for name in distribution_names if name.strip()}, key=normalize_project_name
    )
    for requested_name in requested_names:
        try:
            distribution = importlib_metadata.distribution(requested_name)
        except importlib_metadata.PackageNotFoundError as exc:
            combined.errors.append(
                _error(
                    package_name=requested_name,
                    stage="distribution_lookup",
                    error_type=type(exc).__name__,
                    message=f"installed distribution not found: {requested_name}",
                )
            )
            continue
        except Exception as exc:
            combined.errors.append(
                _error(
                    package_name=requested_name,
                    stage="distribution_lookup",
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )
            continue

        partial = ingest_distribution(
            distribution, requested_name=requested_name, config=config
        )
        if partial.packages and partial.packages[0].package_id in seen_package_ids:
            continue
        if partial.packages:
            seen_package_ids.add(partial.packages[0].package_id)
        combined.packages.extend(partial.packages)
        combined.operations.extend(partial.operations)
        combined.representations.extend(partial.representations)
        combined.signals.extend(partial.signals)
        combined.dependencies.extend(partial.dependencies)
        combined.errors.extend(partial.errors)

    _sort_build(combined)
    return combined


# A descriptive alias for callers that treat ingestion as a complete catalog build.
build_installed_catalog = ingest_installed_distributions


__all__ = [
    "IngestConfig",
    "build_installed_catalog",
    "derive_representations",
    "derive_signals",
    "ingest_distribution",
    "ingest_installed_distributions",
    "parse_dependencies",
]
