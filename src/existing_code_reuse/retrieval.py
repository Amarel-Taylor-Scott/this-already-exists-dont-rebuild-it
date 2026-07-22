"""Deterministic and lightweight retrieval over extracted Python operations.

The catalog records are deliberately independent from the retrieval configuration.  A retrieval
experiment is described by data records below, so changing an analyzer, field set, blocking rule,
or abstention threshold does not require adding another hard-coded "profile" branch.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, fields
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Iterable, Iterator, Literal, Mapping, Sequence

from .models import DerivedSignal, OperationRecord


RetrievalMethod = Literal["exact", "word_tfidf", "char_tfidf", "sqlite_fts5"]
OperationField = Literal[
    "identity",
    "package_name",
    "module",
    "qualified_name",
    "signature",
    "docstring",
]


class RetrievalConfigurationError(ValueError):
    """Raised when a retrieval experiment has an invalid configuration."""


class OptionalRetrievalDependencyError(ImportError):
    """Raised when an explicitly requested optional retrieval dependency is unavailable."""


def _validate_ngram_range(value: tuple[int, int], label: str) -> None:
    if len(value) != 2 or value[0] <= 0 or value[1] < value[0]:
        raise RetrievalConfigurationError(f"invalid {label} n-gram range: {value!r}")


@dataclass(frozen=True, slots=True)
class QuerySignal:
    """A deterministic query facet used to block the catalog before ranking."""

    namespace: str
    value: str
    signal_kind: Literal["label", "blocking_key"] = "blocking_key"


@dataclass(frozen=True, slots=True)
class BlockingConfig:
    """How query facets are matched to :class:`DerivedSignal` records.

    Values within the same ``(signal_kind, namespace)`` group are alternatives.  ``match_mode``
    controls whether an operation must match every namespace group or at least one group.
    """

    enabled: bool = True
    match_mode: Literal["all", "any"] = "all"
    minimum_signal_confidence: float = 0.0
    empty_result: Literal["keep_empty", "use_unblocked_catalog"] = "keep_empty"

    def __post_init__(self) -> None:
        if self.match_mode not in {"all", "any"}:
            raise RetrievalConfigurationError(f"unknown blocking match mode: {self.match_mode}")
        if not 0.0 <= self.minimum_signal_confidence <= 1.0:
            raise RetrievalConfigurationError("minimum signal confidence must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class RetrievalChannelConfig:
    """Configuration for one independently ranked retrieval channel."""

    channel_id: str
    method: RetrievalMethod
    weight: float = 1.0
    candidate_limit: int = 100
    minimum_score: float = 0.0
    fields: tuple[OperationField, ...] = (
        "identity",
        "signature",
        "docstring",
    )
    word_ngram_range: tuple[int, int] = (1, 2)
    char_ngram_range: tuple[int, int] = (3, 5)
    alias_signal_namespaces: tuple[str, ...] = ("alias", "import", "identity")
    minimum_alias_confidence: float = 0.0

    def __post_init__(self) -> None:
        if not self.channel_id.strip():
            raise RetrievalConfigurationError("channel_id cannot be empty")
        if self.method not in {"exact", "word_tfidf", "char_tfidf", "sqlite_fts5"}:
            raise RetrievalConfigurationError(f"unknown retrieval method: {self.method}")
        if self.weight <= 0:
            raise RetrievalConfigurationError("channel weight must be positive")
        if self.candidate_limit <= 0:
            raise RetrievalConfigurationError("candidate_limit must be positive")
        if self.minimum_score < 0:
            raise RetrievalConfigurationError("minimum_score cannot be negative")
        if not 0.0 <= self.minimum_alias_confidence <= 1.0:
            raise RetrievalConfigurationError("minimum alias confidence must be in [0, 1]")
        _validate_ngram_range(self.word_ngram_range, "word")
        _validate_ngram_range(self.char_ngram_range, "character")
        unknown_fields = set(self.fields) - {
            "identity",
            "package_name",
            "module",
            "qualified_name",
            "signature",
            "docstring",
        }
        if unknown_fields:
            raise RetrievalConfigurationError(
                f"unknown operation fields: {', '.join(sorted(unknown_fields))}"
            )


@dataclass(frozen=True, slots=True)
class FusionConfig:
    """Reciprocal-rank fusion settings."""

    rank_constant: float = 60.0

    def __post_init__(self) -> None:
        if self.rank_constant < 0:
            raise RetrievalConfigurationError("rank_constant cannot be negative")


@dataclass(frozen=True, slots=True)
class AbstentionConfig:
    """Map normalized fused scores to confidence and abstain below ``threshold``.

    ``calibration_points`` is a monotonic piecewise-linear calibration curve learned outside this
    module, normally on a held-out calibration split.  Identity calibration is the default.
    """

    threshold: float = 0.0
    calibration_points: tuple[tuple[float, float], ...] = ((0.0, 0.0), (1.0, 1.0))
    score_mode: Literal["rrf_consensus", "rrf_consensus_x_channel_score"] = (
        "rrf_consensus_x_channel_score"
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise RetrievalConfigurationError("abstention threshold must be in [0, 1]")
        if self.score_mode not in {"rrf_consensus", "rrf_consensus_x_channel_score"}:
            raise RetrievalConfigurationError(f"unknown abstention score mode: {self.score_mode}")
        if len(self.calibration_points) < 2:
            raise RetrievalConfigurationError("at least two calibration points are required")
        previous_x = -math.inf
        previous_y = -math.inf
        for x_value, y_value in self.calibration_points:
            if not 0.0 <= x_value <= 1.0 or not 0.0 <= y_value <= 1.0:
                raise RetrievalConfigurationError("calibration points must lie in [0, 1]")
            if x_value <= previous_x:
                raise RetrievalConfigurationError("calibration x values must be strictly increasing")
            if y_value < previous_y:
                raise RetrievalConfigurationError("calibration y values must be nondecreasing")
            previous_x = x_value
            previous_y = y_value


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Complete, serializable description of one retrieval experiment."""

    config_id: str
    channels: tuple[RetrievalChannelConfig, ...]
    result_limit: int = 20
    blocking: BlockingConfig = BlockingConfig()
    fusion: FusionConfig = FusionConfig()
    abstention: AbstentionConfig = AbstentionConfig()

    def __post_init__(self) -> None:
        if not self.config_id.strip():
            raise RetrievalConfigurationError("config_id cannot be empty")
        if not self.channels:
            raise RetrievalConfigurationError("at least one retrieval channel is required")
        identifiers = [channel.channel_id for channel in self.channels]
        if len(identifiers) != len(set(identifiers)):
            raise RetrievalConfigurationError("retrieval channel IDs must be unique")
        if self.result_limit <= 0:
            raise RetrievalConfigurationError("result_limit must be positive")


DEFAULT_RETRIEVAL_CONFIG = RetrievalConfig(
    config_id="exact_identity_v1",
    channels=(RetrievalChannelConfig(channel_id="exact_identity", method="exact"),),
)


def _required_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RetrievalConfigurationError(f"{label} must be an object")
    return value


def _two_items(value: object, label: str) -> tuple[object, object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise RetrievalConfigurationError(f"{label} must contain exactly two integers")
    return value[0], value[1]


def retrieval_config_from_mapping(payload: Mapping[str, object]) -> RetrievalConfig:
    """Build a validated retrieval configuration from a JSON-compatible mapping."""

    raw_channels = payload.get("channels")
    if not isinstance(raw_channels, Sequence) or isinstance(raw_channels, (str, bytes)):
        raise RetrievalConfigurationError("channels must be an array")
    channels: list[RetrievalChannelConfig] = []
    for index, raw_channel in enumerate(raw_channels):
        channel = _required_mapping(raw_channel, f"channels[{index}]")
        fields_value = channel.get("fields", ("identity", "signature", "docstring"))
        if not isinstance(fields_value, Sequence) or isinstance(fields_value, (str, bytes)):
            raise RetrievalConfigurationError(f"channels[{index}].fields must be an array")
        alias_namespaces = channel.get(
            "alias_signal_namespaces", ("alias", "import", "identity")
        )
        if not isinstance(alias_namespaces, Sequence) or isinstance(
            alias_namespaces, (str, bytes)
        ):
            raise RetrievalConfigurationError(
                f"channels[{index}].alias_signal_namespaces must be an array"
            )
        channels.append(
            RetrievalChannelConfig(
                channel_id=str(channel["channel_id"]),
                method=str(channel["method"]),  # type: ignore[arg-type]
                weight=float(channel.get("weight", 1.0)),
                candidate_limit=int(channel.get("candidate_limit", 100)),
                minimum_score=float(channel.get("minimum_score", 0.0)),
                fields=tuple(str(item) for item in fields_value),  # type: ignore[arg-type]
                word_ngram_range=tuple(
                    int(item)
                    for item in _two_items(
                        channel.get("word_ngram_range", (1, 2)),
                        f"channels[{index}].word_ngram_range",
                    )
                ),
                char_ngram_range=tuple(
                    int(item)
                    for item in _two_items(
                        channel.get("char_ngram_range", (3, 5)),
                        f"channels[{index}].char_ngram_range",
                    )
                ),
                alias_signal_namespaces=tuple(str(item) for item in alias_namespaces),
                minimum_alias_confidence=float(
                    channel.get("minimum_alias_confidence", 0.0)
                ),
            )
        )

    blocking = _required_mapping(payload.get("blocking", {}), "blocking")
    fusion = _required_mapping(payload.get("fusion", {}), "fusion")
    abstention = _required_mapping(payload.get("abstention", {}), "abstention")
    raw_points = abstention.get("calibration_points", ((0.0, 0.0), (1.0, 1.0)))
    if not isinstance(raw_points, Sequence) or isinstance(raw_points, (str, bytes)):
        raise RetrievalConfigurationError("abstention.calibration_points must be an array")
    calibration_points = tuple(
        tuple(float(item) for item in _two_items(point, "calibration point"))
        for point in raw_points
    )
    return RetrievalConfig(
        config_id=str(payload["config_id"]),
        channels=tuple(channels),
        result_limit=int(payload.get("result_limit", 20)),
        blocking=BlockingConfig(
            enabled=bool(blocking.get("enabled", True)),
            match_mode=str(blocking.get("match_mode", "all")),  # type: ignore[arg-type]
            minimum_signal_confidence=float(
                blocking.get("minimum_signal_confidence", 0.0)
            ),
            empty_result=str(blocking.get("empty_result", "keep_empty")),  # type: ignore[arg-type]
        ),
        fusion=FusionConfig(rank_constant=float(fusion.get("rank_constant", 60.0))),
        abstention=AbstentionConfig(
            threshold=float(abstention.get("threshold", 0.0)),
            calibration_points=calibration_points,
            score_mode=str(
                abstention.get("score_mode", "rrf_consensus_x_channel_score")
            ),  # type: ignore[arg-type]
        ),
    )


def load_retrieval_profiles(path: str | Path) -> dict[str, RetrievalConfig]:
    """Load named experiment profiles from a versioned JSON document."""

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, Mapping) or str(document.get("schema_version")) != "1":
        raise RetrievalConfigurationError("retrieval profile document must use schema_version 1")
    records = document.get("profiles")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise RetrievalConfigurationError("profiles must be an array")
    profiles: dict[str, RetrievalConfig] = {}
    for index, raw_record in enumerate(records):
        record = dict(_required_mapping(raw_record, f"profiles[{index}]"))
        profile_id = str(record.pop("profile_id", "")).strip()
        if not profile_id:
            raise RetrievalConfigurationError(f"profiles[{index}].profile_id cannot be empty")
        if profile_id in profiles:
            raise RetrievalConfigurationError(f"duplicate retrieval profile: {profile_id}")
        profiles[profile_id] = retrieval_config_from_mapping(record)
    return profiles


@dataclass(frozen=True, slots=True)
class ChannelHit:
    operation_id: str
    score: float
    rank: int
    details: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class HitProvenance:
    channel_id: str
    method: RetrievalMethod
    channel_rank: int
    channel_score: float
    channel_weight: float
    fused_contribution: float
    details: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class FusedCandidate:
    operation_id: str
    score: float
    provenance: tuple[HitProvenance, ...]


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    operation: OperationRecord
    rank: int
    score: float
    provenance: tuple[HitProvenance, ...]

    @property
    def operation_id(self) -> str:
        return self.operation.operation_id


@dataclass(frozen=True, slots=True)
class BlockingOutcome:
    operation_ids: tuple[str, ...]
    catalog_size: int
    blocking_applied: bool
    query_signals: tuple[QuerySignal, ...]
    used_unblocked_fallback: bool = False

    def __post_init__(self) -> None:
        if self.catalog_size < 0:
            raise ValueError("catalog_size cannot be negative")
        if len(self.operation_ids) != len(set(self.operation_ids)):
            raise ValueError("blocked operation IDs must be unique")
        if len(self.operation_ids) > self.catalog_size:
            raise ValueError("candidate count cannot exceed catalog size")

    @property
    def candidate_count(self) -> int:
        return len(self.operation_ids)

    @property
    def candidate_reduction(self) -> float:
        if self.catalog_size == 0:
            return 0.0
        return 1.0 - (self.candidate_count / self.catalog_size)


@dataclass(frozen=True, slots=True)
class RetrievalResponse:
    query: str
    config_id: str
    hits: tuple[RetrievalHit, ...]
    abstained: bool
    confidence: float
    normalized_top_score: float
    abstention_threshold: float
    blocking: BlockingOutcome
    active_channel_ids: tuple[str, ...]

    @property
    def operation_ids(self) -> tuple[str, ...]:
        return tuple(hit.operation_id for hit in self.hits)


def _normalize_identity(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _normalize_signal(value: str) -> str:
    return " ".join(value.casefold().split())


def _operation_identity_aliases(
    operation: OperationRecord,
    alias_values: Iterable[str] = (),
) -> tuple[str, ...]:
    qualified_leaf = operation.qualified_name.rsplit(".", 1)[-1]
    module_leaf = operation.module.rsplit(".", 1)[-1]
    raw_aliases = {
        operation.operation_id,
        operation.package_name,
        operation.module,
        operation.qualified_name,
        qualified_leaf,
        module_leaf,
        f"{operation.module}.{operation.qualified_name}",
        f"{operation.module}:{operation.qualified_name}",
        f"{operation.package_name}.{qualified_leaf}",
        *alias_values,
    }
    aliases = {_normalize_identity(alias) for alias in raw_aliases if alias.strip()}
    return tuple(sorted(alias for alias in aliases if alias))


def _operation_text(operation: OperationRecord, selected_fields: Sequence[OperationField]) -> str:
    parts: list[str] = []
    for selected_field in selected_fields:
        if selected_field == "identity":
            parts.extend(
                (
                    operation.operation_id,
                    operation.package_name,
                    operation.module,
                    operation.qualified_name,
                )
            )
        else:
            parts.append(str(getattr(operation, selected_field)))
    return "\n".join(part for part in parts if part)


def filter_by_derived_signals(
    operations: Sequence[OperationRecord],
    derived_signals: Sequence[DerivedSignal],
    query_signals: Sequence[QuerySignal],
    config: BlockingConfig,
) -> BlockingOutcome:
    """Return the deterministic candidate block for the supplied query facets."""

    ordered_ids = tuple(sorted(operation.operation_id for operation in operations))
    normalized_queries = tuple(
        QuerySignal(
            namespace=_normalize_signal(signal.namespace),
            value=_normalize_signal(signal.value),
            signal_kind=signal.signal_kind,
        )
        for signal in query_signals
        if signal.namespace.strip() and signal.value.strip()
    )
    if not config.enabled or not normalized_queries:
        return BlockingOutcome(
            operation_ids=ordered_ids,
            catalog_size=len(operations),
            blocking_applied=False,
            query_signals=normalized_queries,
        )

    query_groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for signal in normalized_queries:
        query_groups[(signal.signal_kind, signal.namespace)].add(signal.value)

    operation_groups: dict[str, dict[tuple[str, str], set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    known_operation_ids = set(ordered_ids)
    for signal in derived_signals:
        if signal.operation_id not in known_operation_ids:
            continue
        if signal.confidence < config.minimum_signal_confidence:
            continue
        key = (signal.signal_kind, _normalize_signal(signal.namespace))
        operation_groups[signal.operation_id][key].add(_normalize_signal(signal.value))

    matching_ids: list[str] = []
    for operation_id in ordered_ids:
        candidate_groups = operation_groups.get(operation_id, {})
        group_matches = [
            bool(candidate_groups.get(group_key, set()) & requested_values)
            for group_key, requested_values in sorted(query_groups.items())
        ]
        matched = all(group_matches) if config.match_mode == "all" else any(group_matches)
        if matched:
            matching_ids.append(operation_id)

    used_fallback = False
    if not matching_ids and config.empty_result == "use_unblocked_catalog":
        matching_ids = list(ordered_ids)
        used_fallback = True

    return BlockingOutcome(
        operation_ids=tuple(matching_ids),
        catalog_size=len(operations),
        blocking_applied=True,
        query_signals=normalized_queries,
        used_unblocked_fallback=used_fallback,
    )


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[ChannelHit]],
    channel_configs: Mapping[str, RetrievalChannelConfig],
    fusion: FusionConfig = FusionConfig(),
) -> tuple[FusedCandidate, ...]:
    """Fuse channel rankings deterministically while retaining every contribution."""

    scores: dict[str, float] = defaultdict(float)
    provenance: dict[str, list[HitProvenance]] = defaultdict(list)
    for channel_id in sorted(rankings):
        if channel_id not in channel_configs:
            raise RetrievalConfigurationError(f"missing configuration for channel {channel_id!r}")
        channel = channel_configs[channel_id]
        seen: set[str] = set()
        for fallback_rank, hit in enumerate(rankings[channel_id], start=1):
            if hit.operation_id in seen:
                continue
            seen.add(hit.operation_id)
            rank = hit.rank if hit.rank > 0 else fallback_rank
            contribution = channel.weight / (fusion.rank_constant + rank)
            scores[hit.operation_id] += contribution
            provenance[hit.operation_id].append(
                HitProvenance(
                    channel_id=channel_id,
                    method=channel.method,
                    channel_rank=rank,
                    channel_score=hit.score,
                    channel_weight=channel.weight,
                    fused_contribution=contribution,
                    details=hit.details,
                )
            )

    ordered = sorted(scores, key=lambda operation_id: (-scores[operation_id], operation_id))
    return tuple(
        FusedCandidate(
            operation_id=operation_id,
            score=scores[operation_id],
            provenance=tuple(
                sorted(
                    provenance[operation_id],
                    key=lambda item: (item.channel_id, item.channel_rank),
                )
            ),
        )
        for operation_id in ordered
    )


def calibrate_score(raw_score: float, config: AbstentionConfig) -> float:
    """Apply the configured monotonic piecewise-linear calibration curve."""

    clipped = min(1.0, max(0.0, raw_score))
    points = config.calibration_points
    if clipped <= points[0][0]:
        return points[0][1]
    if clipped >= points[-1][0]:
        return points[-1][1]
    for (left_x, left_y), (right_x, right_y) in zip(points, points[1:]):
        if left_x <= clipped <= right_x:
            fraction = (clipped - left_x) / (right_x - left_x)
            return left_y + fraction * (right_y - left_y)
    raise AssertionError("validated calibration curve did not cover the clipped score")


class OperationRetriever:
    """Search an operation sequence or the ``operations`` table of a SQLite catalog."""

    def __init__(
        self,
        operations: Sequence[OperationRecord] | None = None,
        *,
        derived_signals: Sequence[DerivedSignal] = (),
        sqlite_source: sqlite3.Connection | str | Path | None = None,
    ) -> None:
        if operations is None:
            if sqlite_source is None:
                raise ValueError("provide operations or sqlite_source")
            operations = load_operations_from_sqlite(sqlite_source)
        ordered_operations = tuple(sorted(operations, key=lambda item: item.operation_id))
        operation_ids = [operation.operation_id for operation in ordered_operations]
        if len(operation_ids) != len(set(operation_ids)):
            raise ValueError("operation IDs must be unique")
        self.operations = ordered_operations
        self.derived_signals = tuple(derived_signals)
        self.sqlite_source = sqlite_source
        self._operation_by_id = {
            operation.operation_id: operation for operation in ordered_operations
        }

    def search(
        self,
        query: str,
        *,
        config: RetrievalConfig = DEFAULT_RETRIEVAL_CONFIG,
        query_signals: Sequence[QuerySignal] = (),
    ) -> RetrievalResponse:
        if not query.strip():
            raise ValueError("query cannot be empty")

        blocking = filter_by_derived_signals(
            self.operations,
            self.derived_signals,
            query_signals,
            config.blocking,
        )
        allowed_ids = set(blocking.operation_ids)
        candidates = tuple(
            operation for operation in self.operations if operation.operation_id in allowed_ids
        )

        rankings: dict[str, tuple[ChannelHit, ...]] = {}
        channel_by_id = {channel.channel_id: channel for channel in config.channels}
        for channel in config.channels:
            rankings[channel.channel_id] = self._run_channel(query, candidates, channel)

        fused = reciprocal_rank_fusion(rankings, channel_by_id, config.fusion)
        fused = fused[: config.result_limit]
        hits = tuple(
            RetrievalHit(
                operation=self._operation_by_id[item.operation_id],
                rank=rank,
                score=item.score,
                provenance=item.provenance,
            )
            for rank, item in enumerate(fused, start=1)
        )

        active_channels = tuple(
            channel.channel_id for channel in config.channels if rankings[channel.channel_id]
        )
        maximum_top_score = sum(
            channel_by_id[channel_id].weight / (config.fusion.rank_constant + 1)
            for channel_id in active_channels
        )
        rrf_consensus = (
            min(1.0, hits[0].score / maximum_top_score)
            if hits and maximum_top_score > 0
            else 0.0
        )
        if hits and config.abstention.score_mode == "rrf_consensus_x_channel_score":
            top_provenance = hits[0].provenance
            active_weight = sum(channel_by_id[item].weight for item in active_channels)
            channel_strength = (
                sum(
                    item.channel_weight * min(1.0, max(0.0, item.channel_score))
                    for item in top_provenance
                )
                / active_weight
                if active_weight > 0
                else 0.0
            )
            normalized_top_score = rrf_consensus * channel_strength
        else:
            normalized_top_score = rrf_consensus
        confidence = calibrate_score(normalized_top_score, config.abstention)
        abstained = not hits or confidence < config.abstention.threshold

        return RetrievalResponse(
            query=query,
            config_id=config.config_id,
            hits=hits,
            abstained=abstained,
            confidence=confidence,
            normalized_top_score=normalized_top_score,
            abstention_threshold=config.abstention.threshold,
            blocking=blocking,
            active_channel_ids=active_channels,
        )

    def _run_channel(
        self,
        query: str,
        operations: Sequence[OperationRecord],
        channel: RetrievalChannelConfig,
    ) -> tuple[ChannelHit, ...]:
        if not operations:
            return ()
        if channel.method == "exact":
            return self._exact_hits(query, operations, channel)
        if channel.method in {"word_tfidf", "char_tfidf"}:
            return self._tfidf_hits(query, operations, channel)
        if channel.method == "sqlite_fts5":
            return self._fts5_hits(query, operations, channel)
        raise RetrievalConfigurationError(f"unknown retrieval method: {channel.method}")

    def _exact_hits(
        self,
        query: str,
        operations: Sequence[OperationRecord],
        channel: RetrievalChannelConfig,
    ) -> tuple[ChannelHit, ...]:
        normalized_query = _normalize_identity(query)
        scored: list[tuple[str, float, str]] = []
        allowed_namespaces = {_normalize_signal(item) for item in channel.alias_signal_namespaces}
        signals_by_operation: dict[str, list[str]] = defaultdict(list)
        for signal in self.derived_signals:
            if _normalize_signal(signal.namespace) not in allowed_namespaces:
                continue
            if signal.confidence < channel.minimum_alias_confidence:
                continue
            signals_by_operation[signal.operation_id].append(signal.value)

        for operation in operations:
            aliases = _operation_identity_aliases(
                operation,
                signals_by_operation.get(operation.operation_id, ()),
            )
            best_score = 0.0
            best_alias = ""
            for alias in aliases:
                if normalized_query == alias:
                    score = 1.0
                elif alias and f" {alias} " in f" {normalized_query} ":
                    token_count = len(alias.split())
                    score = 0.85 + min(token_count, 5) * 0.025
                else:
                    continue
                if score > best_score or (score == best_score and alias < best_alias):
                    best_score = score
                    best_alias = alias
            if best_score >= channel.minimum_score and best_score > 0:
                scored.append((operation.operation_id, best_score, best_alias))

        scored.sort(key=lambda item: (-item[1], item[0]))
        return tuple(
            ChannelHit(
                operation_id=operation_id,
                score=score,
                rank=rank,
                details=(("matched_alias", alias),),
            )
            for rank, (operation_id, score, alias) in enumerate(
                scored[: channel.candidate_limit], start=1
            )
        )

    def _tfidf_hits(
        self,
        query: str,
        operations: Sequence[OperationRecord],
        channel: RetrievalChannelConfig,
    ) -> tuple[ChannelHit, ...]:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as error:  # pragma: no cover - exercised in minimal installations
            raise OptionalRetrievalDependencyError(
                f"retrieval channel {channel.channel_id!r} uses {channel.method}; "
                "install the optional dependency with `pip install existing-code-reuse[retrieval]`"
            ) from error

        documents = [_operation_text(operation, channel.fields) for operation in operations]
        if channel.method == "word_tfidf":
            vectorizer = TfidfVectorizer(
                analyzer="word",
                ngram_range=channel.word_ngram_range,
                strip_accents="unicode",
                sublinear_tf=True,
            )
        else:
            vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=channel.char_ngram_range,
                strip_accents="unicode",
                sublinear_tf=True,
            )
        try:
            document_matrix = vectorizer.fit_transform(documents)
        except ValueError as error:
            if "empty vocabulary" in str(error).lower():
                return ()
            raise
        query_vector = vectorizer.transform([query])
        scores = (document_matrix @ query_vector.T).toarray().ravel()
        scored = [
            (operation.operation_id, float(score))
            for operation, score in zip(operations, scores, strict=True)
            if score > 0 and score >= channel.minimum_score
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return tuple(
            ChannelHit(
                operation_id=operation_id,
                score=score,
                rank=rank,
                details=(
                    ("analyzer", "word" if channel.method == "word_tfidf" else "char_wb"),
                    ("fields", ",".join(channel.fields)),
                ),
            )
            for rank, (operation_id, score) in enumerate(
                scored[: channel.candidate_limit], start=1
            )
        )

    def _fts5_hits(
        self,
        query: str,
        operations: Sequence[OperationRecord],
        channel: RetrievalChannelConfig,
    ) -> tuple[ChannelHit, ...]:
        allowed_ids = {operation.operation_id for operation in operations}
        expression = _fts_expression(query)
        if not expression:
            return ()

        rows: list[tuple[str, float, str]]
        if self.sqlite_source is not None:
            with _sqlite_connection(self.sqlite_source) as connection:
                if _table_exists(connection, "representations_fts"):
                    rows = _search_existing_fts(
                        connection,
                        expression,
                        allowed_ids,
                        channel.candidate_limit,
                    )
                else:
                    rows = _search_ephemeral_fts(
                        query,
                        operations,
                        channel.fields,
                        channel.candidate_limit,
                    )
        else:
            rows = _search_ephemeral_fts(
                query,
                operations,
                channel.fields,
                channel.candidate_limit,
            )

        filtered_rows = [row for row in rows if row[0] in allowed_ids]
        hits: list[ChannelHit] = []
        for source_rank, (operation_id, bm25_score, source) in enumerate(
            filtered_rows[: channel.candidate_limit], start=1
        ):
            score = 1.0 / source_rank
            if score < channel.minimum_score:
                continue
            hits.append(
                ChannelHit(
                    operation_id=operation_id,
                    score=score,
                    rank=len(hits) + 1,
                    details=(("fts_source", source), ("bm25", f"{bm25_score:.12g}")),
                )
            )
        return tuple(hits)


def search_operations(
    query: str,
    *,
    operations: Sequence[OperationRecord] | None = None,
    derived_signals: Sequence[DerivedSignal] = (),
    sqlite_source: sqlite3.Connection | str | Path | None = None,
    query_signals: Sequence[QuerySignal] = (),
    config: RetrievalConfig = DEFAULT_RETRIEVAL_CONFIG,
) -> RetrievalResponse:
    """Convenience API for one search without explicitly constructing a retriever."""

    return OperationRetriever(
        operations,
        derived_signals=derived_signals,
        sqlite_source=sqlite_source,
    ).search(query, config=config, query_signals=query_signals)


@contextmanager
def _sqlite_connection(
    source: sqlite3.Connection | str | Path,
) -> Iterator[sqlite3.Connection]:
    if isinstance(source, sqlite3.Connection):
        yield source
        return
    connection = sqlite3.connect(str(source))
    try:
        yield connection
    finally:
        connection.close()


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    for schema in ("sqlite_master", "sqlite_temp_master"):
        row = connection.execute(
            f"SELECT 1 FROM {schema} WHERE name = ? LIMIT 1",  # noqa: S608
            (table_name,),
        ).fetchone()
        if row is not None:
            return True
    return False


def _fts_expression(query: str) -> str:
    tokens = re.findall(r"\w+", query.casefold(), flags=re.UNICODE)
    return " OR ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)


def _search_existing_fts(
    connection: sqlite3.Connection,
    expression: str,
    allowed_ids: set[str],
    limit: int,
) -> list[tuple[str, float, str]]:
    try:
        cursor = connection.execute(
            """
            SELECT operation_id, bm25(representations_fts) AS relevance
            FROM representations_fts
            WHERE representations_fts MATCH ?
            ORDER BY relevance ASC, operation_id ASC
            """,
            (expression,),
        )
    except sqlite3.OperationalError as error:
        raise RetrievalConfigurationError(
            "SQLite FTS5 search failed; verify that the catalog contains a valid "
            "representations_fts virtual table"
        ) from error

    rows: list[tuple[str, float, str]] = []
    seen: set[str] = set()
    for operation_id, relevance in cursor:
        operation_id = str(operation_id)
        if operation_id in seen or operation_id not in allowed_ids:
            continue
        seen.add(operation_id)
        rows.append((operation_id, float(relevance), "catalog_representations_fts"))
        if len(rows) >= limit:
            break
    return rows


def _search_ephemeral_fts(
    query: str,
    operations: Sequence[OperationRecord],
    selected_fields: Sequence[OperationField],
    limit: int,
) -> list[tuple[str, float, str]]:
    expression = _fts_expression(query)
    if not expression:
        return []
    connection = sqlite3.connect(":memory:")
    try:
        try:
            connection.execute(
                "CREATE VIRTUAL TABLE operation_search USING fts5(operation_id UNINDEXED, text)"
            )
        except sqlite3.OperationalError as error:
            raise RetrievalConfigurationError(
                "the active SQLite build does not provide FTS5; choose another retrieval "
                "channel or install Python with SQLite FTS5 support"
            ) from error
        connection.executemany(
            "INSERT INTO operation_search(operation_id, text) VALUES (?, ?)",
            (
                (operation.operation_id, _operation_text(operation, selected_fields))
                for operation in operations
            ),
        )
        raw_rows = connection.execute(
            """
            SELECT operation_id, bm25(operation_search) AS relevance
            FROM operation_search
            WHERE operation_search MATCH ?
            ORDER BY relevance ASC, operation_id ASC
            LIMIT ?
            """,
            (expression, limit),
        ).fetchall()
        return [
            (str(operation_id), float(relevance), "ephemeral_operation_fts")
            for operation_id, relevance in raw_rows
        ]
    finally:
        connection.close()


def load_operations_from_sqlite(
    source: sqlite3.Connection | str | Path,
) -> tuple[OperationRecord, ...]:
    """Load operation records from the normalized SQLite catalog without importing storage.py."""

    with _sqlite_connection(source) as connection:
        if not _table_exists(connection, "operations"):
            raise ValueError("SQLite catalog does not contain an operations table")
        cursor = connection.execute("SELECT * FROM operations ORDER BY operation_id")
        column_names = tuple(description[0] for description in cursor.description or ())
        rows = cursor.fetchall()

    model_fields = {item.name: item for item in fields(OperationRecord)}
    records: list[OperationRecord] = []
    for raw_row in rows:
        row = dict(zip(column_names, raw_row, strict=True))
        payload_value = row.get("record_json", row.get("payload_json"))
        if payload_value is not None:
            payload = json.loads(str(payload_value))
        else:
            payload = {name: row[name] for name in model_fields if name in row}
        for integer_field in ("line_start", "line_end"):
            if integer_field in payload:
                payload[integer_field] = int(payload[integer_field])
        try:
            records.append(OperationRecord(**payload))
        except TypeError as error:
            available = ", ".join(sorted(payload))
            raise ValueError(
                f"invalid operation row in SQLite catalog; available fields: {available}"
            ) from error
    return tuple(records)


__all__ = [
    "AbstentionConfig",
    "BlockingConfig",
    "BlockingOutcome",
    "ChannelHit",
    "DEFAULT_RETRIEVAL_CONFIG",
    "FusedCandidate",
    "FusionConfig",
    "HitProvenance",
    "OperationRetriever",
    "OptionalRetrievalDependencyError",
    "QuerySignal",
    "RetrievalChannelConfig",
    "RetrievalConfig",
    "RetrievalConfigurationError",
    "RetrievalHit",
    "RetrievalResponse",
    "calibrate_score",
    "filter_by_derived_signals",
    "load_operations_from_sqlite",
    "load_retrieval_profiles",
    "reciprocal_rank_fusion",
    "retrieval_config_from_mapping",
    "search_operations",
]
