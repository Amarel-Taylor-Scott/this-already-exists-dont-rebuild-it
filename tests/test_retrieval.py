from __future__ import annotations

import builtins
from pathlib import Path
import sqlite3

import pytest

from existing_code_reuse.models import DerivedSignal, OperationRecord
from existing_code_reuse.retrieval import (
    AbstentionConfig,
    BlockingConfig,
    OperationRetriever,
    OptionalRetrievalDependencyError,
    QuerySignal,
    RetrievalChannelConfig,
    RetrievalConfig,
    calibrate_score,
    load_operations_from_sqlite,
    load_retrieval_profiles,
)


ROOT = Path(__file__).resolve().parents[1]


def _operation(
    operation_id: str,
    package: str,
    module: str,
    qualified_name: str,
    docstring: str,
) -> OperationRecord:
    return OperationRecord(
        operation_id=operation_id,
        package_id=f"pypi:{package}@1.0",
        package_name=package,
        package_version="1.0",
        module=module,
        qualified_name=qualified_name,
        kind="function",
        signature=f"{qualified_name}(value)",
        docstring=docstring,
        relative_path=f"{package}/api.py",
        line_start=1,
        line_end=3,
        source_digest=f"sha256:{operation_id}",
        visibility="public",
    )


@pytest.fixture
def operations() -> tuple[OperationRecord, ...]:
    return (
        _operation(
            "pandas.read_csv@1.0",
            "pandas",
            "pandas.io.parsers",
            "read_csv",
            "Read a comma-separated table into a dataframe.",
        ),
        _operation(
            "jsonschema.validate@1.0",
            "jsonschema",
            "jsonschema.validators",
            "validate",
            "Validate a JSON instance against a JSON schema.",
        ),
        _operation(
            "sklearn.calibrate@1.0",
            "scikit-learn",
            "sklearn.calibration",
            "calibrate_probabilities",
            "Calibrate classifier probabilities with isotonic regression.",
        ),
        _operation(
            "text.clean@1.0",
            "text-tools",
            "text_tools.clean",
            "clean_text",
            "Normalize whitespace and lowercase ordinary text.",
        ),
    )


@pytest.fixture
def signals() -> tuple[DerivedSignal, ...]:
    return (
        DerivedSignal(
            signal_id="alias:read-table",
            operation_id="pandas.read_csv@1.0",
            signal_kind="label",
            namespace="alias",
            value="open delimited table",
            confidence=1.0,
            evidence_fields=("docstring",),
            generator="fixture",
            generator_version="1",
        ),
        DerivedSignal(
            signal_id="block:table",
            operation_id="pandas.read_csv@1.0",
            signal_kind="blocking_key",
            namespace="output-artifact",
            value="table",
            confidence=1.0,
            evidence_fields=("docstring",),
            generator="fixture",
            generator_version="1",
        ),
        DerivedSignal(
            signal_id="block:calibrated",
            operation_id="sklearn.calibrate@1.0",
            signal_kind="blocking_key",
            namespace="output-artifact",
            value="calibrated-probabilities",
            confidence=0.95,
            evidence_fields=("docstring",),
            generator="fixture",
            generator_version="1",
        ),
    )


def _config(
    *channels: RetrievalChannelConfig,
    threshold: float = 0.0,
    blocking: BlockingConfig = BlockingConfig(),
    limit: int = 10,
) -> RetrievalConfig:
    return RetrievalConfig(
        config_id="fixture",
        channels=channels,
        result_limit=limit,
        blocking=blocking,
        abstention=AbstentionConfig(threshold=threshold),
    )


def test_exact_identity_and_derived_alias_matching(operations, signals) -> None:
    retriever = OperationRetriever(operations, derived_signals=signals)
    exact = RetrievalChannelConfig(channel_id="exact", method="exact")

    identity_response = retriever.search("please use jsonschema.validate", config=_config(exact))
    alias_response = retriever.search("open delimited table", config=_config(exact))

    assert identity_response.operation_ids[0] == "jsonschema.validate@1.0"
    assert alias_response.operation_ids[0] == "pandas.read_csv@1.0"
    assert alias_response.hits[0].provenance[0].details == (
        ("matched_alias", "open delimited table"),
    )


def test_word_and_character_tfidf_are_independently_configurable(operations) -> None:
    pytest.importorskip("sklearn")
    retriever = OperationRetriever(operations)
    word = RetrievalChannelConfig(channel_id="word", method="word_tfidf")
    character = RetrievalChannelConfig(
        channel_id="character",
        method="char_tfidf",
        char_ngram_range=(2, 5),
    )

    word_response = retriever.search(
        "validate JSON against a schema",
        config=_config(word),
    )
    character_response = retriever.search(
        "calbrate clasifier probabilties isotonic",
        config=_config(character),
    )

    assert word_response.operation_ids[0] == "jsonschema.validate@1.0"
    assert character_response.operation_ids[0] == "sklearn.calibrate@1.0"


def test_tfidf_reports_the_optional_dependency_action(operations, monkeypatch) -> None:
    real_import = builtins.__import__

    def reject_sklearn(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("sklearn"):
            raise ImportError("simulated optional dependency absence")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", reject_sklearn)
    retriever = OperationRetriever(operations)
    word = RetrievalChannelConfig(channel_id="word", method="word_tfidf")

    with pytest.raises(OptionalRetrievalDependencyError, match=r"\[retrieval\]"):
        retriever.search("validate JSON", config=_config(word))


def test_deterministic_blocking_filters_before_ranking(operations, signals) -> None:
    retriever = OperationRetriever(operations, derived_signals=signals)
    word = RetrievalChannelConfig(channel_id="word", method="word_tfidf")
    response = retriever.search(
        "produce probabilities from a classifier",
        query_signals=(
            QuerySignal(
                namespace="output-artifact",
                value="calibrated-probabilities",
            ),
        ),
        config=_config(word),
    )

    assert response.blocking.blocking_applied
    assert response.blocking.operation_ids == ("sklearn.calibrate@1.0",)
    assert response.blocking.candidate_count == 1
    assert response.blocking.candidate_reduction == pytest.approx(0.75)
    assert response.operation_ids == ("sklearn.calibrate@1.0",)


def test_rank_fusion_retains_channel_provenance(operations) -> None:
    pytest.importorskip("sklearn")
    retriever = OperationRetriever(operations)
    config = _config(
        RetrievalChannelConfig(channel_id="identity", method="exact"),
        RetrievalChannelConfig(channel_id="documentation", method="word_tfidf"),
    )
    response = retriever.search(
        "jsonschema validate JSON instance against schema",
        config=config,
    )

    top = response.hits[0]
    assert top.operation_id == "jsonschema.validate@1.0"
    assert {item.channel_id for item in top.provenance} == {"identity", "documentation"}
    assert all(item.fused_contribution > 0 for item in top.provenance)
    assert 0.5 < response.normalized_top_score <= 1.0


def test_calibration_and_threshold_produce_explicit_abstention(operations) -> None:
    retriever = OperationRetriever(operations)
    exact = RetrievalChannelConfig(channel_id="exact", method="exact")
    config = RetrievalConfig(
        config_id="calibrated",
        channels=(exact,),
        abstention=AbstentionConfig(
            threshold=0.7,
            calibration_points=((0.0, 0.0), (1.0, 0.6)),
        ),
    )

    matched = retriever.search("jsonschema.validate", config=config)
    unmatched = retriever.search("capability that does not exist", config=config)

    assert matched.operation_ids[0] == "jsonschema.validate@1.0"
    assert matched.confidence == pytest.approx(0.6)
    assert matched.abstained
    assert unmatched.hits == ()
    assert unmatched.abstained
    assert calibrate_score(0.5, config.abstention) == pytest.approx(0.3)


def test_sqlite_fts5_can_search_operation_sequences(operations) -> None:
    retriever = OperationRetriever(operations)
    fts = RetrievalChannelConfig(channel_id="fts", method="sqlite_fts5")
    try:
        response = retriever.search("isotonic classifier", config=_config(fts))
    except sqlite3.OperationalError:
        pytest.skip("SQLite was built without FTS5")

    assert response.operation_ids[0] == "sklearn.calibrate@1.0"
    assert response.hits[0].provenance[0].details[0] == (
        "fts_source",
        "ephemeral_operation_fts",
    )


def test_sqlite_catalog_loading_and_existing_fts(operations) -> None:
    connection = sqlite3.connect(":memory:")
    operation = operations[0]
    columns = tuple(operation.to_dict())
    column_sql = ", ".join(f"{column} TEXT" for column in columns)
    connection.execute(f"CREATE TABLE operations ({column_sql})")
    placeholders = ", ".join("?" for _ in columns)
    connection.execute(
        f"INSERT INTO operations ({', '.join(columns)}) VALUES ({placeholders})",
        tuple(str(value) for value in operation.to_dict().values()),
    )
    try:
        connection.execute(
            "CREATE VIRTUAL TABLE representations_fts USING "
            "fts5(representation_id UNINDEXED, operation_id UNINDEXED, plane UNINDEXED, text)"
        )
    except sqlite3.OperationalError:
        connection.close()
        pytest.skip("SQLite was built without FTS5")
    connection.execute(
        "INSERT INTO representations_fts VALUES (?, ?, ?, ?)",
        ("rep:one", operation.operation_id, "documentation", "load a pipe separated dataset"),
    )

    loaded = load_operations_from_sqlite(connection)
    retriever = OperationRetriever(sqlite_source=connection)
    fts = RetrievalChannelConfig(channel_id="fts", method="sqlite_fts5")
    response = retriever.search("pipe separated dataset", config=_config(fts))

    assert loaded == (operation,)
    assert response.operation_ids == (operation.operation_id,)
    assert response.hits[0].provenance[0].details[0] == (
        "fts_source",
        "catalog_representations_fts",
    )
    connection.close()


def test_versioned_retrieval_profiles_load_from_data() -> None:
    profiles = load_retrieval_profiles(ROOT / "configs" / "retrieval_profiles.json")

    assert set(profiles) == {"exact", "word", "char", "fts", "hybrid"}
    assert [channel.method for channel in profiles["hybrid"].channels] == [
        "word_tfidf",
        "char_tfidf",
    ]
    assert profiles["hybrid"].blocking.enabled is False
    assert profiles["exact"].abstention.threshold == pytest.approx(0.8)
