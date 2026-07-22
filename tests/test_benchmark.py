from __future__ import annotations

import pytest

from existing_code_reuse.benchmark import (
    BenchmarkTask,
    EvaluationConfig,
    evaluate_responses,
    run_offline_benchmark,
)
from existing_code_reuse.models import OperationRecord
from existing_code_reuse.retrieval import (
    AbstentionConfig,
    BlockingOutcome,
    RetrievalChannelConfig,
    RetrievalConfig,
    RetrievalHit,
    RetrievalResponse,
)


def _operation(operation_id: str, name: str) -> OperationRecord:
    return OperationRecord(
        operation_id=operation_id,
        package_id=f"pypi:fixture-{name}@1",
        package_name=f"fixture-{name}",
        package_version="1",
        module=f"fixture.{name}",
        qualified_name=name,
        kind="function",
        signature=f"{name}()",
        docstring=f"Perform the {name} operation.",
        relative_path="fixture.py",
        line_start=1,
        line_end=2,
        source_digest=f"sha256:{name}",
        visibility="public",
    )


def _response(
    operations: dict[str, OperationRecord],
    ranked_ids: tuple[str, ...],
    *,
    candidate_ids: tuple[str, ...],
    catalog_size: int,
    abstained: bool,
) -> RetrievalResponse:
    return RetrievalResponse(
        query="fixture",
        config_id="fixture",
        hits=tuple(
            RetrievalHit(
                operation=operations[operation_id],
                rank=rank,
                score=1.0 / rank,
                provenance=(),
            )
            for rank, operation_id in enumerate(ranked_ids, start=1)
        ),
        abstained=abstained,
        confidence=0.0 if abstained else 1.0,
        normalized_top_score=0.0 if abstained else 1.0,
        abstention_threshold=0.5,
        blocking=BlockingOutcome(
            operation_ids=candidate_ids,
            catalog_size=catalog_size,
            blocking_applied=len(candidate_ids) != catalog_size,
            query_signals=(),
        ),
        active_channel_ids=("fixture",) if ranked_ids else (),
    )


def test_metrics_support_alternative_acceptable_operation_sets() -> None:
    operations = {
        operation_id: _operation(operation_id, operation_id)
        for operation_id in ("a", "b", "c", "d")
    }
    tasks = (
        BenchmarkTask(
            task_id="reuse",
            query="compose a and b",
            acceptable_operation_sets=(frozenset({"a", "b"}), frozenset({"d"})),
            hard_negative_operation_ids=frozenset({"c"}),
        ),
        BenchmarkTask(task_id="novel", query="write novel behavior", no_reuse=True),
    )
    responses = {
        "reuse": _response(
            operations,
            ("b", "a", "c"),
            candidate_ids=("a",),
            catalog_size=4,
            abstained=False,
        ),
        "novel": _response(
            operations,
            (),
            candidate_ids=("a", "b", "c", "d"),
            catalog_size=4,
            abstained=True,
        ),
    }

    report = evaluate_responses(
        tasks,
        responses,
        config=EvaluationConfig(cutoffs=(1, 2, 3), hard_negative_cutoff=3),
    )

    assert [metric.value for metric in report.recall_at_k] == pytest.approx([0.5, 1.0, 1.0])
    assert report.mean_reciprocal_rank == pytest.approx(1.0)
    assert [metric.value for metric in report.ndcg_at_k] == pytest.approx([1.0, 1.0, 1.0])
    assert report.mean_blocking_recall == pytest.approx(0.5)
    assert report.mean_candidate_reduction == pytest.approx(0.375)
    assert report.hard_negative_violation_rate == pytest.approx(1.0)
    assert report.abstention_accuracy == pytest.approx(1.0)
    assert report.no_reuse_abstention_rate == pytest.approx(1.0)
    assert report.reuse_non_abstention_rate == pytest.approx(1.0)


def test_offline_runner_evaluates_reuse_and_no_reuse_decisions() -> None:
    operations = (
        _operation("parse-json", "parse_json"),
        _operation("parse-yaml", "parse_yaml"),
    )
    tasks = (
        BenchmarkTask(
            task_id="known",
            query="parse_json",
            acceptable_operation_sets=(frozenset({"parse-json"}),),
        ),
        BenchmarkTask(
            task_id="unknown",
            query="transmute an unknown artifact",
            no_reuse=True,
        ),
    )
    retrieval_config = RetrievalConfig(
        config_id="exact",
        channels=(RetrievalChannelConfig(channel_id="exact", method="exact"),),
        abstention=AbstentionConfig(threshold=0.5),
    )

    report = run_offline_benchmark(
        tasks,
        retrieval_config,
        operations=operations,
        evaluation_config=EvaluationConfig(cutoffs=(1,)),
    )

    assert report.recall_at_k[0].value == pytest.approx(1.0)
    assert report.mean_reciprocal_rank == pytest.approx(1.0)
    assert report.abstention_accuracy == pytest.approx(1.0)
    assert report.task_count == 2


def test_task_and_response_validation_fails_loudly() -> None:
    with pytest.raises(ValueError, match="no-reuse"):
        BenchmarkTask(
            task_id="bad",
            query="bad",
            no_reuse=True,
            acceptable_operation_sets=(frozenset({"operation"}),),
        )

    valid_task = BenchmarkTask(task_id="valid", query="novel", no_reuse=True)
    with pytest.raises(ValueError, match="missing retrieval responses"):
        evaluate_responses((valid_task,), {})
