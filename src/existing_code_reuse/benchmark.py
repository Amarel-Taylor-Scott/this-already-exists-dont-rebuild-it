"""Offline evaluation for reuse recognition, retrieval, blocking, and abstention.

Tasks may have several independently acceptable operation sets.  Retrieval metrics therefore score
the best-covered valid set instead of pretending that one historical route is the only correct
answer.  No-reuse tasks are evaluated as abstention decisions and are excluded from relevance
metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sqlite3
from typing import Iterable, Mapping, Sequence

from .models import DerivedSignal, OperationRecord
from .retrieval import (
    OperationRetriever,
    QuerySignal,
    RetrievalConfig,
    RetrievalResponse,
)


@dataclass(frozen=True, slots=True)
class BenchmarkTask:
    """One frozen request with one or more alternative acceptable operation sets."""

    task_id: str
    query: str
    acceptable_operation_sets: tuple[frozenset[str], ...] = ()
    no_reuse: bool = False
    hard_negative_operation_ids: frozenset[str] = frozenset()
    query_signals: tuple[QuerySignal, ...] = ()

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id cannot be empty")
        if not self.query.strip():
            raise ValueError("task query cannot be empty")
        if self.no_reuse and self.acceptable_operation_sets:
            raise ValueError("a no-reuse task cannot also declare acceptable operations")
        if not self.no_reuse and not self.acceptable_operation_sets:
            raise ValueError("a reuse task needs at least one acceptable operation set")
        if any(not operation_set for operation_set in self.acceptable_operation_sets):
            raise ValueError("acceptable operation sets cannot be empty")
        acceptable_union = set().union(*self.acceptable_operation_sets)
        overlap = acceptable_union & self.hard_negative_operation_ids
        if overlap:
            raise ValueError(
                "operations cannot be both acceptable and hard negative: "
                + ", ".join(sorted(overlap))
            )


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Data configuration for deterministic offline metrics."""

    cutoffs: tuple[int, ...] = (1, 5, 10)
    hard_negative_cutoff: int = 10

    def __post_init__(self) -> None:
        if not self.cutoffs:
            raise ValueError("at least one evaluation cutoff is required")
        if any(cutoff <= 0 for cutoff in self.cutoffs):
            raise ValueError("evaluation cutoffs must be positive")
        if len(set(self.cutoffs)) != len(self.cutoffs):
            raise ValueError("evaluation cutoffs must be unique")
        if self.hard_negative_cutoff <= 0:
            raise ValueError("hard-negative cutoff must be positive")


@dataclass(frozen=True, slots=True)
class MetricAtK:
    cutoff: int
    value: float


@dataclass(frozen=True, slots=True)
class TaskEvaluation:
    task_id: str
    expected_abstention: bool
    predicted_abstention: bool
    ranked_operation_ids: tuple[str, ...]
    recall_at_k: tuple[MetricAtK, ...]
    reciprocal_rank: float
    ndcg_at_k: tuple[MetricAtK, ...]
    blocking_recall: float | None
    candidate_reduction: float
    hard_negative_violation: bool | None


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    task_count: int
    reuse_task_count: int
    no_reuse_task_count: int
    recall_at_k: tuple[MetricAtK, ...]
    mean_reciprocal_rank: float
    ndcg_at_k: tuple[MetricAtK, ...]
    mean_blocking_recall: float
    mean_candidate_reduction: float
    hard_negative_violation_rate: float
    abstention_accuracy: float
    no_reuse_abstention_rate: float
    reuse_non_abstention_rate: float
    task_evaluations: tuple[TaskEvaluation, ...]


def _best_set_recall(
    ranked_operation_ids: Sequence[str],
    acceptable_operation_sets: Sequence[frozenset[str]],
    cutoff: int | None = None,
) -> float:
    selected = set(ranked_operation_ids if cutoff is None else ranked_operation_ids[:cutoff])
    if not acceptable_operation_sets:
        return 0.0
    return max(
        len(selected & acceptable) / len(acceptable)
        for acceptable in acceptable_operation_sets
    )


def _reciprocal_rank(
    ranked_operation_ids: Sequence[str],
    acceptable_operation_sets: Sequence[frozenset[str]],
) -> float:
    relevant = set().union(*acceptable_operation_sets) if acceptable_operation_sets else set()
    for rank, operation_id in enumerate(ranked_operation_ids, start=1):
        if operation_id in relevant:
            return 1.0 / rank
    return 0.0


def _dcg(relevances: Sequence[int]) -> float:
    return sum(
        relevance / math.log2(rank + 1)
        for rank, relevance in enumerate(relevances, start=1)
    )


def _best_set_ndcg(
    ranked_operation_ids: Sequence[str],
    acceptable_operation_sets: Sequence[frozenset[str]],
    cutoff: int,
) -> float:
    if not acceptable_operation_sets:
        return 0.0
    selected = ranked_operation_ids[:cutoff]
    scores: list[float] = []
    for acceptable in acceptable_operation_sets:
        relevances = [int(operation_id in acceptable) for operation_id in selected]
        ideal_count = min(len(acceptable), cutoff)
        ideal_dcg = _dcg([1] * ideal_count)
        scores.append(_dcg(relevances) / ideal_dcg if ideal_dcg else 0.0)
    return max(scores)


def evaluate_responses(
    tasks: Sequence[BenchmarkTask],
    responses: Mapping[str, RetrievalResponse],
    *,
    config: EvaluationConfig = EvaluationConfig(),
) -> BenchmarkReport:
    """Evaluate already-produced retrieval responses against frozen benchmark tasks."""

    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("benchmark task IDs must be unique")
    missing = sorted(set(task_ids) - set(responses))
    if missing:
        raise ValueError(f"missing retrieval responses for tasks: {', '.join(missing)}")

    task_evaluations: list[TaskEvaluation] = []
    for task in tasks:
        response = responses[task.task_id]
        ranked_ids = response.operation_ids
        recall_values = (
            tuple(
                MetricAtK(
                    cutoff=cutoff,
                    value=_best_set_recall(
                        ranked_ids,
                        task.acceptable_operation_sets,
                        cutoff,
                    ),
                )
                for cutoff in config.cutoffs
            )
            if not task.no_reuse
            else ()
        )
        ndcg_values = (
            tuple(
                MetricAtK(
                    cutoff=cutoff,
                    value=_best_set_ndcg(
                        ranked_ids,
                        task.acceptable_operation_sets,
                        cutoff,
                    ),
                )
                for cutoff in config.cutoffs
            )
            if not task.no_reuse
            else ()
        )
        reciprocal_rank = (
            _reciprocal_rank(ranked_ids, task.acceptable_operation_sets)
            if not task.no_reuse
            else 0.0
        )
        blocking_recall = (
            _best_set_recall(
                response.blocking.operation_ids,
                task.acceptable_operation_sets,
            )
            if not task.no_reuse
            else None
        )
        hard_negative_violation = (
            bool(
                set(ranked_ids[: config.hard_negative_cutoff])
                & task.hard_negative_operation_ids
            )
            if task.hard_negative_operation_ids
            else None
        )
        task_evaluations.append(
            TaskEvaluation(
                task_id=task.task_id,
                expected_abstention=task.no_reuse,
                predicted_abstention=response.abstained,
                ranked_operation_ids=ranked_ids,
                recall_at_k=recall_values,
                reciprocal_rank=reciprocal_rank,
                ndcg_at_k=ndcg_values,
                blocking_recall=blocking_recall,
                candidate_reduction=response.blocking.candidate_reduction,
                hard_negative_violation=hard_negative_violation,
            )
        )

    reuse_evaluations = [
        evaluation for evaluation in task_evaluations if not evaluation.expected_abstention
    ]
    no_reuse_evaluations = [
        evaluation for evaluation in task_evaluations if evaluation.expected_abstention
    ]
    hard_negative_evaluations = [
        evaluation
        for evaluation in task_evaluations
        if evaluation.hard_negative_violation is not None
    ]

    aggregate_recall = tuple(
        MetricAtK(
            cutoff=cutoff,
            value=_mean(
                _metric_value(evaluation.recall_at_k, cutoff)
                for evaluation in reuse_evaluations
            ),
        )
        for cutoff in config.cutoffs
    )
    aggregate_ndcg = tuple(
        MetricAtK(
            cutoff=cutoff,
            value=_mean(
                _metric_value(evaluation.ndcg_at_k, cutoff)
                for evaluation in reuse_evaluations
            ),
        )
        for cutoff in config.cutoffs
    )

    return BenchmarkReport(
        task_count=len(task_evaluations),
        reuse_task_count=len(reuse_evaluations),
        no_reuse_task_count=len(no_reuse_evaluations),
        recall_at_k=aggregate_recall,
        mean_reciprocal_rank=_mean(
            evaluation.reciprocal_rank for evaluation in reuse_evaluations
        ),
        ndcg_at_k=aggregate_ndcg,
        mean_blocking_recall=_mean(
            evaluation.blocking_recall
            for evaluation in reuse_evaluations
            if evaluation.blocking_recall is not None
        ),
        mean_candidate_reduction=_mean(
            evaluation.candidate_reduction for evaluation in task_evaluations
        ),
        hard_negative_violation_rate=_mean(
            float(bool(evaluation.hard_negative_violation))
            for evaluation in hard_negative_evaluations
        ),
        abstention_accuracy=_mean(
            float(evaluation.expected_abstention == evaluation.predicted_abstention)
            for evaluation in task_evaluations
        ),
        no_reuse_abstention_rate=_mean(
            float(evaluation.predicted_abstention)
            for evaluation in no_reuse_evaluations
        ),
        reuse_non_abstention_rate=_mean(
            float(not evaluation.predicted_abstention)
            for evaluation in reuse_evaluations
        ),
        task_evaluations=tuple(task_evaluations),
    )


def run_offline_benchmark(
    tasks: Sequence[BenchmarkTask],
    retrieval_config: RetrievalConfig,
    *,
    operations: Sequence[OperationRecord] | None = None,
    derived_signals: Sequence[DerivedSignal] = (),
    sqlite_source: sqlite3.Connection | str | Path | None = None,
    evaluation_config: EvaluationConfig = EvaluationConfig(),
) -> BenchmarkReport:
    """Run one retrieval configuration over all tasks and evaluate it offline."""

    retriever = OperationRetriever(
        operations,
        derived_signals=derived_signals,
        sqlite_source=sqlite_source,
    )
    responses = {
        task.task_id: retriever.search(
            task.query,
            config=retrieval_config,
            query_signals=task.query_signals,
        )
        for task in tasks
    }
    return evaluate_responses(tasks, responses, config=evaluation_config)


def _metric_value(metrics: Sequence[MetricAtK], cutoff: int) -> float:
    for metric in metrics:
        if metric.cutoff == cutoff:
            return metric.value
    raise ValueError(f"metric does not contain cutoff {cutoff}")


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0


__all__ = [
    "BenchmarkReport",
    "BenchmarkTask",
    "EvaluationConfig",
    "MetricAtK",
    "TaskEvaluation",
    "evaluate_responses",
    "run_offline_benchmark",
]
