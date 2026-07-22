"""Command-line entry points for the executable research seed."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from importlib import metadata as importlib_metadata
from importlib.resources import as_file, files
import json
from pathlib import Path
import platform
import sqlite3
import sys
from typing import Any, Sequence

from .benchmark import BenchmarkTask, EvaluationConfig, run_offline_benchmark
from .capabilities import seed_capabilities
from .execution import execute_route, plan_route, receipt_json
from .experiments import (
    design_configurations,
    design_manifest,
    load_experiment_space,
    schedule_first_round,
)
from .ingest import IngestConfig, ingest_installed_distributions
from .models import normalize_project_name
from .retrieval import (
    OperationRetriever,
    QuerySignal,
    RetrievalConfig,
    load_retrieval_profiles,
)
from .seed import load_jsonl, load_seed_execution_cases, project_verified_capabilities_to_search
from .storage import read_catalog, write_catalog_batches


def _json_print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def _sqlite_has_fts5() -> bool:
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("CREATE VIRTUAL TABLE test_fts USING fts5(text)")
        return True
    except sqlite3.OperationalError:
        return False
    finally:
        connection.close()


def _optional_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _doctor(_: argparse.Namespace) -> int:
    distributions = {
        (item.metadata.get("Name") or "").strip()
        for item in importlib_metadata.distributions()
        if (item.metadata.get("Name") or "").strip()
    }
    _json_print(
        {
            "status": "ok",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "sqlite": sqlite3.sqlite_version,
            "sqlite_fts5": _sqlite_has_fts5(),
            "installed_distribution_count": len(distributions),
            "optional_dependencies": {
                "pandas": _optional_version("pandas"),
                "scikit-learn": _optional_version("scikit-learn"),
            },
            "ingestion_policy": {
                "imports_target_packages": False,
                "executes_target_packages": False,
                "builds_source_distributions": False,
            },
        }
    )
    return 0


def _ingest_installed(args: argparse.Namespace) -> int:
    requested_packages = sorted(
        {normalize_project_name(name) for name in args.package if name.strip()}
    )
    config = IngestConfig(
        max_files_per_distribution=args.max_files,
        max_bytes_per_file=args.max_file_bytes,
        max_total_bytes_per_distribution=args.max_distribution_bytes,
        include_private=args.include_private,
    )
    output = Path(args.output)
    write_catalog_batches(
        output,
        (
            ingest_installed_distributions((package,), config=config)
            for package in requested_packages
        ),
    )
    connection = sqlite3.connect(output)
    try:
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        counts = {
            name: int(connection.execute(f"SELECT count(*) FROM {name}").fetchone()[0])
            for name in (
                "packages",
                "operations",
                "representations",
                "signals",
                "dependencies",
                "errors",
            )
        }
        package_versions = dict(
            connection.execute(
                "SELECT normalized_name, version FROM packages ORDER BY normalized_name"
            ).fetchall()
        )
        error_types = Counter(
            str(json.loads(row[0]).get("error_type", "unknown"))
            for row in connection.execute("SELECT error_json FROM errors")
        )
    finally:
        connection.close()
    result = {
        "status": "ok" if integrity == "ok" else "failed_integrity",
        "requested_packages": requested_packages,
        "counts": counts,
        "package_versions": package_versions,
        "error_types": dict(sorted(error_types.items())),
        "database": str(output),
        "database_bytes": output.stat().st_size,
        "sqlite_integrity": integrity,
        "next_command": f"reuse-code search --database {output} --profile fts --query 'parse a version requirement'",
    }
    _json_print(result)
    return 0 if integrity == "ok" else 1


def _retrieval_config(
    profile: str,
    *,
    result_limit: int,
    use_blocking: bool = False,
    profile_file: str | Path | None = None,
) -> RetrievalConfig:
    if profile_file is not None:
        profiles = load_retrieval_profiles(profile_file)
    else:
        checkout_profile_file = Path("configs/retrieval_profiles.json")
        source_profile_file = (
            Path(__file__).resolve().parents[2] / "configs/retrieval_profiles.json"
        )
        if checkout_profile_file.is_file():
            profiles = load_retrieval_profiles(checkout_profile_file)
        elif source_profile_file.is_file():
            profiles = load_retrieval_profiles(source_profile_file)
        else:
            packaged_profile_file = files("existing_code_reuse").joinpath(
                "retrieval_profiles.json"
            )
            with as_file(packaged_profile_file) as materialized_profile_file:
                profiles = load_retrieval_profiles(materialized_profile_file)
    if profile not in profiles:
        available = ", ".join(sorted(profiles))
        raise ValueError(f"unknown retrieval profile {profile!r}; available: {available}")
    base = profiles[profile]
    return replace(
        base,
        config_id=f"{profile}_{'blocked' if use_blocking else 'unblocked'}_v1",
        result_limit=result_limit,
        channels=tuple(
            replace(channel, candidate_limit=max(channel.candidate_limit, result_limit))
            for channel in base.channels
        ),
        blocking=replace(base.blocking, enabled=use_blocking),
    )


def _search(args: argparse.Namespace) -> int:
    if args.database:
        if args.block:
            catalog = read_catalog(args.database)
            retriever = OperationRetriever(
                catalog.operations,
                derived_signals=catalog.signals,
                sqlite_source=args.database,
            )
        else:
            retriever = OperationRetriever(sqlite_source=args.database)
    else:
        operations, signals = project_verified_capabilities_to_search()
        retriever = OperationRetriever(operations, derived_signals=signals)
    query_signals = tuple(
        QuerySignal(namespace=namespace, value=value)
        for namespace, value in (item.split("=", 1) for item in args.signal)
    )
    response = retriever.search(
        args.query,
        config=_retrieval_config(
            args.profile,
            result_limit=args.limit,
            use_blocking=args.block,
            profile_file=args.profile_file,
        ),
        query_signals=query_signals,
    )
    _json_print(
        {
            "query": response.query,
            "config_id": response.config_id,
            "abstained": response.abstained,
            "confidence": response.confidence,
            "candidate_count": response.blocking.candidate_count,
            "candidate_reduction": response.blocking.candidate_reduction,
            "hits": [
                {
                    "rank": hit.rank,
                    "operation_id": hit.operation_id,
                    "package": hit.operation.package_name,
                    "version": hit.operation.package_version,
                    "module": hit.operation.module,
                    "qualified_name": hit.operation.qualified_name,
                    "score": hit.score,
                    "channels": [asdict(item) for item in hit.provenance],
                }
                for hit in response.hits
            ],
        }
    )
    return 0


def _load_benchmark_tasks(path: Path) -> tuple[BenchmarkTask, ...]:
    tasks: list[BenchmarkTask] = []
    for item in load_jsonl(path):
        tasks.append(
            BenchmarkTask(
                task_id=item["task_id"],
                query=item["query"],
                acceptable_operation_sets=tuple(
                    frozenset(operation_set) for operation_set in item["acceptable_sets"]
                ),
                no_reuse=not bool(item["reuse_expected"]),
                hard_negative_operation_ids=frozenset(item.get("hard_negatives", ())),
                query_signals=tuple(
                    QuerySignal(
                        namespace=signal["namespace"],
                        value=signal["value"],
                        signal_kind=signal.get("signal_kind", "blocking_key"),
                    )
                    for signal in item.get("query_signals", ())
                ),
            )
        )
    return tuple(tasks)


def _benchmark_retrieval(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    operations, signals = project_verified_capabilities_to_search()
    tasks = _load_benchmark_tasks(project_root / "data/seed/retrieval_tasks.jsonl")
    requested_profiles = args.profile or ["exact", "word", "char", "fts", "hybrid"]
    results: dict[str, Any] = {}
    for profile in requested_profiles:
        for blocked in ((False, True) if profile == "hybrid" else (False,)):
            config = _retrieval_config(
                profile,
                result_limit=10,
                use_blocking=blocked,
                profile_file=args.profile_file,
            )
            report = run_offline_benchmark(
                tasks,
                config,
                operations=operations,
                derived_signals=signals,
                evaluation_config=EvaluationConfig(
                    cutoffs=(1, 3, 5),
                    hard_negative_cutoff=1,
                ),
            )
            results[config.config_id] = asdict(report)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1",
        "warning": "The six-task seed is a software smoke test, not a statistical result.",
        "catalog_operation_count": len(operations),
        "task_count": len(tasks),
        "profiles": results,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _json_print({**payload, "output": str(output)})
    return 0


def _run_seed(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_seed_execution_cases(
        project_root / "data/seed/execution_tasks.json",
        project_root=project_root,
        output_root=project_root,
    )
    case_results: list[dict[str, Any]] = []
    capabilities = seed_capabilities()
    for case in cases:
        planned = plan_route(
            case.task.input_artifact_type,
            case.task.required_output_type,
            capabilities,
            required_stages=case.task.required_stages,
            allow_residual=case.task.allow_residual,
        )
        if case.track == "no_reuse":
            decision_pass = planned.status == case.expected_decision
            case_results.append(
                {
                    "task_id": case.task.task_id,
                    "track": case.track,
                    "decision": planned.status,
                    "reason": planned.reason,
                    "passed": decision_pass,
                }
            )
            continue

        selected_route = case.candidate_route or case.route
        receipt = execute_route(case.task, selected_route)
        receipt_path = output_dir / f"{case.task.task_id}.receipt.json"
        receipt_path.write_text(receipt_json(receipt), encoding="utf-8")
        expected = case.expected_decision
        decision_pass = receipt["status"] == expected if expected else bool(receipt["accepted"])
        case_results.append(
            {
                "task_id": case.task.task_id,
                "track": case.track,
                "planner_status": planned.status,
                "execution_status": receipt["status"],
                "accepted": receipt["accepted"],
                "residual_authored_lines": receipt.get("residual_authored_lines", 0),
                "receipt": str(receipt_path),
                "passed": decision_pass,
            }
        )

    summary = {
        "schema_version": "1",
        "task_count": len(case_results),
        "passed_task_count": sum(bool(item["passed"]) for item in case_results),
        "all_passed": all(bool(item["passed"]) for item in case_results),
        "model_calls": 0,
        "cases": case_results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _json_print({**summary, "summary": str(summary_path)})
    return 0 if summary["all_passed"] else 1


def _design_experiments(args: argparse.Namespace) -> int:
    """Create a bounded conditional sweep manifest without claiming measured results."""

    if args.space:
        space = load_experiment_space(args.space)
    else:
        checkout_space = Path("configs/experiment_space.json")
        source_space = Path(__file__).resolve().parents[2] / "configs/experiment_space.json"
        if checkout_space.is_file():
            space = load_experiment_space(checkout_space)
        elif source_space.is_file():
            space = load_experiment_space(source_space)
        else:
            packaged_space = files("existing_code_reuse").joinpath("experiment_space.json")
            with as_file(packaged_space) as materialized_space:
                space = load_experiment_space(materialized_space)
    configurations = design_configurations(
        space,
        strategy=args.strategy,
        max_experiments=args.max_experiments,
        seed=args.seed,
        max_resource_tier=args.max_resource_tier,
    )
    if not configurations:
        raise ValueError("the selected strategy/resource tier produced no valid configurations")
    if args.budget not in space.budget_by_id:
        available = ", ".join(sorted(space.budget_by_id))
        raise ValueError(f"unknown budget {args.budget!r}; available: {available}")
    trials = schedule_first_round(
        configurations,
        budget=space.budget_by_id[args.budget],
        registry_digest=space.registry_digest,
    )
    manifest = design_manifest(
        space,
        configurations,
        trials,
        strategy=args.strategy,
        seed=args.seed,
        max_resource_tier=args.max_resource_tier,
        requested_configuration_count=args.max_experiments,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _json_print(
        {
            "status": "scheduled",
            "output": str(output),
            "registry_digest": space.registry_digest,
            "dimension_count": len(space.dimensions),
            "constraint_count": len(space.constraints),
            "raw_cartesian_size": space.raw_cartesian_size,
            "requested_configuration_count": args.max_experiments,
            "scheduled_configuration_count": len(configurations),
            "underfilled": len(configurations) < args.max_experiments,
            "budget": args.budget,
            "warning": manifest["warning"],
        }
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reuse-code",
        description="Find and validate existing Python capabilities before generating replacements.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Report local feature and dependency availability.")
    doctor.set_defaults(handler=_doctor)

    ingest = subparsers.add_parser(
        "ingest-installed",
        help="Statically catalog explicitly named installed distributions without importing them.",
    )
    ingest.add_argument("--package", action="append", required=True)
    ingest.add_argument("--output", default="data/generated/catalog.sqlite")
    ingest.add_argument("--max-files", type=int, default=1_000)
    ingest.add_argument("--max-file-bytes", type=int, default=2_000_000)
    ingest.add_argument("--max-distribution-bytes", type=int, default=50_000_000)
    ingest.add_argument("--include-private", action="store_true")
    ingest.set_defaults(handler=_ingest_installed)

    search = subparsers.add_parser("search", help="Search the verified seed or a generated catalog.")
    search.add_argument("--query", required=True)
    search.add_argument("--database")
    search.add_argument("--profile", choices=("exact", "word", "char", "fts", "hybrid"), default="hybrid")
    search.add_argument("--limit", type=int, default=10)
    search.add_argument(
        "--profile-file",
        help="Override the bundled versioned retrieval-profile JSON document.",
    )
    search.add_argument("--block", action="store_true")
    search.add_argument(
        "--signal",
        action="append",
        default=[],
        metavar="NAMESPACE=VALUE",
        help="Add an explicit deterministic query facet.",
    )
    search.set_defaults(handler=_search)

    benchmark = subparsers.add_parser(
        "benchmark-retrieval", help="Run the small offline retrieval smoke benchmark."
    )
    benchmark.add_argument("--project-root", default=".")
    benchmark.add_argument(
        "--profile-file",
        help="Override the bundled versioned retrieval-profile JSON document.",
    )
    benchmark.add_argument("--profile", action="append", choices=("exact", "word", "char", "fts", "hybrid"))
    benchmark.add_argument("--output", default="reports/live/retrieval-benchmark.json")
    benchmark.set_defaults(handler=_benchmark_retrieval)

    run_seed = subparsers.add_parser(
        "run-seed", help="Execute the five direct/composed/negative/no-reuse/residual seed cases."
    )
    run_seed.add_argument("--project-root", default=".")
    run_seed.add_argument("--output-dir", default="reports/live/receipts")
    run_seed.set_defaults(handler=_run_seed)

    design = subparsers.add_parser(
        "design-experiments",
        help="Generate a bounded conditional retrieval/model sweep manifest.",
    )
    design.add_argument(
        "--space",
        help="Override the bundled versioned conditional experiment-space JSON document.",
    )
    design.add_argument(
        "--strategy",
        choices=(
            "baseline",
            "one_factor",
            "pairwise_screen",
            "random_valid",
            "mixed_screen",
            "full_factorial",
        ),
        default="mixed_screen",
    )
    design.add_argument("--max-experiments", type=int, default=200)
    design.add_argument("--seed", type=int, default=17)
    design.add_argument(
        "--max-resource-tier",
        default="t4_full",
        help="Maximum tier declared by the selected registry (default: t4_full).",
    )
    design.add_argument(
        "--budget",
        default="smoke",
        help="Budget ID declared by the selected registry (default: smoke).",
    )
    design.add_argument("--output", default="reports/live/experiment-design.json")
    design.set_defaults(handler=_design_experiments)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    for signal in getattr(args, "signal", ()):
        if "=" not in signal:
            parser.error("--signal must use NAMESPACE=VALUE")
    return int(args.handler(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
