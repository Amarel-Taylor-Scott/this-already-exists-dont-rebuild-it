"""Conditional experiment-space design for retrieval and small-model studies.

The package catalog is evidence.  This module does not alter that evidence and does not pretend
that declaring an experiment option means the corresponding dependency or model is available.  It
only validates, fingerprints, and schedules configurations from a versioned JSON registry.

Large retrieval studies are conditional spaces, not safe Cartesian products.  For example, an
HNSW index is meaningless when no dense representation is selected, an Ollama runtime requires a
model with an Ollama tag, and a Matryoshka truncation is valid only for models trained for it.  The
constraint records in the registry keep those relationships explicit and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
from typing import Iterable, Iterator, Mapping, Sequence


EXPERIMENT_STAGE_ORDER = (
    "corpus",
    "representation",
    "query",
    "blocking",
    "embedding",
    "index",
    "retrieval",
    "rerank",
    "compatibility",
    "planning",
    "verification",
)


class ExperimentSpaceError(ValueError):
    """Raised when an experiment registry or selected configuration is invalid."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ExperimentSpaceError(f"{label} must be an object")
    return value


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ExperimentSpaceError(f"{label} must be an array")
    return value


@dataclass(frozen=True, slots=True)
class ExperimentOption:
    """One selectable value on an experiment dimension."""

    option_id: str
    resource_tier: str = "cpu_small"
    enabled: bool = True
    estimated_vram_gb: float | None = None
    tags: tuple[str, ...] = ()
    metadata: tuple[tuple[str, object], ...] = ()

    def metadata_dict(self) -> dict[str, object]:
        return dict(self.metadata)


@dataclass(frozen=True, slots=True)
class ExperimentDimension:
    """A factor that selects one option; ``stage`` is its earliest affected artifact stage."""

    dimension_id: str
    stage: str
    description: str
    options: tuple[ExperimentOption, ...]
    baseline: str

    def __post_init__(self) -> None:
        if not self.dimension_id.strip():
            raise ExperimentSpaceError("dimension_id cannot be empty")
        option_ids = [option.option_id for option in self.options]
        if not option_ids or len(option_ids) != len(set(option_ids)):
            raise ExperimentSpaceError(
                f"dimension {self.dimension_id!r} must have unique options"
            )
        if self.baseline not in option_ids:
            raise ExperimentSpaceError(
                f"baseline {self.baseline!r} is not an option for {self.dimension_id!r}"
            )

    def option(self, option_id: str) -> ExperimentOption:
        for option in self.options:
            if option.option_id == option_id:
                return option
        raise ExperimentSpaceError(
            f"unknown option {option_id!r} for dimension {self.dimension_id!r}"
        )


@dataclass(frozen=True, slots=True)
class ConditionalConstraint:
    """If every ``when`` selection matches, all ``require`` selections must match."""

    constraint_id: str
    when: tuple[tuple[str, tuple[str, ...]], ...]
    require: tuple[tuple[str, tuple[str, ...]], ...]
    reason: str


@dataclass(frozen=True, slots=True)
class Budget:
    """A reproducible amount of benchmark work for one sweep round."""

    budget_id: str
    task_limit: int
    package_limit: int
    query_repeats: int
    learned_seeds: int
    route_execution_limit: int


@dataclass(frozen=True, slots=True)
class ExperimentConfiguration:
    """A complete selection across every registered dimension."""

    selections: tuple[tuple[str, str], ...]
    config_id: str
    source: str

    @classmethod
    def build(
        cls,
        selections: Mapping[str, str],
        *,
        source: str,
    ) -> "ExperimentConfiguration":
        ordered = tuple(sorted((str(key), str(value)) for key, value in selections.items()))
        return cls(
            selections=ordered,
            config_id="exp-" + _digest(ordered).split(":", 1)[1][:16],
            source=source,
        )

    def as_dict(self) -> dict[str, str]:
        return dict(self.selections)


@dataclass(frozen=True, slots=True)
class TrialRecord:
    """One scheduled experiment at one resource budget."""

    trial_id: str
    configuration: ExperimentConfiguration
    budget: Budget
    round_index: int
    status: str = "scheduled"


@dataclass(frozen=True, slots=True)
class ExperimentSpace:
    """Versioned dimensions, constraints, budgets, and named reference configurations."""

    schema_version: str
    dimensions: tuple[ExperimentDimension, ...]
    constraints: tuple[ConditionalConstraint, ...]
    budgets: tuple[Budget, ...]
    named_configurations: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
    resource_tiers: tuple[str, ...]
    runner_coverage: tuple[tuple[str, object], ...]
    registry_digest: str

    def __post_init__(self) -> None:
        if len(self.resource_tiers) != len(set(self.resource_tiers)):
            raise ExperimentSpaceError("resource tiers must be unique")
        dimension_ids = [item.dimension_id for item in self.dimensions]
        if len(dimension_ids) != len(set(dimension_ids)):
            raise ExperimentSpaceError("dimension IDs must be unique")
        if not self.resource_tiers:
            raise ExperimentSpaceError("at least one resource tier is required")
        for dimension in self.dimensions:
            if dimension.stage not in EXPERIMENT_STAGE_ORDER:
                raise ExperimentSpaceError(
                    f"dimension {dimension.dimension_id!r} uses unknown stage "
                    f"{dimension.stage!r}"
                )
            for option in dimension.options:
                if option.resource_tier not in self.resource_tiers:
                    raise ExperimentSpaceError(
                        f"option {dimension.dimension_id}={option.option_id} uses unknown "
                        f"resource tier {option.resource_tier!r}"
                    )
                if option.estimated_vram_gb is not None and option.estimated_vram_gb < 0:
                    raise ExperimentSpaceError(
                        f"option {dimension.dimension_id}={option.option_id} has negative VRAM"
                    )
        known_dimensions = set(dimension_ids)
        constraint_ids = [item.constraint_id for item in self.constraints]
        if len(constraint_ids) != len(set(constraint_ids)):
            raise ExperimentSpaceError("constraint IDs must be unique")
        options_by_dimension = {
            item.dimension_id: {option.option_id for option in item.options}
            for item in self.dimensions
        }
        for constraint in self.constraints:
            referenced = {item[0] for item in constraint.when + constraint.require}
            unknown = referenced - known_dimensions
            if unknown:
                raise ExperimentSpaceError(
                    f"constraint {constraint.constraint_id!r} references unknown dimensions: "
                    + ", ".join(sorted(unknown))
                )
            for dimension_id, values in constraint.when + constraint.require:
                unknown_options = set(values) - options_by_dimension[dimension_id]
                if unknown_options:
                    raise ExperimentSpaceError(
                        f"constraint {constraint.constraint_id!r} references unknown options "
                        f"for {dimension_id}: " + ", ".join(sorted(unknown_options))
                    )
        budget_ids = [item.budget_id for item in self.budgets]
        if not budget_ids or len(budget_ids) != len(set(budget_ids)):
            raise ExperimentSpaceError("budget IDs must be present and unique")
        for budget in self.budgets:
            if min(
                budget.task_limit,
                budget.package_limit,
                budget.query_repeats,
                budget.learned_seeds,
            ) <= 0 or budget.route_execution_limit < 0:
                raise ExperimentSpaceError(
                    f"budget {budget.budget_id!r} has an invalid negative or zero limit"
                )
        configuration_names = [name for name, _ in self.named_configurations]
        if len(configuration_names) != len(set(configuration_names)):
            raise ExperimentSpaceError("named configuration names must be unique")

    @property
    def dimension_by_id(self) -> dict[str, ExperimentDimension]:
        return {item.dimension_id: item for item in self.dimensions}

    @property
    def budget_by_id(self) -> dict[str, Budget]:
        return {item.budget_id: item for item in self.budgets}

    @property
    def baseline(self) -> dict[str, str]:
        return {item.dimension_id: item.baseline for item in self.dimensions}

    @property
    def raw_cartesian_size(self) -> int:
        return math.prod(sum(option.enabled for option in item.options) for item in self.dimensions)

    def named_configuration(self, name: str) -> dict[str, str]:
        for config_name, selections in self.named_configurations:
            if config_name == name:
                result = self.baseline
                result.update(dict(selections))
                return result
        raise ExperimentSpaceError(f"unknown named configuration: {name}")

    def resource_rank(self, tier: str) -> int:
        try:
            return self.resource_tiers.index(tier)
        except ValueError as error:
            raise ExperimentSpaceError(f"unknown resource tier: {tier}") from error

    def validate(
        self,
        selections: Mapping[str, str],
        *,
        allow_disabled: bool = False,
        partial: bool = False,
        max_resource_tier: str | None = None,
    ) -> tuple[str, ...]:
        """Return all validation failures for a complete or partial selection."""

        failures: list[str] = []
        dimensions = self.dimension_by_id
        unknown_dimensions = set(selections) - set(dimensions)
        if unknown_dimensions:
            failures.append("unknown dimensions: " + ", ".join(sorted(unknown_dimensions)))
        if not partial:
            missing = set(dimensions) - set(selections)
            if missing:
                failures.append("missing dimensions: " + ", ".join(sorted(missing)))
        maximum_rank = (
            self.resource_rank(max_resource_tier) if max_resource_tier is not None else None
        )
        for dimension_id, option_id in selections.items():
            if dimension_id not in dimensions:
                continue
            try:
                option = dimensions[dimension_id].option(option_id)
            except ExperimentSpaceError as error:
                failures.append(str(error))
                continue
            if not option.enabled and not allow_disabled:
                failures.append(f"{dimension_id}={option_id} is disabled")
            if maximum_rank is not None and self.resource_rank(option.resource_tier) > maximum_rank:
                failures.append(
                    f"{dimension_id}={option_id} requires {option.resource_tier}, above "
                    f"{max_resource_tier}"
                )

        for constraint in self.constraints:
            applies = True
            for dimension_id, allowed_values in constraint.when:
                if dimension_id not in selections:
                    applies = False
                    break
                if selections[dimension_id] not in allowed_values:
                    applies = False
                    break
            if not applies:
                continue
            for dimension_id, allowed_values in constraint.require:
                if dimension_id not in selections:
                    if partial:
                        continue
                    failures.append(
                        f"{constraint.constraint_id}: missing {dimension_id}; {constraint.reason}"
                    )
                elif selections[dimension_id] not in allowed_values:
                    failures.append(
                        f"{constraint.constraint_id}: {dimension_id}={selections[dimension_id]} "
                        f"must be one of {sorted(allowed_values)}; {constraint.reason}"
                    )
        return tuple(failures)

    def option_metadata(self, dimension_id: str, option_id: str) -> dict[str, object]:
        return self.dimension_by_id[dimension_id].option(option_id).metadata_dict()


def _parse_selector(value: object, label: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    mapping = _mapping(value, label)
    result: list[tuple[str, tuple[str, ...]]] = []
    for dimension_id, raw_allowed in mapping.items():
        allowed = tuple(str(item) for item in _sequence(raw_allowed, f"{label}.{dimension_id}"))
        if not allowed:
            raise ExperimentSpaceError(f"{label}.{dimension_id} cannot be empty")
        result.append((str(dimension_id), allowed))
    return tuple(sorted(result))


def experiment_space_from_mapping(document: Mapping[str, object]) -> ExperimentSpace:
    """Parse and validate an experiment-space JSON document."""

    if str(document.get("schema_version")) != "1":
        raise ExperimentSpaceError("experiment space must use schema_version 1")
    resource_tiers = tuple(
        str(item) for item in _sequence(document.get("resource_tiers", ()), "resource_tiers")
    )
    dimensions: list[ExperimentDimension] = []
    for index, raw_dimension in enumerate(
        _sequence(document.get("dimensions", ()), "dimensions")
    ):
        dimension = _mapping(raw_dimension, f"dimensions[{index}]")
        options: list[ExperimentOption] = []
        for option_index, raw_option in enumerate(
            _sequence(dimension.get("options", ()), f"dimensions[{index}].options")
        ):
            option = _mapping(raw_option, f"dimensions[{index}].options[{option_index}]")
            reserved = {
                "option_id",
                "resource_tier",
                "enabled",
                "estimated_vram_gb",
                "tags",
            }
            metadata = tuple(
                sorted((str(key), value) for key, value in option.items() if key not in reserved)
            )
            options.append(
                ExperimentOption(
                    option_id=str(option["option_id"]),
                    resource_tier=str(option.get("resource_tier", "cpu_small")),
                    enabled=bool(option.get("enabled", True)),
                    estimated_vram_gb=(
                        float(option["estimated_vram_gb"])
                        if option.get("estimated_vram_gb") is not None
                        else None
                    ),
                    tags=tuple(
                        str(item)
                        for item in _sequence(option.get("tags", ()), "option tags")
                    ),
                    metadata=metadata,
                )
            )
        dimensions.append(
            ExperimentDimension(
                dimension_id=str(dimension["dimension_id"]),
                stage=str(dimension["stage"]),
                description=str(dimension.get("description", "")),
                options=tuple(options),
                baseline=str(dimension["baseline"]),
            )
        )

    constraints: list[ConditionalConstraint] = []
    for index, raw_constraint in enumerate(
        _sequence(document.get("constraints", ()), "constraints")
    ):
        constraint = _mapping(raw_constraint, f"constraints[{index}]")
        constraints.append(
            ConditionalConstraint(
                constraint_id=str(constraint["constraint_id"]),
                when=_parse_selector(constraint.get("when", {}), "when"),
                require=_parse_selector(constraint.get("require", {}), "require"),
                reason=str(constraint.get("reason", "conditional requirement")),
            )
        )

    budgets: list[Budget] = []
    for index, raw_budget in enumerate(_sequence(document.get("budgets", ()), "budgets")):
        budget = _mapping(raw_budget, f"budgets[{index}]")
        budgets.append(
            Budget(
                budget_id=str(budget["budget_id"]),
                task_limit=int(budget["task_limit"]),
                package_limit=int(budget["package_limit"]),
                query_repeats=int(budget.get("query_repeats", 1)),
                learned_seeds=int(budget.get("learned_seeds", 1)),
                route_execution_limit=int(budget.get("route_execution_limit", 0)),
            )
        )

    named: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    named_document = _mapping(document.get("named_configurations", {}), "named_configurations")
    for name, raw_selections in named_document.items():
        selections = _mapping(raw_selections, f"named_configurations.{name}")
        named.append(
            (
                str(name),
                tuple(sorted((str(key), str(value)) for key, value in selections.items())),
            )
        )

    canonical_document = json.loads(_canonical_json(document))
    runner_coverage = _mapping(document.get("runner_coverage", {}), "runner_coverage")
    space = ExperimentSpace(
        schema_version="1",
        dimensions=tuple(dimensions),
        constraints=tuple(constraints),
        budgets=tuple(budgets),
        named_configurations=tuple(sorted(named)),
        resource_tiers=resource_tiers,
        runner_coverage=tuple(sorted((str(key), value) for key, value in runner_coverage.items())),
        registry_digest=_digest(canonical_document),
    )
    baseline_failures = space.validate(space.baseline)
    if baseline_failures:
        raise ExperimentSpaceError(
            "invalid baseline configuration: " + "; ".join(baseline_failures)
        )
    for name, _ in space.named_configurations:
        failures = space.validate(space.named_configuration(name))
        if failures:
            raise ExperimentSpaceError(
                f"invalid named configuration {name!r}: " + "; ".join(failures)
            )
    return space


def load_experiment_space(path: str | Path) -> ExperimentSpace:
    return experiment_space_from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))


def _deduplicate(
    configurations: Iterable[ExperimentConfiguration],
) -> list[ExperimentConfiguration]:
    result: dict[str, ExperimentConfiguration] = {}
    for configuration in configurations:
        result.setdefault(configuration.config_id, configuration)
    return list(result.values())


def _valid_configuration(
    space: ExperimentSpace,
    selections: Mapping[str, str],
    *,
    source: str,
    max_resource_tier: str | None,
) -> ExperimentConfiguration | None:
    if space.validate(selections, max_resource_tier=max_resource_tier):
        return None
    return ExperimentConfiguration.build(selections, source=source)


def _eligible_options(
    space: ExperimentSpace,
    dimension: ExperimentDimension,
    max_resource_tier: str | None,
) -> list[ExperimentOption]:
    maximum_rank = (
        space.resource_rank(max_resource_tier) if max_resource_tier is not None else None
    )
    options = [
        option
        for option in dimension.options
        if option.enabled
        and (
            maximum_rank is None
            or space.resource_rank(option.resource_tier) <= maximum_rank
        )
    ]
    options.sort(key=lambda option: option.option_id != dimension.baseline)
    if not options:
        raise ExperimentSpaceError(
            f"dimension {dimension.dimension_id!r} has no enabled option at or below "
            f"resource tier {max_resource_tier!r}"
        )
    return options


def baseline_and_named_design(
    space: ExperimentSpace,
    *,
    max_resource_tier: str | None = None,
) -> list[ExperimentConfiguration]:
    candidates = [
        _valid_configuration(
            space,
            space.baseline,
            source="baseline",
            max_resource_tier=max_resource_tier,
        )
    ]
    for name, _ in space.named_configurations:
        candidates.append(
            _valid_configuration(
                space,
                space.named_configuration(name),
                source=f"named:{name}",
                max_resource_tier=max_resource_tier,
            )
        )
    return _deduplicate(item for item in candidates if item is not None)


def one_factor_design(
    space: ExperimentSpace,
    *,
    max_resource_tier: str | None = None,
) -> list[ExperimentConfiguration]:
    """Change one factor at a time from the registry baseline."""

    candidates = baseline_and_named_design(space, max_resource_tier=max_resource_tier)
    baseline = space.baseline
    for dimension in space.dimensions:
        for option in dimension.options:
            if option.option_id == dimension.baseline:
                continue
            selections = dict(baseline)
            selections[dimension.dimension_id] = option.option_id
            candidate = _valid_configuration(
                space,
                selections,
                source=f"one_factor:{dimension.dimension_id}",
                max_resource_tier=max_resource_tier,
            )
            if candidate is not None:
                candidates.append(candidate)
    return _deduplicate(candidates)


def pairwise_screen_design(
    space: ExperimentSpace,
    *,
    max_experiments: int,
    seed: int,
    max_resource_tier: str | None = None,
) -> list[ExperimentConfiguration]:
    """Exercise pairs of changed factors in a fixed baseline context.

    This is deliberately called a *screen*, not a covering array.  Conditional spaces can make a
    nominal pair valid only when a third factor changes.  Random conditional trials supplement this
    deterministic screen in :func:`mixed_screen_design`.
    """

    candidates = one_factor_design(space, max_resource_tier=max_resource_tier)
    baseline = space.baseline
    for left_index, left in enumerate(space.dimensions):
        for right in space.dimensions[left_index + 1 :]:
            for left_option in left.options:
                for right_option in right.options:
                    if (
                        left_option.option_id == left.baseline
                        or right_option.option_id == right.baseline
                    ):
                        continue
                    selections = dict(baseline)
                    selections[left.dimension_id] = left_option.option_id
                    selections[right.dimension_id] = right_option.option_id
                    candidate = _valid_configuration(
                        space,
                        selections,
                        source=f"pairwise_screen:{left.dimension_id}+{right.dimension_id}",
                        max_resource_tier=max_resource_tier,
                    )
                    if candidate is not None:
                        candidates.append(candidate)
    candidates = _deduplicate(candidates)
    if len(candidates) <= max_experiments:
        return candidates
    protected = baseline_and_named_design(space, max_resource_tier=max_resource_tier)
    protected_ids = {item.config_id for item in protected}
    remainder = [item for item in candidates if item.config_id not in protected_ids]
    random.Random(seed).shuffle(remainder)
    return (protected + remainder[: max(0, max_experiments - len(protected))])[
        :max_experiments
    ]


def random_valid_design(
    space: ExperimentSpace,
    *,
    max_experiments: int,
    seed: int,
    max_resource_tier: str | None = None,
    attempts_per_experiment: int = 200,
) -> list[ExperimentConfiguration]:
    """Draw deterministic random configurations and retain only conditionally valid ones."""

    if max_experiments <= 0:
        return []
    rng = random.Random(seed)
    candidates = baseline_and_named_design(
        space, max_resource_tier=max_resource_tier
    )[:max_experiments]
    seen = {item.config_id for item in candidates}
    eligible_by_dimension = {
        dimension.dimension_id: _eligible_options(space, dimension, max_resource_tier)
        for dimension in space.dimensions
    }
    attempts = 0
    maximum_attempts = max_experiments * attempts_per_experiment
    while len(candidates) < max_experiments and attempts < maximum_attempts:
        attempts += 1
        if attempts % 2 and candidates:
            selections = rng.choice(candidates).as_dict()
            mutation_count = rng.randint(1, min(8, len(space.dimensions)))
            for dimension in rng.sample(list(space.dimensions), mutation_count):
                selections[dimension.dimension_id] = rng.choice(
                    eligible_by_dimension[dimension.dimension_id]
                ).option_id
        else:
            selections = {
                dimension.dimension_id: rng.choice(
                    eligible_by_dimension[dimension.dimension_id]
                ).option_id
                for dimension in space.dimensions
            }
        candidate = _valid_configuration(
            space,
            selections,
            source="random_valid",
            max_resource_tier=max_resource_tier,
        )
        if candidate is not None and candidate.config_id not in seen:
            seen.add(candidate.config_id)
            candidates.append(candidate)
    return candidates


def mixed_screen_design(
    space: ExperimentSpace,
    *,
    max_experiments: int,
    seed: int,
    max_resource_tier: str | None = None,
) -> list[ExperimentConfiguration]:
    """Combine baseline, one-factor, conditional pair screening, and random interaction trials."""

    pair_budget = max(1, math.ceil(max_experiments * 0.7))
    candidates = pairwise_screen_design(
        space,
        max_experiments=pair_budget,
        seed=seed,
        max_resource_tier=max_resource_tier,
    )
    random_candidates = random_valid_design(
        space,
        max_experiments=max_experiments,
        seed=seed + 1,
        max_resource_tier=max_resource_tier,
    )
    candidates = _deduplicate(itertools.chain(candidates, random_candidates))
    return candidates[:max_experiments]


def full_factorial_design(
    space: ExperimentSpace,
    *,
    max_experiments: int,
    max_resource_tier: str | None = None,
) -> Iterator[ExperimentConfiguration]:
    """Yield valid Cartesian configurations with constraint pruning and a hard search guard."""

    if max_experiments <= 0:
        raise ExperimentSpaceError("full factorial requires a positive max_experiments limit")
    option_lists = [
        [option.option_id for option in _eligible_options(space, dimension, max_resource_tier)]
        for dimension in space.dimensions
    ]
    emitted = 0
    visited_partial_nodes = 0
    maximum_partial_nodes = max(25_000, max_experiments * 5_000)
    selections: dict[str, str] = {}

    def walk(index: int) -> Iterator[ExperimentConfiguration]:
        nonlocal emitted, visited_partial_nodes
        if emitted >= max_experiments:
            return
        dimension = space.dimensions[index]
        for option_id in option_lists[index]:
            visited_partial_nodes += 1
            if visited_partial_nodes > maximum_partial_nodes:
                raise ExperimentSpaceError(
                    "full-factorial safety guard reached before the requested number of valid "
                    "configurations; use mixed_screen or a narrower registry"
                )
            selections[dimension.dimension_id] = option_id
            if not space.validate(
                selections,
                partial=True,
                max_resource_tier=max_resource_tier,
            ):
                if index + 1 == len(space.dimensions):
                    emitted += 1
                    yield ExperimentConfiguration.build(
                        selections,
                        source="full_factorial",
                    )
                else:
                    yield from walk(index + 1)
            selections.pop(dimension.dimension_id, None)
            if emitted >= max_experiments:
                return

    yield from walk(0)


def design_configurations(
    space: ExperimentSpace,
    *,
    strategy: str,
    max_experiments: int,
    seed: int = 17,
    max_resource_tier: str | None = None,
) -> list[ExperimentConfiguration]:
    """Create a bounded, deterministic configuration design."""

    if max_experiments <= 0:
        raise ExperimentSpaceError("max_experiments must be positive")
    if max_resource_tier is not None:
        space.resource_rank(max_resource_tier)
    if strategy == "baseline":
        return baseline_and_named_design(space, max_resource_tier=max_resource_tier)[
            :max_experiments
        ]
    if strategy == "one_factor":
        return one_factor_design(space, max_resource_tier=max_resource_tier)[:max_experiments]
    if strategy == "pairwise_screen":
        return pairwise_screen_design(
            space,
            max_experiments=max_experiments,
            seed=seed,
            max_resource_tier=max_resource_tier,
        )
    if strategy == "random_valid":
        return random_valid_design(
            space,
            max_experiments=max_experiments,
            seed=seed,
            max_resource_tier=max_resource_tier,
        )
    if strategy == "mixed_screen":
        return mixed_screen_design(
            space,
            max_experiments=max_experiments,
            seed=seed,
            max_resource_tier=max_resource_tier,
        )
    if strategy == "full_factorial":
        return list(
            full_factorial_design(
                space,
                max_experiments=max_experiments,
                max_resource_tier=max_resource_tier,
            )
        )
    raise ExperimentSpaceError(f"unknown design strategy: {strategy}")


def schedule_first_round(
    configurations: Sequence[ExperimentConfiguration],
    *,
    budget: Budget,
    registry_digest: str,
) -> list[TrialRecord]:
    """Schedule the first measured round; later rounds require measured promotion decisions."""

    return [
        TrialRecord(
            trial_id="trial-"
            + _digest((registry_digest, configuration.config_id, budget.budget_id, 0)).split(
                ":", 1
            )[1][:16],
            configuration=configuration,
            budget=budget,
            round_index=0,
        )
        for configuration in configurations
    ]


def stage_configuration_key(
    space: ExperimentSpace,
    configuration: ExperimentConfiguration,
    through_stage: str,
) -> str:
    """Fingerprint configuration factors, not corpus inputs or mutable model artifacts."""

    if through_stage not in EXPERIMENT_STAGE_ORDER:
        raise ExperimentSpaceError(f"unknown experiment stage: {through_stage}")
    cutoff = EXPERIMENT_STAGE_ORDER.index(through_stage)
    allowed_stages = set(EXPERIMENT_STAGE_ORDER[: cutoff + 1])
    dimensions = space.dimension_by_id
    selected = [
        (dimension_id, option_id)
        for dimension_id, option_id in configuration.selections
        if dimensions[dimension_id].stage in allowed_stages
    ]
    return "stage-config-" + _digest(
        (space.registry_digest, through_stage, selected)
    ).split(":", 1)[1][:24]


def artifact_cache_key(
    stage_configuration_id: str,
    *,
    input_digests: Mapping[str, str],
    resolved_artifact_digests: Mapping[str, str],
) -> str:
    """Create a cache identity only after inputs and external artifacts are immutable."""

    if not input_digests:
        raise ExperimentSpaceError("artifact cache keys require at least one input digest")
    if any(not str(value).strip() for value in input_digests.values()):
        raise ExperimentSpaceError("input digests cannot be empty")
    if any(not str(value).strip() for value in resolved_artifact_digests.values()):
        raise ExperimentSpaceError("resolved artifact digests cannot be empty")
    return "artifact-cache-" + _digest(
        (
            stage_configuration_id,
            tuple(sorted(input_digests.items())),
            tuple(sorted(resolved_artifact_digests.items())),
        )
    ).split(":", 1)[1][:24]


def design_manifest(
    space: ExperimentSpace,
    configurations: Sequence[ExperimentConfiguration],
    trials: Sequence[TrialRecord],
    *,
    strategy: str,
    seed: int,
    max_resource_tier: str | None,
    requested_configuration_count: int | None = None,
) -> dict[str, object]:
    """Return a JSON-compatible, receipt-ready design manifest."""

    return {
        "schema_version": "1",
        "registry_digest": space.registry_digest,
        "strategy": strategy,
        "seed": seed,
        "max_resource_tier": max_resource_tier,
        "dimension_count": len(space.dimensions),
        "constraint_count": len(space.constraints),
        "runner_coverage": dict(space.runner_coverage),
        "raw_cartesian_size": space.raw_cartesian_size,
        "requested_configuration_count": requested_configuration_count,
        "scheduled_configuration_count": len(configurations),
        "underfilled": (
            requested_configuration_count is not None
            and len(configurations) < requested_configuration_count
        ),
        "warning": (
            "This file schedules experiments; it contains no accuracy, latency, cost, or "
            "accepted-outcome claims. Schedule and stage-configuration IDs do not pin corpus "
            "inputs or mutable external model artifacts and must not be used as result/cache IDs."
        ),
        "configurations": [
            {
                "config_id": item.config_id,
                "source": item.source,
                "selections": item.as_dict(),
                "stage_configuration_keys": {
                    stage: stage_configuration_key(space, item, stage)
                    for stage in EXPERIMENT_STAGE_ORDER
                },
            }
            for item in configurations
        ],
        "trials": [
            {
                "trial_id": item.trial_id,
                "config_id": item.configuration.config_id,
                "budget_id": item.budget.budget_id,
                "round_index": item.round_index,
                "status": item.status,
                "budget": {
                    "task_limit": item.budget.task_limit,
                    "package_limit": item.budget.package_limit,
                    "query_repeats": item.budget.query_repeats,
                    "learned_seeds": item.budget.learned_seeds,
                    "route_execution_limit": item.budget.route_execution_limit,
                },
            }
            for item in trials
        ],
    }


def pareto_front(
    rows: Sequence[Mapping[str, object]],
    objectives: Mapping[str, str],
) -> list[Mapping[str, object]]:
    """Return non-dominated measured rows for ``maximize``/``minimize`` objectives.

    Missing, non-numeric, NaN, and infinite objective values are excluded rather than silently
    imputed.  Hard acceptance and safety constraints should be applied before calling this helper.
    """

    if not objectives:
        raise ExperimentSpaceError("at least one Pareto objective is required")
    for direction in objectives.values():
        if direction not in {"maximize", "minimize"}:
            raise ExperimentSpaceError(f"unknown objective direction: {direction}")
    usable: list[tuple[Mapping[str, object], tuple[float, ...]]] = []
    for row in rows:
        values: list[float] = []
        for name, direction in objectives.items():
            try:
                value = float(row[name])
            except (KeyError, TypeError, ValueError):
                break
            if not math.isfinite(value):
                break
            values.append(value if direction == "maximize" else -value)
        else:
            usable.append((row, tuple(values)))

    frontier: list[Mapping[str, object]] = []
    for index, (row, values) in enumerate(usable):
        dominated = False
        for other_index, (_, other) in enumerate(usable):
            if index == other_index:
                continue
            if all(left >= right for left, right in zip(other, values, strict=True)) and any(
                left > right for left, right in zip(other, values, strict=True)
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return frontier
