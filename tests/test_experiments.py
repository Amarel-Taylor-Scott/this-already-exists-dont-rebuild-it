from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from existing_code_reuse.experiments import (
    ExperimentConfiguration,
    ExperimentSpaceError,
    artifact_cache_key,
    design_configurations,
    experiment_space_from_mapping,
    load_experiment_space,
    pareto_front,
    schedule_first_round,
    stage_configuration_key,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPACE_PATH = PROJECT_ROOT / "configs" / "experiment_space.json"


def test_published_experiment_space_has_valid_reference_configurations() -> None:
    space = load_experiment_space(SPACE_PATH)

    assert len(space.dimensions) >= 35
    assert len(space.constraints) >= 20
    assert space.raw_cartesian_size > 10**20
    assert dict(space.runner_coverage)["full_configuration_materializer"] is False
    assert space.validate(space.baseline) == ()
    for name, _ in space.named_configurations:
        assert space.validate(space.named_configuration(name)) == ()


def test_conditional_constraints_reject_meaningless_dense_index_pairs() -> None:
    space = load_experiment_space(SPACE_PATH)
    selected = space.baseline
    selected["dense_model"] = "all_minilm_l6_v2"

    failures = space.validate(selected)

    assert any("no_vector_index_means_no_dense_model" in failure for failure in failures)


def test_registry_rejects_constraint_option_typos() -> None:
    document = json.loads(SPACE_PATH.read_text(encoding="utf-8"))
    document["constraints"][0]["require"] = {"dense_model": ["not_a_model"]}

    with pytest.raises(ExperimentSpaceError, match="unknown options"):
        experiment_space_from_mapping(document)


def test_mixed_screen_is_bounded_reproducible_unique_and_valid() -> None:
    space = load_experiment_space(SPACE_PATH)

    first = design_configurations(
        space,
        strategy="mixed_screen",
        max_experiments=75,
        seed=29,
        max_resource_tier="t4_full",
    )
    second = design_configurations(
        space,
        strategy="mixed_screen",
        max_experiments=75,
        seed=29,
        max_resource_tier="t4_full",
    )

    assert [item.config_id for item in first] == [item.config_id for item in second]
    assert len(first) == 75
    assert len({item.config_id for item in first}) == len(first)
    assert all(space.validate(item.as_dict(), max_resource_tier="t4_full") == () for item in first)


@pytest.mark.parametrize(
    "strategy",
    (
        "baseline",
        "one_factor",
        "pairwise_screen",
        "random_valid",
        "mixed_screen",
        "full_factorial",
    ),
)
@pytest.mark.parametrize(
    "resource_tier",
    ("cpu_small", "cpu_large", "t4_light", "t4_full", "beyond_t4"),
)
def test_every_design_strategy_honors_a_one_configuration_bound(
    strategy: str,
    resource_tier: str,
) -> None:
    space = load_experiment_space(SPACE_PATH)

    configurations = design_configurations(
        space,
        strategy=strategy,
        max_experiments=1,
        seed=41,
        max_resource_tier=resource_tier,
    )

    assert len(configurations) == 1
    assert space.validate(
        configurations[0].as_dict(), max_resource_tier=resource_tier
    ) == ()


def test_random_screen_reaches_valid_t4_light_interaction_arms() -> None:
    space = load_experiment_space(SPACE_PATH)
    configurations = design_configurations(
        space,
        strategy="random_valid",
        max_experiments=20,
        seed=43,
        max_resource_tier="t4_light",
    )

    assert len(configurations) == 20
    assert any(item.source == "random_valid" for item in configurations)


def test_llm_query_features_can_be_ablated_without_llm_descriptions() -> None:
    space = load_experiment_space(SPACE_PATH)
    selected = space.baseline
    selected.update(
        {
            "query_facet_extractor": "small_model_structured_intent",
            "enrichment_model": "qwen3_5_0_8b",
            "llm_runtime": "transformers",
            "llm_quantization": "fp16",
        }
    )

    assert space.validate(selected, max_resource_tier="t4_light") == ()


def test_inactive_lsh_reranker_and_embedding_factors_are_canonical() -> None:
    space = load_experiment_space(SPACE_PATH)

    no_lsh = space.baseline
    no_lsh["lsh_resolution"] = "multi_probe"
    assert any("no_lsh_uses_neutral_resolution" in item for item in space.validate(no_lsh))

    no_reranker = space.baseline
    no_reranker["rerank_depth"] = "100"
    assert any("no_reranker_uses_neutral_depth" in item for item in space.validate(no_reranker))

    no_dense = space.baseline
    no_dense["embedding_precision"] = "int8"
    assert any("no_dense_means_no_vector_index" in item for item in space.validate(no_dense))


def test_matryoshka_constraints_follow_registered_model_dimensions() -> None:
    space = load_experiment_space(SPACE_PATH)
    modernbert = space.baseline
    modernbert.update(
        {
            "dense_model": "modernbert_embed_base",
            "vector_index": "numpy_exact",
            "embedding_dimension": "matryoshka_256",
        }
    )
    assert space.validate(modernbert) == ()

    snowflake = space.baseline
    snowflake.update(
        {
            "dense_model": "snowflake_arctic_embed_m_v2",
            "vector_index": "numpy_exact",
            "embedding_dimension": "matryoshka_512",
        }
    )
    assert any("matryoshka_512_supported_models" in item for item in space.validate(snowflake))


def test_stage_configuration_key_tracks_dependencies_and_ignores_later_changes() -> None:
    space = load_experiment_space(SPACE_PATH)
    base = ExperimentConfiguration.build(space.baseline, source="test")
    changed_values = base.as_dict()
    changed_values["verification_depth"] = "syntax_only_control"
    changed = ExperimentConfiguration.build(changed_values, source="test")

    assert stage_configuration_key(space, base, "embedding") == stage_configuration_key(
        space, changed, "embedding"
    )
    assert stage_configuration_key(space, base, "verification") != stage_configuration_key(
        space, changed, "verification"
    )

    dense_values = base.as_dict()
    dense_values["dense_model"] = "all_minilm_l6_v2"
    dense_values["vector_index"] = "numpy_exact"
    dense = ExperimentConfiguration.build(dense_values, source="test")
    assert stage_configuration_key(space, base, "retrieval") != stage_configuration_key(
        space, dense, "retrieval"
    )

    runtime_values = base.as_dict()
    runtime_values.update(
        {
            "description_generator": "llm_single_view_candidate",
            "enrichment_model": "qwen3_5_0_8b",
            "llm_runtime": "transformers",
            "llm_quantization": "fp16",
        }
    )
    runtime = ExperimentConfiguration.build(runtime_values, source="test")
    assert stage_configuration_key(space, base, "representation") != stage_configuration_key(
        space, runtime, "representation"
    )


def test_artifact_cache_key_requires_immutable_input_identity() -> None:
    with pytest.raises(ExperimentSpaceError, match="input digest"):
        artifact_cache_key("stage-config-123", input_digests={}, resolved_artifact_digests={})

    first = artifact_cache_key(
        "stage-config-123",
        input_digests={"catalog": "sha256:catalog-a"},
        resolved_artifact_digests={"model": "sha256:model-a"},
    )
    second = artifact_cache_key(
        "stage-config-123",
        input_digests={"catalog": "sha256:catalog-a"},
        resolved_artifact_digests={"model": "sha256:model-b"},
    )
    assert first != second


def test_scheduled_trial_identity_includes_budget() -> None:
    space = load_experiment_space(SPACE_PATH)
    configuration = ExperimentConfiguration.build(space.baseline, source="test")

    smoke = schedule_first_round(
        [configuration],
        budget=space.budget_by_id["smoke"],
        registry_digest=space.registry_digest,
    )[0]
    screen = schedule_first_round(
        [configuration],
        budget=space.budget_by_id["screen"],
        registry_digest=space.registry_digest,
    )[0]

    assert smoke.trial_id != screen.trial_id
    other_registry = schedule_first_round(
        [configuration],
        budget=space.budget_by_id["smoke"],
        registry_digest="sha256:changed-registry",
    )[0]
    assert smoke.trial_id != other_registry.trial_id
    assert replace(smoke, status="measured").configuration == configuration


def test_pareto_front_keeps_quality_cost_tradeoffs_and_excludes_bad_rows() -> None:
    rows = [
        {"id": "fast", "recall": 0.80, "latency": 2.0},
        {"id": "balanced", "recall": 0.90, "latency": 4.0},
        {"id": "dominated", "recall": 0.70, "latency": 6.0},
        {"id": "accurate", "recall": 0.95, "latency": 10.0},
        {"id": "missing", "recall": None, "latency": 1.0},
    ]

    result = pareto_front(rows, {"recall": "maximize", "latency": "minimize"})

    assert {row["id"] for row in result} == {"fast", "balanced", "accurate"}
