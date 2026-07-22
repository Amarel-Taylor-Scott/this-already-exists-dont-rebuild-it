# Experiment and ablation matrix

## Versioned conditional configuration

The declared experimental option envelope is [`configs/experiment_space.json`](../configs/experiment_space.json):
41 dimensions across corpus, representation, query, blocking, retrieval, embedding, index,
reranking, compatibility, planning, runtime, and verification, with 38 constraints that reject
nonsensical combinations. Generate a bounded schedule with:

```bash
uv run reuse-code design-experiments \
  --strategy mixed_screen \
  --max-experiments 200 \
  --max-resource-tier t4_full \
  --budget smoke
```

The raw Cartesian envelope is approximately 2.10 × 10^32 configurations, so it is not an
execution plan. Screen in stages, promote survivors under increasing budgets, and reserve the
holdout for selected configurations. A generated manifest is scheduled work, not measured
evidence. The [conditional experiment-system guide](retrieval_model_experiment_system.md) documents
all dimensions, constraints, Kaggle T4/model-runtime qualification, storage and LSH choices, and
receipt requirements.

## Primary outcomes

Keep separate leaderboards. Do not hide tradeoffs in one arbitrary score.

| Track | Primary measures |
|---|---|
| Reuse recognition | PR-AUC, F1, Brier score, calibration error, false-reuse and missed-reuse rates |
| Candidate blocking | blocking recall, candidate reduction, false exclusion, latency |
| Operation retrieval | Recall@K, MRR, nDCG@K, compatible Recall@K, hard-negative rate |
| Route retrieval | required-node/edge recall, adapter recall, dependency completeness, route edit distance |
| Binding | parameter/artifact/schema exactness, identity preservation, policy conformance |
| Execution | install/import/compile/run/acceptance rates, repair count, clean replay |
| Contribution | selected/called/produced/consumed/contributing states and fixed-route ablations |
| Efficiency | latency, CPU/GPU time, peak memory, index bytes, model calls/tokens, verified money or null |
| Avoided reconstruction | reusable behavior reimplemented, glue code, audited novel residual, dead code |
| Accumulation | first-task versus next-independent-related-task cost and acceptance |

The headline end-to-end result is the failure-inclusive resource vector to the first independently
accepted, quality-qualified outcome.

## Corpus and granularity experiments

- package-only versus module versus operation versus executable-example records;
- constructor/class lifecycle retained versus flattened methods;
- 100–250, 1,000, and 10,000 package snapshots;
- operation counts and ambiguity by popularity decile/domain/artifact kind;
- metadata-only versus static-source versus smoke-tested versus contract-tested candidates;
- current release versus older major versions and platform-specific artifacts.

## Representation experiments

Evaluate independently and in combinations:

- normalized package/import/symbol identity;
- signature and type hints;
- docstring purpose;
- implementation body;
- individual example;
- package summary;
- input contract;
- output contract;
- limitations/refusals/failure evidence;
- workflow role;
- composition partners;
- business-language purpose;
- fixed token chunks versus entity-aware chunks;
- single concatenated vector versus separately indexed fields versus late interaction.

## Candidate-generation and ranking experiments

- exact aliases/imports;
- word and character TF-IDF;
- fielded BM25/FTS5;
- trigram/fuzzy names;
- deterministic blocking keys, individually and in unions/intersections;
- MinHash/SimHash/AST/call fingerprints;
- each dense model/revision/dimension/precision;
- sparse+dense reciprocal-rank fusion;
- calibrated weighted fusion;
- compact cross-encoder reranking at several depths;
- hierarchical package → module → operation retrieval;
- typed graph expansion;
- compatibility before versus after reranking;
- retrieval K, candidate budget, cold/warm cache, and update latency.

## Label and small-model experiments

- deterministic labels only;
- TF-IDF logistic regression and linear SVM;
- optional tree model over retrieval/facet features;
- compact bi-encoder;
- multi-task encoder predicting reuse, actions, artifacts, stages, and retrieval embedding;
- quantized CPU variants;
- small constrained recipe selector;
- frontier router as equal-access upper bound;
- human/oracle retrieval.

All models must report parameter count, artifact size, hardware, peak memory, batch/query latency,
training time, dataset/code digests, calibration, and seeds.

## Compatibility and adapter experiments

- version/Python/wheel-tag filtering;
- dependency marker/extras resolution;
- package/import namespace ownership;
- signature and static-type checks;
- shape/dtype/nullability/order/cardinality checks;
- side-effect/resource/lifecycle policy;
- adapter primitives individually and cumulatively;
- inferred versus dynamically observed versus contract-tested edges;
- positive routes plus vocabulary-matched wrong-type/version/platform/license candidates.

## Leakage-resistant splits

- prompt/paraphrase holdout;
- new combination of known operations;
- unseen operation/package family;
- temporal release cutoff;
- held-out major version;
- repository/example-template lineage holdout;
- cold long-tail packages;
- no valid reusable supply;
- terminology-matched incompatible candidates.

Near-duplicate prompts, examples, source forks, package aliases, and generated paraphrase families
must remain in one split group.

## Statistical protocol

- freeze catalog, task, representation, index, configuration, and environment digests;
- give all conditions the same package supply, documentation policy, compute/time/context budgets,
  retry policy, and evaluator;
- separate index-build, cold query, warm query, environment-build, and execution costs;
- tune on validation data under equal budgets;
- use multiple seeds for learned components;
- report paired bootstrap confidence intervals and paired comparisons;
- report by domain, popularity decile, evidence level, artifact/platform type, and documentation/type
  coverage;
- retain failed and unfavorable runs as data.
