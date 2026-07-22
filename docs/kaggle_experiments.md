# Kaggle experiment notebooks

The notebooks are executable experiment drivers for Kaggle CPU/GPU sessions. They are intentionally
split so extraction can be cached, expensive feature generation can be resumed, and routing
experiments cannot silently rewrite source observations.

| Notebook | Responsibility | Primary output |
|---|---|---|
| `01_kaggle_build_capability_catalog.ipynb` | Static package, subpackage, module, class, function, and method extraction | normalized catalog, hierarchy, FTS index, Parquet mirrors, manifest |
| `02_kaggle_generate_search_features.ipynb` | Attributes, descriptions, phrases, labels, keys, ports, edges, fingerprints, LSH buckets, and embedding variants | append-only feature ledger and versioned index projections |
| `03_kaggle_small_model_routing_lab.ipynb` | Constrained goal/recipe selection, compatibility checks, multi-step routes, abstention, and mixture ablations | task predictions, candidate provenance, metrics, latency, and experiment manifest |
| `04_kaggle_conditional_retrieval_model_sweep.ipynb` | Hardware/model qualification and bounded conditional design over retrieval, embeddings, LSH, indexes, planners, runtimes, and verification | hardware receipt, scheduled design, model qualification receipts, and retained failures |

Each notebook starts with editable settings and Kaggle-compatible install/clone cells. Defaults are
small smoke runs. Increasing a package list, embedding model, or route-task grid is an explicit
configuration change recorded in the output manifest.

Notebook 04 reads the 41-dimension, 38-constraint registry in
[`configs/experiment_space.json`](../configs/experiment_space.json). Its default `mixed_screen`
design is bounded by configuration count, resource tier, seed, and one of the `smoke`, `screen`,
`confirm`, or `holdout` budgets. The generated design contains no measured performance claims.
See the [conditional experiment-system guide](retrieval_model_experiment_system.md) for all stages,
model/runtime qualification, storage and LSH options, metrics, and receipt requirements.

## Flexible features without an unbounded canonical schema

Observed package facts stay normalized and immutable. Search experiments use the polymorphic
envelope in [`derived-feature.schema.json`](../schemas/derived-feature.schema.json). The open
`feature_kind`, `namespace`, and JSON `value` fields allow new attributes without adding a database
column or migrating source records. Examples include:

- `description/purpose`, `search_phrase/user_wording`, and `label/action_object`;
- `port/input_artifact`, `port/output_shape`, and `attribute/effect`;
- `edge/calls`, `edge/returns_compatible_with`, and `compatibility_path/adapter_chain`;
- `fingerprint/minhash`, `lsh_bucket/ast_shingle`, and `embedding/documentation`;
- `onion_layer/package_summary`, `onion_layer/subpackage`, and `onion_layer/symbol_example`.

Every feature identifies its generator and revision, source evidence, configuration digest,
confidence, and review state. Generated descriptions and model labels therefore remain hypotheses.
Kind-specific projections may place text in FTS, vectors in an ANN index, edges in adjacency tables,
or LSH buckets in lookup tables while retaining a link to the same ledger record.

## Kaggle data flow

1. Enable Internet for the bootstrap/clone and package-download cells, or attach the repository and
   wheels as a Kaggle Dataset for a network-free run.
2. Run notebook 01 and save `/kaggle/working/capability_catalog` as a Dataset version.
3. Attach that Dataset to notebook 02. Save each feature release separately rather than overwriting
   the catalog.
4. Attach both releases to notebook 03 and run frozen retrieval/routing profiles.
5. Use notebook 04 for hardware/model qualification and bounded sweep scheduling. Reuse the frozen
   catalog/tasks and unload one external model before loading the next.
6. Download manifests, task-level predictions, model/runtime receipts, failures, and execution
   receipts before the Kaggle session expires.

The catalog notebook does not import or execute target packages. Any later dynamic contract probes
need a separate hardened runner; a Kaggle notebook is not a security boundary for hostile packages.
Likewise, a model-generated description, edge, or route remains a candidate until reviewed and,
where applicable, independently executed and accepted.

## Scaling rule

Parallelize by immutable project/release/artifact shards and merge by stable identifiers. Do not
build a 10,000-package Python object graph in one notebook process. Promote a feature or embedding
to a serving index only after its incremental value survives equal-budget, leave-one-out, temporal,
package-family, and hard-negative evaluation.
