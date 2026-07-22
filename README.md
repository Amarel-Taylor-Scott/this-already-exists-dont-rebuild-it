# This Already Exists — Don't Rebuild It

An execution-grounded research project for testing whether coding systems can recognize that
working Python capabilities already exist, find the correct package operation and version, compose
compatible operations, and generate only the genuinely missing part.

This repository is a distinct, Python/PyPI-focused experimental stub related to Taedri. It is not a
copy of Taedri, does not require Taedri, and does not yet claim that a small model beats a frontier
model or that 10,000 packages are executable.

## The question

Most code benchmarks ask whether a model can recreate a solution. This project asks a different,
falsifiable question:

> Given a request and a frozen catalog of existing package operations, which combination of exact
> lookup, deterministic labels, blocking, sparse retrieval, embeddings, compatibility checks,
> adapters, and small models reaches an independently accepted outcome with the least genuinely new
> code and model work?

Retrieval is not success. A strong reuse result requires the selected operation to execute, produce
an output, have that output consumed, and contribute to an accepted result.

## Current first slice

The initial code is deliberately narrow enough to test:

- static inspection of installed Python distributions without importing the target package;
- package, function, class, method, signature, docstring, dependency, and source-digest records;
- observed representations kept separate from versioned derived labels and blocking keys;
- exact, SQLite FTS5, word/character TF-IDF, deterministic blocking, and rank fusion baselines;
- multiple acceptable operation sets, hard negatives, and correct no-reuse abstention;
- named artifact ports and pre-execution compatibility rejection;
- an allowlisted pandas route with direct reuse, multi-operation composition, a hard negative, a
  no-reuse result, and one explicitly counted task-specific residual;
- execution receipts with exact package versions, artifact digests, downstream consumption, an
  independent output check, latency, and zero-model telemetry.

Generated databases, reports, and receipts are not committed as source. Rebuild them from the
versioned configuration and seed data.

## Quick start

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --all-extras
uv run reuse-code doctor
uv run reuse-code ingest-installed --package pandas --package scikit-learn --package packaging
uv run reuse-code benchmark-retrieval
uv run reuse-code design-experiments --strategy mixed_screen --max-experiments 200 --budget smoke
uv run reuse-code run-seed
uv run pytest
```

The `UV_CACHE_DIR` override is only needed in environments whose default cache is read-only.

## Kaggle labs

Run the notebooks in order:

1. [`01_kaggle_build_capability_catalog.ipynb`](notebooks/01_kaggle_build_capability_catalog.ipynb)
   installs or clones the project and statically extracts package → subpackage → module → symbol
   evidence into a bounded immutable shard.
2. [`02_kaggle_generate_search_features.ipynb`](notebooks/02_kaggle_generate_search_features.ipynb)
   creates the append-only registry for arbitrary attributes, descriptions, phrases, labels, ports,
   edges, fingerprints, LSH buckets, onion layers, and field-specific embeddings.
3. [`03_kaggle_small_model_routing_lab.ipynb`](notebooks/03_kaggle_small_model_routing_lab.ipynb)
   evaluates deterministic and small-model route selection at symbol granularity, including
   compatibility gates, multi-step composition, hard negatives, and abstention.
4. [`04_kaggle_conditional_retrieval_model_sweep.ipynb`](notebooks/04_kaggle_conditional_retrieval_model_sweep.ipynb)
   qualifies the Kaggle hardware/model runtime and generates bounded, conditionally valid sweeps
   across the versioned retrieval, embedding, index, LSH, planner, and verification registry.

All four include Kaggle `!pip install` bootstrap cells and write artifacts under
`/kaggle/working`. A sweep design is a schedule rather than a measured result; model and retrieval
receipts are written separately. The registry intentionally includes design-only options whose
execution adapters have not yet been connected, and every manifest reports that partial runner
coverage. The optional generative arm defaults to Google's small
[Gemma 4 E2B](https://ai.google.dev/gemma/docs/core) checkpoint when enabled; deterministic
validation remains authoritative. Large catalogs are produced as independently rerunnable shards,
not as one 10,000-package in-memory notebook object.

## Evidence levels

| Level | What it proves |
|---|---|
| Metadata only | A distribution record exists and its declared metadata was observed. |
| Static source | A symbol/signature/docstring was parsed from a specific artifact or installed file. |
| Smoke tested | The exact package release installed/imported and a focused operation ran. |
| Contract tested | Named inputs, outputs, and failure boundaries passed an explicit test. |
| Accepted route | The operation's output was consumed by a replayed route that passed an independent evaluator. |

Searchable does not mean executable. Similar does not mean compatible. Generated labels do not
become facts merely because a model produced them.

## Scale plan

1. **Seed:** prove the whole request → retrieval → compatibility → execution → acceptance chain.
2. **100–250 packages:** deep extraction, reviewed examples, version/platform identity, typed
   contracts, and 300–500 high-quality intent-to-operation and recipe tasks.
3. **1,000 packages:** stress incremental ingestion, namespace collisions, long-tail retrieval,
   major-version drift, native wheels, hard negatives, and index experiments.
4. **10,000+ packages:** scale only after additional packages measurably add useful accepted-outcome
   coverage rather than mainly adding ambiguity and storage cost.

At scale, one PyPI project is not one capability. The catalog must preserve project → release →
artifact → module → symbol → example/test → inferred capability → verified adapter/route. Separate
wheel variants may contain different Python/native code and require separate identities.

## Research and design

- [Architecture and 10,000-package ingestion design](docs/architecture.md)
- [Current research and adjacent systems](docs/research_landscape.md)
- [Experiment and ablation matrix](docs/experiment_matrix.md)
- [Conditional retrieval/model experiment system and Kaggle T4 guidance](docs/retrieval_model_experiment_system.md)
- [Kaggle experiment notebooks and feature ledger](docs/kaggle_experiments.md)
- [Initial engineering validation record](reports/initial-validation.json)
- [Decisions, claims, and open questions](docs/decisions.md)
- [Data and generated-artifact policy](data/README.md)

The design builds on official PyPI/PyPA interfaces rather than scraping project pages: the
[Simple Repository API](https://packaging.python.org/specifications/simple-repository-api/),
[PyPI JSON API](https://docs.pypi.org/api/json/), and
[PyPI BigQuery datasets](https://docs.pypi.org/api/bigquery/). Package metadata is evidence, not a
trust verdict; PyPI explicitly notes that uploaded JSON metadata can differ from the distribution
contents.

## Relationship to Taedri

Taedri is the broader existing-code semantic compiler and production architecture. This repository
is an independent benchmark and PyPI experimentation laboratory. A future one-way adapter can map
frozen package-operation records, compatibility evidence, routes, and receipts into Taedri. Taedri
must remain one system under test rather than the source of benchmark truth.

## Licensing status

No repository license has been selected yet. The package records retain upstream license evidence,
and large-scale work must distinguish index-only metadata, linked source, redistributable content,
executable artifacts, and generated adapters. Public availability is not assumed to grant
redistribution rights.
