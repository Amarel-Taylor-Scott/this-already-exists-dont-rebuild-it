# Decisions, current claims, and open questions

## Decisions made for the first slice

- The repository is standalone and vendor-neutral; Taedri is an optional future integration.
- Python/PyPI is the first domain.
- Exact source/package observations remain authoritative; all search representations are derived.
- Static ingestion does not import target packages or execute build files.
- Searchable operations and verified executable capabilities are separate datasets.
- Configurations choose retrieval lanes; the code does not declare one universal winner.
- Multiple accepted operation sets are allowed.
- No reuse, prohibited reuse, insufficient evidence, and a genuine missing capability are valid
  decisions rather than benchmark failures.
- Exact package versions and artifact/source digests appear in execution evidence.
- Large artifacts, archives, embeddings, indexes, and generated reports stay out of Git.
- The scale gate is 100–250 → 1,000 → 10,000+, with evidence at every stage.
- Retrieval/model experimentation uses a versioned conditional registry and bounded screens; the
  raw Cartesian envelope is not an execution plan.
- Schedule and stage-configuration IDs are not artifact cache identities. Cacheable artifacts also
  require frozen corpus/task inputs and resolved model/index digests.
- Supervised, calibrated, optimized, and historical features declare grouped development,
  cross-fitted, or temporal lineage; the holdout is never a fitting source.
- A registry option may be design-only. A manifest reports partial runner coverage, and a valid
  configuration is not called executable until an adapter emits a receipt.

## Current evidence

Only generated test/benchmark receipts count. Architecture text and illustrative schemas do not
prove accuracy, token savings, small-model superiority, or 10,000-package readiness.

## Open decisions requiring evidence or owner choice

- repository source-code license;
- artifact storage and frozen public dataset location;
- which package/doc/example content may be redistributed versus linked/indexed only;
- initial 100–250 package sampling strata and supported Python/platform matrix;
- whether Griffe alone covers extraction needs or SCIP/scip-python is required immediately;
- sandbox implementation after evaluating OpenSSF Package Analysis;
- first adapter-language representation;
- first compact embedding/reranking models promoted beyond experiments;
- Kaggle publication only after local evaluator parity and leakage audit are green;
- exact Taedri one-way export/import schema.

## Explicit non-claims

- Catalog size is not reusable-capability count.
- A package install is not successful reuse.
- A call is not contribution.
- Co-use is not typed compatibility.
- A dependency lock is not runtime validation.
- A semantic match is not permission or safety.
- A generated label is not ground truth.
- Tokens are not dollars; monetary cost remains null without verified billing.
