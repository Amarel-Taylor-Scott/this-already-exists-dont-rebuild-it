# Data and generated artifacts

`data/seed/` contains only small, reviewable inputs and judgments needed to reproduce the first
tests. It is not a statistically meaningful training set.

Generated outputs belong in ignored directories:

- `data/generated/` for local catalog snapshots;
- `indexes/` for lexical/dense indexes;
- `reports/live/` for current receipts and metrics;
- `reports/releases/` only when a frozen manifest is intentionally prepared for release;
- external content-addressed storage for package archives, large datasets, vectors, and models.

Every frozen data release must include source cutoffs/serials, artifact hashes, schema/extractor
versions, configurations, counts, errors, licenses/provenance, splits, checksums, and a leakage audit.

