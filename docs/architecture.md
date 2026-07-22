# Architecture and 10,000-package ingestion design

## Boundary

Version 1 is Python/PyPI only. It indexes real package supply at several granularities, evaluates
retrieval and compatibility, and executes a reviewed subset. Cross-language code, SaaS connectors,
general knowledge objects, multi-tenancy, and a production Taedri server are later concerns.

The plain-language boundary is:

```text
PyPI snapshot
→ artifact and operation extraction
→ labels, keys, chunks, fingerprints, and embeddings
→ candidate retrieval experiments
→ version, policy, artifact, and adapter compatibility
→ constrained operation/route selection
→ isolated execution
→ independent acceptance and contribution receipt
```

## Do not collapse the evidence layers

The system has three different layers:

1. **Observed facts:** artifact hashes, metadata, files, AST symbols, signatures, docstrings, type
   stubs, examples, imports, calls, and test results.
2. **Derived search signals:** normalized tokens, purposes, intent labels, artifact hints, blocking
   keys, fingerprints, chunks, embeddings, graph weights, and model scores. Every value names its
   generator, revision, source fields, and confidence.
3. **Verified capabilities:** exact package release/artifact, named input/output ports, environment,
   invocation, limitations, adapter, smoke/contract tests, and accepted-route evidence.

A package-level summary must never be counted as a verified operation. A static signature does not
prove semantic input/output compatibility. An embedding hit does not prove that two operations can
compose.

## Canonical entities

| Entity | Required identity/evidence |
|---|---|
| Project | normalized PyPI name and index serial/provenance |
| Release | project, exact normalized version, upload/release facts |
| Artifact | filename, SHA-256, distribution type, Python/ABI/platform tags, yanked state |
| Module | artifact hash, source path, module/import name |
| Operation | module, qualified name, kind, signature, source span/digest |
| Representation | entity, field/plane, chunk, source fields, generator/revision |
| Label/blocking key | namespace, value, confidence, evidence fields, generator/revision |
| Example | source path/URL, license, imports/calls/assertions, execution state |
| Port/contract | direction, name, artifact/type/shape, state, cardinality, effects |
| Adapter | source/target contracts, version ranges, lossiness, code hash, tests |
| Compatibility claim | exact endpoints, environment/policy, status, evidence, validity interval |
| Request/judgment | prompt, artifacts, constraints, acceptable sets, hard negatives, split |
| Route/run/receipt | bound versions, calls, artifacts, lineage, evaluator, model/resource usage |

Use append-only variants for derived data and experiment results. Correcting a label creates a new
derivation; it does not rewrite the source observation.

## Authoritative ingestion path

### Registry snapshot

- Use the JSON form of the
  [Simple Repository API](https://packaging.python.org/specifications/simple-repository-api/) for
  project/artifact discovery and incremental serials.
- Use the [PyPI JSON API](https://docs.pypi.org/api/json/) for convenient descriptions, URLs,
  ownership, and vulnerability summaries, while retaining PyPI's warning that upload-time metadata
  can differ from the artifact.
- Use [PyPI BigQuery datasets](https://docs.pypi.org/api/bigquery/) for immutable distribution
  metadata and stratified popularity sampling; the JSON API's download counts are deprecated and
  always `-1`.
- Store each selected distribution artifact by its advertised SHA-256. A release is not a sufficient
  identity because wheels for different platforms may carry different native or Python content.

### Static control plane

Never install, import, build, or run untrusted packages in the control process.

1. Inspect wheel/sdist members with file-count, uncompressed-byte, compression-ratio, path,
   recursion, and timeout limits.
2. Read `METADATA`, `WHEEL`, `RECORD`, entry points, license files, type markers/stubs, and wheel SBOMs.
3. Parse Python with CPython AST for the seed; compare and likely extend with
   [Griffe](https://mkdocstrings.github.io/griffe/) for aliases, API docs, signatures, serialization,
   and breaking-change information.
4. Reuse [SCIP](https://github.com/scip-code/scip),
   [scip-python](https://github.com/sourcegraph/scip-python), or
   [Glean](https://github.com/facebookincubator/Glean) for scalable symbol/reference facts rather
   than creating a proprietary cross-reference format first.
5. Extract examples separately from README/docs, doctests, notebooks, example directories, and
   focused tests. Preserve file-level source and license evidence.

### Dynamic evidence plane

Dynamic import, reflection, probes, and tests belong in disposable network-denied sandboxes with no
credentials, a read-only base, explicit writable output, CPU/memory/time/process quotas, and complete
logs. [OpenSSF Package Analysis](https://github.com/ossf/package-analysis) is a reference design to
evaluate before building a new dynamic analysis system.

## Package selection

“Top 10,000 downloads” would overfit popular data/web packages and underrepresent the long tail. A
frozen release should combine:

- popularity strata from BigQuery;
- topic/classifier strata;
- pure Python, compiled, native-library, CLI-only, plugin, and namespace packages;
- recently released and stable/older versions;
- strong and weak documentation/type/example coverage;
- license-confidence strata;
- deliberately confusable names and purposes;
- packages needed by real benchmark requests.

Publish the selection query, PyPI serial or cutoff, artifact hashes, exclusions, and every ingestion
failure. Do not silently replace failed packages with easy ones.

## Retrieval cascade

All lanes are registered experiment configurations, not permanent architectural winners.

1. Exact normalized package, import, symbol, alias, error, and version identifiers.
2. Multiple overlapping deterministic keys for language, kind, action, object, input/output artifact,
   mode, effects, workflow stage, platform, license, evidence, and risk.
3. Fielded word BM25/FTS5 and character trigram/fuzzy retrieval.
4. MinHash/SimHash/AST/call fingerprints for duplicates, notebook/example lineage, and structural
   families—not as a substitute for semantic intent.
5. Dense retrieval over separately versioned identity, purpose, signature, documentation, example,
   implementation, contract, limitation, and failure planes.
6. Reciprocal-rank or learned fusion.
7. Compact shortlist reranking.
8. Hard version, artifact, platform, policy, resource, and port compatibility.
9. Typed graph expansion and adapter insertion.
10. Calibrated reuse, no-reuse, prohibited, insufficient-evidence, or genuine-gap decision.

Entity-aware chunks are the default unit. Fixed token windows remain an ablation.

## Embedding feature registry

Do not add 100 vector columns to the canonical operation table. Use a derived registry keyed by:

```text
(entity_id, field_or_chunk_id, model_id, model_revision,
 preprocessing_revision, pooling, dimension, precision, vector_digest, index_release)
```

One million 768-dimensional fp32 vectors require about 3.07 GB before ANN overhead. One hundred
such representations would be roughly 307 GB per million entities; five million entities would be
about 1.54 TB. Generate many candidates for offline experiments, but promote only measured winners
to serving indexes. Test fp16, dimensional truncation, and product quantization explicitly.

Initial model families to compare include:

- [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) with its paired compact
  reranker;
- [BGE-Code-v1](https://huggingface.co/BAAI/bge-code-v1);
- [Jina code embeddings](https://huggingface.co/jinaai/jina-embeddings-v2-base-code);
- [BGE-M3](https://huggingface.co/BAAI/bge-m3) as dense/sparse/multi-vector comparison;
- [ColBERT](https://github.com/stanford-futuredata/ColBERT) late interaction;
- [SPLADE](https://github.com/naver/splade) learned sparse retrieval.

These are experiment arms, not chosen winners.

## Compatibility layers

Keep these questions distinct:

1. `Requires-Dist`, extras, markers, and dependency resolution.
2. `Requires-Python` interpreter compatibility.
3. Python/ABI/platform wheel tags.
4. import ownership and namespace packages.
5. symbol existence and version-conditioned signatures.
6. static types and stubs.
7. data shape, dtype, order, nullability, cardinality, mutability, and state.
8. sync/async/streaming, lifecycle, resources, side effects, and exceptions.
9. native/external dependencies.
10. license, vulnerability, provenance, and organizational policy.

Reuse [`packaging.requirements`](https://packaging.pypa.io/en/stable/requirements.html), markers,
versions, and tags. A dependency resolver success is not runtime compatibility. Preserve
compatibility states as `compatible`, `compatible_with_adapter`, `requires_probe`, `incompatible`,
`prohibited`, or `unknown`, always scoped to an exact environment and evidence level.

## Adapters

Do not hand-author adapters for every package. Define a small declarative adapter language for the
high-frequency boundaries:

- rename/reorder/default arguments;
- record/dict/dataclass mappings;
- nominal/structural type coercion;
- array shape, dtype, and device conversion;
- iterator/list/stream materialization;
- path/file-like/bytes conversion;
- sync/async and callback/iterator normalization;
- lifecycle/context management;
- exception normalization;
- version-conditioned signature shims.

Every adapter is a versioned artifact with source/target contracts, supported versions, lossiness,
generated-code hash, tests, provenance, and known counterexamples.

## Small-model role

The small model should rank or fill a constrained request/recipe structure, not freely author a
complete program. Deterministic code then verifies exact symbols, resolves packages, checks ports and
policy, inserts tested adapters, emits imports/glue/lock information, and executes the route.

Compare deterministic-only, a compact classifier/encoder, a small code selector/planner, and a
frontier upper bound under equal package/document access and fixed budgets.

The first Kaggle planner arm uses the configurable Gemma-family harness in notebook 03. Google's
[Gemma 4 documentation](https://ai.google.dev/gemma/docs/core) lists the E2B and E4B variants as the
small edge-oriented models; start with `google/gemma-4-E2B-it` where its license, access, and Kaggle
memory permit. Model output is only a proposed `GoalIR` or `RecipeIR`: unknown operation IDs,
incompatible ports, prohibited effects, and invalid routes are deterministically rejected. Record
the exact model revision, quantization, prompt, thinking mode, candidate access, token budget, and
latency so model comparisons are equal-budget experiments rather than anecdotes.
