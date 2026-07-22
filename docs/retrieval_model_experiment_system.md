# Conditional retrieval and model experiment system

This document defines how to compare deterministic labels, NLP features, LSH and blocking,
lexical and dense retrieval, graph expansion, reranking, compatibility checks, small-model
planning, runtimes, and execution verification without confusing an enormous declared search
space with measured evidence.

The versioned registry is [`configs/experiment_space.json`](../configs/experiment_space.json). It
currently contains 41 dimensions across 11 earliest-affected artifact stages, 38 conditional
constraints, seven named
reference configurations, and four resource budgets. Its raw Cartesian product is
209,731,541,697,396,471,339,417,600,000,000 configurations. That number describes the option
envelope only. It is not a run count, coverage claim, or reason to attempt a literal grid search.

The experiment designer validates combinations, gives them stable schedule and stage-configuration
identifiers, and writes a bounded schedule. It does not run the configurations or
create accuracy, latency, cost, or accepted-outcome evidence.

Runner coverage is currently partial and is recorded in the registry and every design manifest.
The existing exact/word/character/FTS/hybrid profiles, deterministic seed verifier, embedding
qualification, Transformers Gemma 4 qualification, and existing-server Ollama qualification are
connected components. Unsloth, llama.cpp, most LSH/ANN/service/reranker/graph arms, storage
quantization, learned fusion, and full configuration-to-benchmark orchestration remain design-only
until an adapter and receipt writer are added. Conditional validity must never be read as runtime
availability.

## Generate a bounded design

The bundled registry is available from a checkout and from the built package:

```bash
uv run reuse-code design-experiments \
  --strategy mixed_screen \
  --max-experiments 200 \
  --max-resource-tier t4_full \
  --budget smoke \
  --seed 17 \
  --output reports/live/experiment-design.json
```

The output includes the registry digest, every complete selection, its origin, stable
configuration and trial identifiers, resource budget, and configuration fingerprints through each
stage. Those fingerprints intentionally omit corpus and mutable external artifacts. An artifact
cache key is created only after input digests and resolved model/index artifact digests are known.
The manifest says `scheduled`; a separate runner and receipt are required before any result can be
called measured.

Available design strategies are:

| Strategy | Purpose | Important limit |
|---|---|---|
| `baseline` | Deterministic baseline plus named reference configurations | A reference set, not a model ranking |
| `one_factor` | Change one option at a time from the baseline | Misses interactions and conditional changes that require another factor |
| `pairwise_screen` | Change two factors in a fixed baseline context | Deliberately not claimed to be a complete covering array |
| `random_valid` | Seeded sampling followed by conditional validation | Coverage depends on the limit and constraint density |
| `mixed_screen` | Baseline, named, one-factor, pair-screen, and valid random arms | Recommended first broad screen |
| `full_factorial` | Enumerate conditionally valid Cartesian rows | Stops at `--max-experiments` or a hard partial-search guard; use only on a narrowed registry |

The resource ceiling filters options before scheduling. It does not prove that a model will fit
the available GPU, because framework overhead, KV cache, batch shape, and the exact artifact also
affect memory.

## Stages and dimensions

Each dimension chooses one registered option. Its `stage` is the earliest artifact that the factor
can affect, not merely the factor's conceptual category. That deliberately moves enrichment/runtime
identity into representation keys and planner identity into rerank keys, preventing reuse of an
artifact produced by different model weights or runtime settings. A new model, label generator,
blocking key, index, or runtime is added as a versioned option rather than hidden in notebook code.

| Stage | Dimensions | What changes |
|---|---|---|
| Corpus | `catalog_unit`, `evidence_floor`, `training_data_policy` | Entity/evidence supply and the grouped, cross-fitted, temporal, external, or no-task-fit lineage allowed for learned state |
| Representation | `representation_set`, `chunk_policy`, `text_normalization`, `deterministic_feature_set`, `nlp_feature_generator`, `description_generator`, `enrichment_model`, `llm_runtime`, `llm_quantization`, `thinking_mode`, `planner_context_budget` | Observed fields, derived views, and the earliest model/runtime dependencies that can generate them |
| Query | `query_facet_extractor`, `query_expansion` | Action, object, input/output, constraint, domain, and alternative query views |
| Blocking | `blocking_key_set`, `blocking_logic`, `lsh_family`, `lsh_resolution` | Candidate reduction before ranking |
| Retrieval | `lexical_retriever`, `fusion_method`, `graph_edge_set`, `graph_expansion` | Exact/sparse ranking, channel fusion, and bounded graph evidence |
| Embedding | `dense_model`, `embedding_input_view`, `embedding_instruction`, `embedding_pooling`, `embedding_dimension`, `embedding_precision` | Model and exact vector-generation contract |
| Index | `similarity_metric`, `vector_index`, `ann_effort` | Exact or approximate vector search and its effort setting |
| Rerank | `reranker`, `rerank_depth`, `planner_model` | Candidate-order refinement; planner identity is included here because a small-LLM reranker can reuse it |
| Compatibility | `compatibility_gate`, `abstention_policy` | Version, type, artifact, effect, and uncertainty gates |
| Planning | `planner_candidate_count`, `structured_output` | Candidate budget and constrained recipe selection; provider/model factors already entered at their earliest affected stage |
| Verification | `verification_depth` | Static, import, smoke, contract, route, and independent acceptance checks |

This separation supports three necessary comparisons:

1. deterministic evidence and retrieval with no model loaded;
2. one model lane added to the same frozen candidate supply;
3. end-to-end planning and execution only after retrieval and compatibility are independently
   qualified.

Sentiment and tone are registered as a negative-control NLP arm. They may matter for a task that
explicitly asks for sentiment analysis or an urgency/policy signal, but they should not be assumed
to improve general package-operation retrieval.

## Conditional validity

The 38 constraints remove nonsensical or misleading selections before they consume benchmark
budget. Examples include:

- no dense model implies no vector index, and every dense model must name an index;
- Matryoshka truncation is allowed only for a model registered as supporting that dimension;
- binary vectors use the Hamming/Faiss-LSH arm rather than cosine search;
- graph expansion requires an edge supply;
- an LLM-generated description must name its enrichment model and runtime;
- a deterministic planner cannot silently load a model server;
- Transformers, Unsloth, llama.cpp, and Ollama accept different quantization formats;
- Gemma 4 E4B is restricted to a quantized arm for the 16 GB Kaggle tier;
- supervised, calibrated, optimized, and historical features must name development/cross-fit or
  temporal lineage, while accepted-route/query history must precede the evaluated request;
- all model output is schema-checked or post-validated before it can nominate an allowlisted
  operation.

Constraints establish that a trial is meaningful enough to run. They do not establish semantic
compatibility or execution safety. Unknown input/output, version, platform, or side-effect
compatibility still fails closed in the route verifier.

## Staged search instead of an uncontrolled Cartesian product

Use the same frozen tasks and candidate supply within a comparison round. Promote configurations
only when their incremental value survives the current budget.

### 1. Runtime and artifact qualification

Before measuring retrieval, record the GPU, CUDA, PyTorch, Python, package versions, model Hub ID,
resolved revision, file digests, quantization, maximum context tested, peak allocated/reserved
memory, and a small deterministic output check. A model that does not load, returns non-finite
vectors, violates the output schema, or changes a route under a cache-parity check is a retained
failure, not a skipped row.

### 2. Representation and blocking screen

Start with the deterministic lexical baseline. Change one representation, normalization, feature,
blocking, or LSH factor at a time, then run the bounded pair screen. Blocking is promoted only if
its candidate reduction is worth its measured false-exclusion risk. Before ANN or reranking, use an
exact unblocked retrieval lane as the diagnostic oracle.

### 3. Embedding and index screen

Compare model revisions under the same query/document instructions and source fields, then compare
field views, pooling, supported Matryoshka dimensions, and output precision within the surviving
models. Evaluate exact NumPy or Faiss Flat search before HNSW, IVF, PQ, or service-backed indexes so
embedding quality is not confounded with approximate-index recall.

### 4. Fusion, graph, and reranking

Add exact/lexical/dense channels independently, then evaluate reciprocal-rank or calibrated fusion.
Only then test graph expansion and rerank depth. Record every candidate's channel score and rank so
an apparent gain can be attributed to the channel that supplied it.

### 5. Compatibility and planner screen

Freeze the retrieved candidate lists before comparing deterministic, FunctionGemma, general small
models, and Gemma 4 planners. The planner selects only stable operation or adapter identifiers from
the supplied allowlist. It cannot invent an executable capability. Its output remains a proposal
until compatibility and verification pass.

### 6. Confirm and holdout

The registry budgets increase from `smoke` (12 tasks, 20 packages) to `screen` (75 tasks, 100
packages), `confirm` (300 tasks, 250 packages), and `holdout` (500 tasks, 250 packages). These are
caps in a schedule, not completed benchmark counts. Successive promotion can use a deterministic
rule or a tool such as [Optuna's conditional/multi-objective samplers](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html),
but the holdout remains sealed and is run once for the selected configurations.

Do not collapse retrieval quality, false reuse, execution acceptance, latency, memory, and storage
into one arbitrary score. Keep the Pareto frontier and expose the tradeoffs.

## Kaggle T4 model qualification

Notebook 04, `04_kaggle_conditional_retrieval_model_sweep.ipynb`, is the GPU qualification and
bounded-design driver. It creates outputs under `/kaggle/working`, unloads one model before
loading the next, and keeps download, build, query, and route-execution costs separate. Its current
bootstrap and immutable-revision resolver require Internet access. Official
[`kagglehub`](https://github.com/Kaggle/kagglehub) can attach cached resources inside a Kaggle
notebook, but notebook 04 does not yet implement a network-free local-artifact loader or artifact
digest writer; those remain required before an attached-resource run can be called frozen.

Accept any gated model terms before starting the notebook and provide the Hugging Face token as a
Kaggle secret, never as a stored source cell. Access denial, missing artifacts, and revision
mismatch are qualification failures and stay in the output ledger. A future Internet-disabled arm
must attach every wheel and model artifact, hash those local artifacts, and avoid Hub resolution.

An NVIDIA T4 has 16 GB GDDR6 and supports FP16, INT8, and INT4 operations according to the
[official T4 specification](https://www.nvidia.com/en-us/data-center/tesla-t4/). Do not assume a
dtype from the accelerator name. Probe the actual runtime with
[`torch.cuda.is_bf16_supported()`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.is_bf16_supported.html)
and record the result. FP16 is the normal reference on a T4; BF16 is selected only when the runtime
reports support.

### Gemma 4

The first text-planner qualification target is `google/gemma-4-E2B-it`. Google's
[Gemma 4 overview](https://ai.google.dev/gemma/docs/core) estimates 11.4 GB for E2B at BF16 and
2.9 GB at Q4, while E4B is estimated at 17.9 GB at BF16 and 4.5 GB at Q4. Those estimates include
weight-loading overhead but exclude supporting software and growing KV cache, so every actual
context and batch setting still needs a peak-memory receipt. The registry therefore permits a
careful E2B 16-bit reference arm and keeps E4B quantized on the 16 GB tier.

The advertised 128K context for the small family is not a Kaggle operating target. Qualify short
2K, 4K, and 8K planner contexts first and increase only after recording memory and latency. The
router does not need multimodal inputs merely because the model supports them.

Use the current official
[Gemma 4 Transformers inference path](https://ai.google.dev/gemma/docs/core/huggingface_inference)
as the reference implementation. Pin Transformers and all model dependencies, resolve the exact
Hub revision, use the official processor/model class for Gemma 4, and preserve the complete load or
generation exception. Quantized Transformers arms must record the exact
[`bitsandbytes` configuration](https://huggingface.co/docs/transformers/quantization/bitsandbytes).

Treat the other runtimes as separate experiments:

| Runtime | Role | Qualification requirement |
|---|---|---|
| Transformers | Reference implementation and model revision | Exact Python packages, model revision, dtype/quantization, processor, peak memory, cache-parity result |
| Unsloth | Optional accelerated Transformers-compatible arm | Pin the Unsloth stack, use its declared [Gemma 4 support](https://github.com/unslothai/unsloth/discussions/4800), and compare the same frozen prompts and accepted JSON against the reference |
| llama.cpp server | Direct GGUF server arm | Exact build revision, command, GGUF filename/digest, GPU-layer count, context, health check, and server telemetry |
| Ollama | Convenience local-server arm | Exact Ollama version, [Gemma 4 tag](https://ollama.com/library/gemma4), underlying artifact identity if available, health check, and the same JSON allowlist validator |

Ollama is useful for rapidly testing local model tags, but it is not the only or authoritative
path. Starting a daemon, pulling a different quantization, or changing a Modelfile changes the
trial. The notebook should connect to an explicitly enabled server rather than silently download a
large daemon and model during a deterministic arm.

[FunctionGemma](https://ai.google.dev/gemma/docs/functiongemma) is a specialized function-calling
arm, not a general-purpose planner assumed to work out of the box. Its base behavior and any
task-specific fine-tuning are separate configurations with separate artifacts and receipts.

### EmbeddingGemma and the embedding panel

Google describes [EmbeddingGemma](https://ai.google.dev/gemma/docs/embeddinggemma) as a 308M
multilingual model trained across more than 100 languages, with a 2K-token input and flexible
768-to-128-dimensional output through Matryoshka Representation Learning. It is one embedding arm,
not the default winner.

For this project:

- use separate query and document encoding methods/instructions when the model supplies them;
- encode individually or in length buckets during qualification, then assert that every vector is
  finite and nonzero;
- compare native dimension with only the registered supported truncations and re-normalize after
  truncation;
- keep a CPU/FP32 reference receipt before testing storage quantization;
- compare against deterministic TF-IDF/BM25, MiniLM, BGE small, E5, Nomic, Qwen3 embedding,
  BGE-M3, Jina code, CodeRankEmbed, and UniXcoder options under the same tasks;
- use code-oriented embeddings only when the input view actually contains signatures, examples,
  source, or natural-language-to-code instructions.

Model-card benchmark tables are priors for which arms to try. They are not results for this corpus.
The registered model cards include
[MiniLM](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
[BGE small](https://huggingface.co/BAAI/bge-small-en-v1.5),
[E5 small](https://huggingface.co/intfloat/e5-small-v2),
[Nomic Embed Text](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5),
[ModernBERT Embed](https://huggingface.co/nomic-ai/modernbert-embed-base),
[Snowflake Arctic Embed](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0),
[Qwen3 Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B),
[BGE-M3](https://huggingface.co/BAAI/bge-m3),
[Jina code](https://huggingface.co/jinaai/jina-embeddings-v2-base-code),
[CodeRankEmbed](https://huggingface.co/nomic-ai/CodeRankEmbed), and
[UniXcoder](https://huggingface.co/microsoft/unixcoder-base). Each trial still resolves an
immutable revision rather than treating the mutable model-card URL as an artifact identity.

## Blocking and LSH families

Exact blocking keys should be evaluated before approximate hashing. The registry includes identity,
operation kind, action/object/artifact, workflow/domain, input/output contract, dependency family,
and multi-key cascade options with union, intersection, minimum-match, fallback, and learned
blocking logic.

| Family | Appropriate input | Required comparison |
|---|---|---|
| MinHash threshold | Sets of normalized word, character, AST, or call shingles | Jaccard threshold, permutations, blocking recall, bucket skew |
| MinHash Forest | Set shingles when a top-K approximate neighbor query is desired | Recall and candidate count against exact Jaccard |
| MinHash Ensemble | Unequal-size sets where containment matters | Containment recall across size strata |
| Weighted MinHash | Weighted term or feature multisets | Weighted-Jaccard oracle and weight normalization |
| SimHash 64/128 | Near-duplicate text or feature vectors | Hamming radius, collision rate, and false exclusion |
| Random-hyperplane LSH | Dense vectors under cosine similarity | Exact-cosine recall and number of probes |
| Faiss LSH | Binary codes | Hamming search against the exact vector/index control |
| AST/call MinHash | Structural shingles | Package-family and version-drift splits |

The [`datasketch` MinHash LSH documentation](https://ekzhu.com/datasketch/lsh.html) is the source
for the threshold index; its Forest and Ensemble variants solve different query contracts and
must not be treated as interchangeable settings of one algorithm.

Blocking has one hard obligation: preserve relevant supply. A predeclared promotion threshold such
as at least 99.5% reviewed blocking recall may be useful, but its value is a project decision to be
measured on frozen tasks, not a property of any LSH family.

## Storage, lexical search, and vector indexes

Keep canonical observations and derived features in the normalized SQLite/Parquet artifacts.
Search systems are disposable, versioned projections:

- [SQLite FTS5](https://www.sqlite.org/fts5.html) for local fielded lexical baselines;
- exact NumPy and Faiss Flat for dense-quality and ANN-recall oracles;
- Faiss HNSW, IVF Flat, IVF-PQ, and binary LSH for bounded index tradeoffs;
- [Qdrant hybrid queries](https://qdrant.tech/documentation/search/hybrid-queries/) for a local
  service-backed sparse/dense/multistage arm;
- [LanceDB](https://docs.lancedb.com/) for an embedded vector/table arm.

[Faiss documents](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes) that Flat search is
exhaustive, while HNSW and IVF introduce accuracy/memory/effort tradeoffs and PQ introduces
compression. Accordingly, every ANN trial records build/training parameters, index digest and
bytes, `efSearch` or `nprobe`, cold and warm latency, update cost, and recall against the exact
index. A larger system is not assumed to be better; SQLite plus exact vectors may remain the right
answer at the reviewed-corpus scale.

Do not duplicate canonical facts into multiple authorities. Every FTS row, vector, bucket, or
graph edge points back to the stable package-release/operation/feature identifier and generator
revision.

## Benchmark supply and metrics

The intended confirmation corpus is 100–250 deeply extracted packages and 300–500 reviewed tasks;
those numbers are a target, not a completed dataset. Include:

- direct operation tasks, multi-operation routes, and multiple acceptable operation sets;
- hard negatives matched on vocabulary but wrong on type, version, platform, effect, or license;
- explicit no-valid-reuse tasks and tasks that require a novel residual;
- paraphrase, package-family, temporal release, major-version, long-tail, and source-lineage splits;
- graded relevance judgments for useful-but-not-final candidates, with exact route requirements
  kept separately.

Keep metric families separate:

| Layer | Measures |
|---|---|
| Blocking | Reviewed blocking recall, candidate reduction, false exclusion, latency, bucket size/skew |
| Operation retrieval | Recall@K, MRR, nDCG@K, compatible Recall@K, hard-negative rank |
| Route retrieval | Required-node/edge recall, adapter recall, dependency completeness, route edit distance |
| Reuse/abstention | PR-AUC, F1, Brier score, calibration error, false reuse, missed reuse, no-route accuracy |
| Compatibility | Version/type/schema/artifact/effect checks, unknown/rejected/accepted counts |
| Planning | Valid structured output, allowlist adherence, correct operation/arguments/order, repair count |
| Execution | Install/import/run/contract/route/independent-acceptance rates, clean replay |
| Efficiency | Build/query/execution latency, CPU/GPU time, peak memory, index bytes, model calls/tokens, verified cost or null |

The end-to-end headline is the failure-inclusive resource vector to the first independently accepted
result. Retrieval metrics remain visible even when execution fails, and an accepted execution does
not erase a bad hard-negative or abstention result.

## Receipts and evidence boundaries

Every measured run should retain:

- registry, schedule configuration, stage-configuration, artifact-cache, corpus, task, split,
  environment, and source-code digests;
- package release/artifact identities and evidence floor;
- model ID, resolved revision, local artifact digest, tokenizer/processor, dtype, quantization,
  instructions, pooling, output dimension, runtime, and server/build revision;
- random seeds, task/package limits, retries, timeouts, candidate budget, context budget, and
  verification depth;
- per-channel candidates, raw scores/ranks, fusion/rerank decisions, compatibility rejections, and
  planner payload;
- index parameters, bytes, cold/warm timings, peak host/GPU memory, calls/tokens, and exceptions;
- selected/called/produced/consumed/contributing states and the independent acceptance result.

Evidence remains layered:

- parsed metadata and source observations are facts about a named artifact;
- deterministic tags are reproducible derived features;
- NLP- and LLM-generated descriptions, labels, examples, problems, solutions, or edges are
  candidates with generator provenance and review state;
- similarity and retrieval rank are evidence for consideration, not compatibility;
- compatibility checks can reject a route but cannot prove behavior that was never executed;
- only a replayed, independently evaluated execution receipt establishes an accepted route.

Failed downloads, load errors, NaN vectors, empty outputs, schema violations, timeouts, rejected
routes, and unfavorable metrics remain in the experiment ledger. Removing them would make runtime
and model comparisons look better than the system actually performed.
