# Research landscape — July 21, 2026

## What is already available

The project should integrate or benchmark existing primitives before implementing replacements.

| System | Existing capability | What remains to test/build here |
|---|---|---|
| [Context7](https://github.com/upstash/context7) | Current version-specific library documentation and examples | Function/class contracts, compatibility, adapters, route execution, and contribution evidence |
| [Tessl](https://docs.tessl.io/) | Self-described versioned context for 10,000+ OSS packages | Independently reproduce API-use results; test typed cross-package composition and executable reuse |
| [Griffe](https://mkdocstrings.github.io/griffe/) | Python API extraction, aliases, signatures, docs, serialization, breaking-change checks | Capability inference, runtime contracts, and accepted outcomes |
| [bandersnatch](https://github.com/pypa/bandersnatch) | Standards-compliant PyPI mirroring | Frozen sampling, operation extraction, and benchmark evidence |
| [SCIP](https://github.com/scip-code/scip), [scip-python](https://github.com/sourcegraph/scip-python), [Glean](https://github.com/facebookincubator/Glean) | Symbols, definitions/references, calls, and reusable code-index formats | Purpose, artifacts, runtime behavior, policy, adapters, and route validation |
| [Zoekt](https://github.com/sourcegraph/zoekt) | Fast trigram code search | One candidate lane, not the complete reuse decision |
| [deps.dev](https://docs.deps.dev/api/v3/), [OSV](https://google.github.io/osv.dev/api/), [GUAC](https://github.com/guacsec/guac) | Dependency/license/advisory/supply-chain graph data | Callable API truth and execution compatibility |
| [OpenSSF Package Analysis](https://github.com/ossf/package-analysis) | Sandboxed dynamic package behavior observation | Integrate dynamic evidence with operation contracts and routes |

GitHub's [Stack Graphs](https://github.com/github/stack-graphs) was archived in September 2025; it is
useful research, not a preferred new dependency.

## Benchmarks that cover parts of the problem

- [ToolRet](https://arxiv.org/abs/2503.01763) contains about 7,600 retrieval tasks and 43,000 tools;
  it shows that ordinary retrieval strength does not automatically transfer to tool retrieval.
- [CodeFlowBench](https://arxiv.org/abs/2504.21751) evaluates implementing later behavior through
  reuse of earlier functions over dependency structures.
- [SWE-ContextBench](https://arxiv.org/abs/2602.08316) tests reuse of related prior experience and
  reports that relevant compact experience can help while irrelevant experience can hurt.
- [ContextBench](https://arxiv.org/abs/2602.05892) separates repository context retrieval from final
  issue success and measures explored versus actually used context.
- [DI-BENCH](https://arxiv.org/abs/2501.13699) evaluates dependency inference on 581 testable
  repositories and reports a large gap between plausible dependency text and executable projects.
- [GitChameleon 2.0](https://arxiv.org/abs/2507.12367) evaluates version-aware Python API use.
- [CodeSearchNet](https://github.com/github/CodeSearchNet),
  [CoIR](https://github.com/coir-team/coir),
  [RepoBench](https://github.com/Leolty/repobench), and
  [CodeRAG-Bench](https://github.com/code-rag-bench/code-rag-bench) provide useful code retrieval
  tasks and corpora.
- [DS-1000](https://github.com/xlang-ai/DS-1000) and
  [ODEX](https://code-eval.github.io/) test practical library use, but not the complete open-catalog
  reuse, compatibility, route, and contribution chain.

No single benchmark measures the full target:

```text
reuse recognition
→ operation/version retrieval
→ hard-negative rejection
→ typed composition and adapter selection
→ dependency/environment resolution
→ execution and independent acceptance
→ contributing reuse and residual-code accounting
```

## Research claim discipline

The supplied design documents are specifications, not empirical evidence. The repository must not
claim any of the following until a frozen run produces receipts and uncertainty estimates:

- a compact model matches or beats a frontier router;
- operation cards accurately represent PyPI behavior at scale;
- static metadata is enough to decide cross-package compatibility;
- embeddings reduce total failure-inclusive cost;
- contribution can be inferred merely from a call trace;
- solving one novel gap makes a later independent task cheaper;
- a public package's documentation, examples, or code can be redistributed.

## Key primary-source constraints

- The [PyPI JSON API](https://docs.pypi.org/api/json/) exposes useful metadata and known
  vulnerabilities but warns that metadata is supplied at upload time and may not match artifacts.
- PyPI directs download-volume work to its
  [BigQuery datasets](https://docs.pypi.org/api/bigquery/); JSON download fields are deprecated.
- Python packaging's [Core Metadata](https://packaging.python.org/specifications/core-metadata/)
  supports SPDX `License-Expression`, but the expression applies to the distribution archive, not
  automatically every related project artifact.
- [`pip inspect`](https://pip.pypa.io/en/stable/cli/pip_inspect/) emits a stable JSON environment
  report. `pip install --dry-run --ignore-installed --report` can resolve without installing, but a
  resolver result still does not prove runtime behavior.
- [OSV's batch API](https://google.github.io/osv.dev/post-v1-querybatch/) supports package-version
  vulnerability enrichment. An advisory absence is not proof of safety.
- PyPI provenance/attestations establish origin, not benign behavior.

