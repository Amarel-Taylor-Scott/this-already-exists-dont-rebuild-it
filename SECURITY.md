# Security model

Package artifacts and their metadata, documentation, examples, build scripts, and source code are
untrusted inputs.

- The static control plane must not import, install, build, or execute inspected packages.
- Archive readers must enforce explicit member-count, size, compression-ratio, traversal, nesting,
  and timeout limits.
- Dynamic analysis must run in disposable, credential-free, network-denied sandboxes with a
  read-only base, limited writable output, and CPU/memory/time/process quotas.
- Artifact inspection, build, and runtime evaluation are separate trust zones.
- Package names, generated descriptions, docs, and examples can contain prompt injection and must be
  treated as data rather than instructions.
- Exact artifact hashes, package versions, environment facts, calls, outputs, and logs are retained.
- PyPI provenance and an absence of known OSV advisories are useful evidence, not safety guarantees.

Do not report suspected malicious packages in public issues before coordinating with the relevant
registry or maintainer security process.

