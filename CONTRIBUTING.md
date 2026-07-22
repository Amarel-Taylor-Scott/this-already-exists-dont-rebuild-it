# Contributing

Contributions should add executable evidence or a reproducible experiment rather than inflate the
catalog count.

Before proposing a new component, search the repository and the adjacent systems listed in
`docs/research_landscape.md`. Prefer an adapter to a mature existing project when it supplies the
needed primitive.

For code changes:

1. keep observed facts separate from generated/derived values;
2. add generator/schema versions and provenance for new records;
3. add a hard negative or failure case with every new compatibility rule;
4. run `uv run pytest` and `uv run ruff check .`;
5. do not commit package archives, model weights, indexes, generated databases, or credentials;
6. state what the result proves and what it does not prove.

