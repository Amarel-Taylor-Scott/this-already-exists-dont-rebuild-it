from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    "01_kaggle_build_capability_catalog.ipynb",
    "02_kaggle_generate_search_features.ipynb",
    "03_kaggle_small_model_routing_lab.ipynb",
    "04_kaggle_conditional_retrieval_model_sweep.ipynb",
)


def _python_without_ipython_commands(source: str) -> str:
    transformed: list[str] = []
    for line in source.splitlines():
        if line.lstrip().startswith(("!", "%")):
            indentation = line[: len(line) - len(line.lstrip())]
            transformed.append(indentation + "pass")
        else:
            transformed.append(line)
    return "\n".join(transformed)


def test_kaggle_notebooks_are_clean_runnable_sources() -> None:
    for name in NOTEBOOKS:
        path = ROOT / "notebooks" / name
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        cell_ids = [cell["id"] for cell in notebook["cells"] if "id" in cell]
        assert len(cell_ids) == len(set(cell_ids))
        code = "\n".join(
            "".join(cell.get("source", ()))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        assert "!pip install" in code
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            assert cell.get("outputs", []) == []
            assert cell.get("execution_count") is None
            ast.parse(_python_without_ipython_commands("".join(cell.get("source", ()))))


def test_notebooks_cover_symbol_features_and_constrained_routing() -> None:
    feature_notebook = (ROOT / "notebooks" / NOTEBOOKS[1]).read_text(encoding="utf-8")
    routing_notebook = (ROOT / "notebooks" / NOTEBOOKS[2]).read_text(encoding="utf-8")
    sweep_notebook = (ROOT / "notebooks" / NOTEBOOKS[3]).read_text(encoding="utf-8")

    for required in ("feature_kind", "entity_id", "MinHash", "embedding"):
        assert required in feature_notebook
    for required in ("GoalIR", "RecipeIR", "no_reuse", "compatibility"):
        assert required in routing_notebook
    for required in (
        "mixed_screen",
        "schedule_first_round",
        "qualify_embedding_model",
        "qualify_transformers_generator",
        "qualify_ollama_generator",
        "synthetic_smoke_only",
    ):
        assert required in sweep_notebook
