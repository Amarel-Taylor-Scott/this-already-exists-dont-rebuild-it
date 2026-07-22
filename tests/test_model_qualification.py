from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from existing_code_reuse.model_qualification import (
    _GENERATOR_ALLOWED,
    _embedding_prefixes,
    _resolve_huggingface_revision,
    _transformers_text_generation,
    _validate_route_payload,
    extract_first_json_object,
    qualify_embedding_model,
    qualify_ollama_generator,
)


def test_extract_first_json_object_does_not_repair_invalid_prefix() -> None:
    value = extract_first_json_object('not json {broken} then {"decision":"abstain"} trailing')
    assert value == {"decision": "abstain"}


def test_embedding_instruction_names_and_pinned_revisions_are_strict() -> None:
    with pytest.raises(ValueError, match="unsupported embedding instruction"):
        _embedding_prefixes("example/model", "invented_instruction")

    revision = "a" * 40
    assert _resolve_huggingface_revision("example/model", revision) == revision
    assert _embedding_prefixes(
        "nomic-ai/CodeRankEmbed", "natural_language_to_code"
    ) == ("Represent this query for searching relevant code: ", "")
    assert _embedding_prefixes(
        "jinaai/jina-code-embeddings-0.5b", "natural_language_to_code"
    ) == (
        "Find the most relevant code snippet given the following query:\n",
        "Candidate code snippet:\n",
    )


def test_route_payload_is_allowlisted_and_fail_closed() -> None:
    valid = {
        "decision": "reuse",
        "operation_ids": ["pandas.read_csv"],
        "confidence": 0.8,
        "rationale": "The selected operation matches the requested input.",
    }
    assert _validate_route_payload(valid, _GENERATOR_ALLOWED) == (True, "valid")

    invented = {**valid, "operation_ids": ["invented.package.operation"]}
    accepted, reason = _validate_route_payload(invented, _GENERATOR_ALLOWED)
    assert not accepted
    assert "allowlist" in reason

    extra_key = {**valid, "code": "import pandas"}
    accepted, reason = _validate_route_payload(extra_key, _GENERATOR_ALLOWED)
    assert not accepted
    assert reason == "unknown keys"

    empty_reuse = {**valid, "operation_ids": []}
    assert _validate_route_payload(empty_reuse, _GENERATOR_ALLOWED) == (
        False,
        "reuse decisions require at least one operation",
    )

    string_confidence = {**valid, "confidence": "0.8"}
    assert _validate_route_payload(string_confidence, _GENERATOR_ALLOWED) == (
        False,
        "confidence must be a JSON number",
    )

    boolean_confidence = {**valid, "confidence": True}
    assert _validate_route_payload(boolean_confidence, _GENERATOR_ALLOWED) == (
        False,
        "confidence must be a JSON number",
    )


def test_unsupported_embedding_precision_is_a_preserved_failure(tmp_path) -> None:
    output = tmp_path / "receipt.json"
    receipt = qualify_embedding_model(
        "unused/model",
        precision="binary",
        output_path=output,
    )
    assert receipt["status"] == "failed"
    assert "output-storage/index experiments" in str(receipt["error"])
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "failed"


def test_embeddinggemma_fp16_is_rejected_before_model_download(tmp_path) -> None:
    receipt = qualify_embedding_model(
        "google/embeddinggemma-300m",
        precision="fp16",
        output_path=tmp_path / "embeddinggemma.json",
    )
    assert receipt["status"] == "failed"
    assert "prohibited" in str(receipt["error"])


def test_transformers_fp16_does_not_silently_fall_back_to_cpu(monkeypatch) -> None:
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        float16=object(),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    with pytest.raises(ValueError, match="refusing to relabel"):
        _transformers_text_generation(
            "unused/model",
            revision="a" * 40,
            quantization="fp16",
            max_new_tokens=1,
            thinking=False,
            use_cache=True,
            cache_dir=None,
        )


def test_ollama_qualification_is_limited_to_loopback(tmp_path) -> None:
    receipt = qualify_ollama_generator(
        "example:model",
        base_url="https://remote.example.invalid",
        output_path=tmp_path / "ollama.json",
    )
    assert receipt["status"] == "failed"
    assert "loopback" in str(receipt["error"])
