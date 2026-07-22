"""Measured qualification helpers for embedding and local generation backends.

These helpers are intentionally separate from retrieval scoring.  A backend can load and emit valid
vectors or JSON while still being a poor retriever or planner.  Qualification receipts establish
only runtime facts: exact revision, instructions/template, device, precision, finite outputs,
latency, peak accelerator memory, parsing, and deterministic allowlist validation.

Optional heavyweight libraries are imported inside the functions that need them.  The base package
therefore remains small and deterministic-only installations can use the experiment designer.
"""

from __future__ import annotations

from contextlib import suppress
from datetime import datetime, timezone
import gc
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Mapping
from urllib import request as urllib_request
from urllib.parse import urlparse


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _optional_version(name: str) -> str | None:
    with suppress(importlib_metadata.PackageNotFoundError):
        return importlib_metadata.version(name)
    return None


def _write_receipt(output_path: str | Path | None, receipt: Mapping[str, object]) -> None:
    if output_path is None:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def hardware_snapshot() -> dict[str, object]:
    """Capture enough runtime detail to distinguish Kaggle accelerator environments."""

    snapshot: dict[str, object] = {
        "captured_at": _utc_now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "libraries": {
            name: _optional_version(name)
            for name in (
                "torch",
                "transformers",
                "sentence-transformers",
                "accelerate",
                "bitsandbytes",
                "unsloth",
                "huggingface-hub",
            )
        },
        "cuda": {"available": False},
    }
    try:
        import torch

        cuda: dict[str, object] = {
            "available": bool(torch.cuda.is_available()),
            "torch_cuda_version": torch.version.cuda,
        }
        if torch.cuda.is_available():
            cuda.update(
                {
                    "device_count": torch.cuda.device_count(),
                    "devices": [
                        {
                            "index": index,
                            "name": torch.cuda.get_device_name(index),
                            "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                            "compute_capability": list(torch.cuda.get_device_capability(index)),
                        }
                        for index in range(torch.cuda.device_count())
                    ],
                    "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                }
            )
        snapshot["cuda"] = cuda
    except Exception as error:  # optional runtime probe must preserve failures
        snapshot["cuda"] = {"available": False, "probe_error": repr(error)}

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        snapshot["nvidia_smi"] = {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as error:
        snapshot["nvidia_smi"] = {"probe_error": repr(error)}
    snapshot["snapshot_digest"] = _digest(snapshot)
    return snapshot


def _resolve_huggingface_revision(model_id: str, requested_revision: str | None) -> str | None:
    if requested_revision is not None and len(requested_revision) == 40 and all(
        character in "0123456789abcdefABCDEF" for character in requested_revision
    ):
        return requested_revision.lower()
    try:
        from huggingface_hub import model_info

        resolved = str(model_info(model_id, revision=requested_revision).sha)
        if len(resolved) != 40 or any(
            character not in "0123456789abcdefABCDEF" for character in resolved
        ):
            raise ValueError(f"Hub returned a non-immutable revision: {resolved!r}")
        return resolved.lower()
    except Exception as error:
        raise RuntimeError(
            f"could not resolve an immutable Hugging Face revision for {model_id!r}"
        ) from error


def _normalize_rows(matrix):
    import numpy as np

    values = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(values).all():
        raise ValueError("embedding matrix contains NaN or infinity")
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("embedding matrix contains a zero or invalid norm")
    return values / norms


def _embedding_prefixes(model_id: str, instruction: str) -> tuple[str, str]:
    supported_instructions = {
        "none",
        "model_default_asymmetric",
        "intentionally_swapped_control",
        "natural_language_to_code",
        "capability_retrieval_task",
        "generic_search_query_document",
    }
    if instruction not in supported_instructions:
        raise ValueError(f"unsupported embedding instruction: {instruction}")
    lowered = model_id.casefold()
    if instruction == "none":
        return "", ""
    if instruction == "intentionally_swapped_control":
        query, document = _embedding_prefixes(model_id, "model_default_asymmetric")
        return document, query
    if instruction == "natural_language_to_code":
        if "jina-code-embeddings" in lowered:
            return (
                "Find the most relevant code snippet given the following query:\n",
                "Candidate code snippet:\n",
            )
        if "coderankembed" in lowered:
            return "Represent this query for searching relevant code: ", ""
        if "qwen3-embedding" in lowered:
            return (
                "Instruct: Retrieve Python code capabilities that solve the user request\nQuery: ",
                "",
            )
        return "query: ", "passage: "
    if instruction == "capability_retrieval_task":
        return (
            "Instruct: Find a reusable Python package operation with compatible inputs and outputs\nQuery: ",
            "",
        )
    if instruction == "generic_search_query_document":
        return "search_query: ", "search_document: "
    if "e5-" in lowered:
        return "query: ", "passage: "
    if "nomic-embed" in lowered or "modernbert-embed" in lowered:
        return "search_query: ", "search_document: "
    if "bge-small-en" in lowered:
        return "Represent this sentence for searching relevant passages: ", ""
    if "coderankembed" in lowered:
        return "Represent this query for searching relevant code: ", ""
    if "jina-code-embeddings" in lowered:
        return (
            "Find the most relevant code snippet given the following query:\n",
            "Candidate code snippet:\n",
        )
    if "qwen3-embedding" in lowered:
        return (
            "Instruct: Retrieve Python code capabilities that solve the user request\nQuery: ",
            "",
        )
    return "", ""


def _encode_one_embedding(
    model,
    text: str,
    *,
    role: str,
    use_native_asymmetric_methods: bool,
):
    """Encode one row while preserving a model's query/document contract when available."""

    method_name = f"encode_{role}"
    method = getattr(model, method_name, None) if use_native_asymmetric_methods else None
    if method is None:
        method = model.encode
    return method(
        [text],
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )[0]


_EMBEDDING_SMOKE_QUERIES = (
    "load a comma-separated file into a pandas dataframe",
    "fill missing numeric values with each column median",
    "train a gradient-boosted binary classifier",
    "calculate receiver operating characteristic area under the curve",
    "serialize records to a JSON file",
    "draw a line chart from a dataframe",
)

_EMBEDDING_SMOKE_DOCUMENTS = (
    "pandas.read_csv reads a comma-separated values file into a DataFrame.",
    "pandas.DataFrame.fillna replaces missing values; medians can be supplied per numeric column.",
    "sklearn.ensemble.HistGradientBoostingClassifier fits a histogram gradient boosting classifier.",
    "sklearn.metrics.roc_auc_score computes the area under the receiver operating characteristic curve.",
    "json.dump serializes a Python object as JSON to a file-like object.",
    "pandas.DataFrame.plot.line draws a line plot from dataframe columns.",
)


def _validate_matryoshka_dimension(model_id: str, dimension: int, native_dimension: int) -> None:
    lowered = model_id.casefold()
    registered: tuple[tuple[str, set[int]], ...] = (
        ("embeddinggemma", {128, 256, 512}),
        ("qwen3-embedding", {64, 128, 256, 512}),
        ("nomic-embed-text", {64, 128, 256, 512}),
        ("modernbert-embed", {256}),
        ("snowflake-arctic-embed", {256}),
        ("jina-code-embeddings-0.5b", {64, 128, 256, 512}),
        ("jina-code-embeddings-1.5b", {128, 256, 512}),
    )
    for fragment, dimensions in registered:
        if fragment in lowered:
            if dimension not in dimensions or dimension >= native_dimension:
                raise ValueError(
                    f"dimension {dimension} is not a registered Matryoshka output for "
                    f"{model_id!r} with native dimension {native_dimension}"
                )
            return
    raise ValueError(
        f"dimension truncation is not enabled for unregistered model {model_id!r}"
    )


def qualify_embedding_model(
    model_id: str,
    *,
    revision: str | None = None,
    instruction: str = "model_default_asymmetric",
    precision: str = "fp32",
    truncate_dimension: int | None = None,
    device: str | None = None,
    cache_dir: str | Path | None = None,
    trust_remote_code: bool = False,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    """Load one embedding model, run a finite-vector/top-1 smoke check, and write a receipt."""

    started = time.perf_counter()
    receipt: dict[str, object] = {
        "schema_version": "1",
        "qualification_kind": "embedding_runtime_smoke",
        "status": "started",
        "model_id": model_id,
        "requested_revision": revision,
        "instruction": instruction,
        "precision": precision,
        "truncate_dimension": truncate_dimension,
        "trust_remote_code": trust_remote_code,
        "synthetic_smoke_only": True,
        "hardware": hardware_snapshot(),
    }
    model = None
    try:
        if precision not in {"fp32", "fp16"}:
            raise ValueError(
                "runtime activation precision must be fp32 or fp16; int8, uint8, and binary "
                "are separate output-storage/index experiments"
            )
        if "embeddinggemma" in model_id.casefold() and precision == "fp16":
            raise ValueError("EmbeddingGemma FP16 activation qualification is prohibited")
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer

        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if precision == "fp16" and selected_device == "cpu":
            raise ValueError("FP16 activation qualification requires a CUDA device")
        query_prefix, document_prefix = _embedding_prefixes(model_id, instruction)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        resolved_revision = _resolve_huggingface_revision(model_id, revision)
        model = SentenceTransformer(
            model_id,
            revision=resolved_revision,
            device=selected_device,
            cache_folder=str(cache_dir) if cache_dir is not None else None,
            token=os.getenv("HF_TOKEN") or None,
            trust_remote_code=trust_remote_code,
        )
        if precision == "fp16":
            model.half()
        else:
            model.float()
        use_native_asymmetric_methods = instruction in {
            "model_default_asymmetric",
            "intentionally_swapped_control",
        } and all(hasattr(model, name) for name in ("encode_query", "encode_document"))
        query_role = "document" if instruction == "intentionally_swapped_control" else "query"
        document_role = "query" if instruction == "intentionally_swapped_control" else "document"
        queries = [
            text if use_native_asymmetric_methods else query_prefix + text
            for text in _EMBEDDING_SMOKE_QUERIES
        ]
        documents = [
            text if use_native_asymmetric_methods else document_prefix + text
            for text in _EMBEDDING_SMOKE_DOCUMENTS
        ]

        # Individual encoding avoids mixed-length padding being mistaken for a valid runtime on a
        # model/backend pair.  Throughput batching is a later, separately measured experiment.
        query_vectors = np.vstack(
            [
                _encode_one_embedding(
                    model,
                    text,
                    role=query_role,
                    use_native_asymmetric_methods=use_native_asymmetric_methods,
                )
                for text in queries
            ]
        )
        document_vectors = np.vstack(
            [
                _encode_one_embedding(
                    model,
                    text,
                    role=document_role,
                    use_native_asymmetric_methods=use_native_asymmetric_methods,
                )
                for text in documents
            ]
        )
        native_dimension = int(query_vectors.shape[1])
        if truncate_dimension is not None:
            if truncate_dimension <= 0 or truncate_dimension > native_dimension:
                raise ValueError(
                    f"invalid truncate_dimension={truncate_dimension} for native dimension "
                    f"{native_dimension}"
                )
            if truncate_dimension < native_dimension:
                _validate_matryoshka_dimension(
                    model_id,
                    truncate_dimension,
                    native_dimension,
                )
            query_vectors = query_vectors[:, :truncate_dimension]
            document_vectors = document_vectors[:, :truncate_dimension]
        query_vectors = _normalize_rows(query_vectors)
        document_vectors = _normalize_rows(document_vectors)
        similarities = query_vectors @ document_vectors.T
        top_ids = similarities.argmax(axis=1)
        top1_accuracy = float(np.mean(top_ids == np.arange(len(queries))))
        output_directory = Path(output_path).parent if output_path is not None else None
        vector_files: dict[str, object] = {}
        if output_directory is not None:
            output_directory.mkdir(parents=True, exist_ok=True)
            vector_suffix = _digest(
                (
                    model_id,
                    resolved_revision,
                    instruction,
                    precision,
                    truncate_dimension,
                )
            ).split(":", 1)[1][:12]
            query_path = output_directory / f"qualification-query-vectors-{vector_suffix}.npy"
            document_path = (
                output_directory / f"qualification-document-vectors-{vector_suffix}.npy"
            )
            np.save(query_path, query_vectors)
            np.save(document_path, document_vectors)
            vector_files = {
                "query_vectors": str(query_path),
                "document_vectors": str(document_path),
                "query_digest": "sha256:"
                + hashlib.sha256(query_vectors.tobytes(order="C")).hexdigest(),
                "document_digest": "sha256:"
                + hashlib.sha256(document_vectors.tobytes(order="C")).hexdigest(),
            }
        receipt.update(
            {
                "status": "measured",
                "resolved_revision": resolved_revision,
                "device": selected_device,
                "native_dimension": native_dimension,
                "output_dimension": int(query_vectors.shape[1]),
                "query_count": len(queries),
                "document_count": len(documents),
                "query_prefix_digest": _digest(query_prefix),
                "document_prefix_digest": _digest(document_prefix),
                "encoder_method": (
                    "encode_query_and_encode_document"
                    if use_native_asymmetric_methods
                    else "encode_with_explicit_prefixes"
                ),
                "finite_vectors": True,
                "nonzero_vectors": True,
                "synthetic_top1_accuracy": top1_accuracy,
                "synthetic_top1_ids": [int(value) for value in top_ids],
                "vector_files": vector_files,
                "peak_cuda_memory_bytes": (
                    int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
                ),
                "peak_cuda_reserved_bytes": (
                    int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0
                ),
            }
        )
    except Exception as error:
        receipt.update({"status": "failed", "error": repr(error)})
    finally:
        receipt["elapsed_seconds"] = time.perf_counter() - started
        receipt["completed_at"] = _utc_now()
        receipt["receipt_digest"] = _digest(
            {key: value for key, value in receipt.items() if key != "receipt_digest"}
        )
        _write_receipt(output_path, receipt)
        if model is not None:
            del model
        gc.collect()
        with suppress(Exception):
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return receipt


def extract_first_json_object(text: str) -> dict[str, object]:
    """Return the first syntactically valid JSON object without repairing model output."""

    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        with suppress(json.JSONDecodeError):
            value, _ = decoder.raw_decode(text[index:])
            if isinstance(value, dict):
                return value
    raise ValueError("model output did not contain a JSON object")


def _validate_route_payload(payload: Mapping[str, object], allowed_ids: set[str]) -> tuple[bool, str]:
    if set(payload) - {"decision", "operation_ids", "confidence", "rationale"}:
        return False, "unknown keys"
    if payload.get("decision") not in {"reuse", "no_reuse", "abstain"}:
        return False, "invalid decision"
    operation_ids = payload.get("operation_ids")
    if not isinstance(operation_ids, list) or not all(isinstance(item, str) for item in operation_ids):
        return False, "operation_ids must be a string array"
    if len(operation_ids) != len(set(operation_ids)):
        return False, "operation_ids must not contain duplicates"
    if set(operation_ids) - allowed_ids:
        return False, "operation_ids contain values outside the allowlist"
    confidence = payload.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return False, "confidence must be a JSON number"
    if not math.isfinite(float(confidence)) or not 0 <= float(confidence) <= 1:
        return False, "confidence must lie in [0, 1]"
    rationale = payload.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        return False, "rationale must be a nonempty string"
    if payload["decision"] == "reuse" and not operation_ids:
        return False, "reuse decisions require at least one operation"
    if payload["decision"] != "reuse" and operation_ids:
        return False, "non-reuse decisions cannot contain operations"
    return True, "valid"


_GENERATOR_ALLOWED = {
    "pandas.read_csv",
    "pandas.DataFrame.fillna",
    "sklearn.model_selection.cross_val_score",
    "sklearn.ensemble.HistGradientBoostingClassifier",
    "sklearn.metrics.roc_auc_score",
}


def _generator_instruction() -> str:
    return (
        "Select only IDs from the allowlist to solve the request. Return exactly one JSON object "
        "with keys decision, operation_ids, confidence, rationale. decision is reuse, no_reuse, "
        "or abstain. Do not invent packages, symbols, adapters, or code."
    )


def _generator_user_payload() -> dict[str, object]:
    return {
        "request": "Load train.csv, fill missing values, train a binary classifier with cross-validation, and report ROC-AUC.",
        "allowed_operation_ids": sorted(_GENERATOR_ALLOWED),
    }


def _transformers_text_generation(
    model_id: str,
    *,
    revision: str | None,
    quantization: str,
    max_new_tokens: int,
    thinking: bool,
    use_cache: bool,
    cache_dir: str | Path | None,
) -> tuple[str, dict[str, object], object, object]:
    import torch

    token = os.getenv("HF_TOKEN") or None
    model_kwargs: dict[str, object] = {
        "revision": revision,
        "device_map": "auto",
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "token": token,
    }
    if quantization == "fp16":
        if not torch.cuda.is_available():
            raise ValueError(
                "Transformers FP16 qualification requires CUDA; refusing to relabel a CPU "
                "FP32 load as FP16"
            )
        model_kwargs["dtype"] = torch.float16
    elif quantization in {"bitsandbytes_int8", "bitsandbytes_nf4"}:
        from transformers import BitsAndBytesConfig

        if quantization == "bitsandbytes_int8":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=(
                    torch.bfloat16
                    if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                    else torch.float16
                ),
            )
    else:
        raise ValueError(f"unsupported Transformers quantization: {quantization}")

    messages = [
        {"role": "system", "content": _generator_instruction()},
        {"role": "user", "content": _canonical_json(_generator_user_payload())},
    ]
    if "gemma-4" in model_id.casefold():
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(
            model_id,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            token=token,
        )
        model = AutoModelForMultimodalLM.from_pretrained(model_id, **model_kwargs)
        encoded = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=thinking,
        ).to(model.device)
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=use_cache,
        )
        new_tokens = generated[0, encoded["input_ids"].shape[1] :]
        output = processor.decode(new_tokens, skip_special_tokens=True)
        telemetry = {
            "input_tokens": int(encoded["input_ids"].shape[1]),
            "output_tokens": int(new_tokens.shape[0]),
            "chat_template_digest": _digest(
                getattr(processor, "chat_template", None)
                or getattr(getattr(processor, "tokenizer", None), "chat_template", None)
            ),
        }
        return output, telemetry, model, processor

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        token=token,
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    template_kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "return_dict": True,
    }
    if "qwen3" in model_id.casefold() or "smollm3" in model_id.casefold():
        template_kwargs["enable_thinking"] = thinking
    encoded = tokenizer.apply_chat_template(messages, **template_kwargs).to(model.device)
    generated = model.generate(
        **encoded,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=use_cache,
    )
    new_tokens = generated[0, encoded["input_ids"].shape[1] :]
    output = tokenizer.decode(new_tokens, skip_special_tokens=True)
    telemetry = {
        "input_tokens": int(encoded["input_ids"].shape[1]),
        "output_tokens": int(new_tokens.shape[0]),
        "chat_template_digest": _digest(getattr(tokenizer, "chat_template", None)),
    }
    return output, telemetry, model, tokenizer


def qualify_transformers_generator(
    model_id: str,
    *,
    revision: str | None = None,
    quantization: str = "fp16",
    max_new_tokens: int = 256,
    thinking: bool = False,
    cache_dir: str | Path | None = None,
    check_cache_parity: bool = True,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    """Run a strict allowlisted JSON-planning smoke check through Transformers."""

    started = time.perf_counter()
    receipt: dict[str, object] = {
        "schema_version": "1",
        "qualification_kind": "generator_runtime_smoke",
        "status": "started",
        "backend": "transformers",
        "model_id": model_id,
        "requested_revision": revision,
        "quantization": quantization,
        "thinking": thinking,
        "max_new_tokens": max_new_tokens,
        "synthetic_smoke_only": True,
        "allowed_operation_ids": sorted(_GENERATOR_ALLOWED),
        "hardware": hardware_snapshot(),
    }
    loaded: list[object] = []
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        resolved_revision = _resolve_huggingface_revision(model_id, revision)
        outputs: dict[str, object] = {}
        parsed_payloads: dict[str, object] = {}
        for use_cache in ((True, False) if check_cache_parity else (True,)):
            output, telemetry, model, processor = _transformers_text_generation(
                model_id,
                revision=resolved_revision,
                quantization=quantization,
                max_new_tokens=max_new_tokens,
                thinking=thinking,
                use_cache=use_cache,
                cache_dir=cache_dir,
            )
            loaded.extend((model, processor))
            payload = extract_first_json_object(output)
            valid, reason = _validate_route_payload(payload, _GENERATOR_ALLOWED)
            key = "use_cache_true" if use_cache else "use_cache_false"
            outputs[key] = {"raw_output": output, "telemetry": telemetry}
            parsed_payloads[key] = payload
            if not valid:
                raise ValueError(f"{key} payload validation failed: {reason}")
            # Loading twice at once can exceed a 16 GB device. Release immediately between parity
            # arms; each output retains its own measured telemetry.
            del model, processor
            loaded.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        parity = (
            parsed_payloads.get("use_cache_true") == parsed_payloads.get("use_cache_false")
            if check_cache_parity
            else None
        )
        receipt.update(
            {
                "status": "measured" if parity is not False else "failed_parity",
                "resolved_revision": resolved_revision,
                "outputs": outputs,
                "parsed_payloads": parsed_payloads,
                "cache_parity_checked": check_cache_parity,
                "cache_semantic_parity": parity,
                "peak_cuda_memory_bytes": (
                    int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
                ),
                "peak_cuda_reserved_bytes": (
                    int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0
                ),
            }
        )
    except Exception as error:
        receipt.update({"status": "failed", "error": repr(error)})
    finally:
        receipt["elapsed_seconds"] = time.perf_counter() - started
        receipt["completed_at"] = _utc_now()
        receipt["receipt_digest"] = _digest(
            {key: value for key, value in receipt.items() if key != "receipt_digest"}
        )
        _write_receipt(output_path, receipt)
        loaded.clear()
        gc.collect()
        with suppress(Exception):
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return receipt


def qualify_ollama_generator(
    model_tag: str,
    *,
    base_url: str = "http://127.0.0.1:11434",
    thinking: bool = False,
    timeout_seconds: float = 900,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    """Qualify an already-running local Ollama server through its JSON-schema API."""

    started = time.perf_counter()
    receipt: dict[str, object] = {
        "schema_version": "1",
        "qualification_kind": "generator_runtime_smoke",
        "status": "started",
        "backend": "ollama",
        "model_tag": model_tag,
        "base_url": base_url,
        "thinking": thinking,
        "synthetic_smoke_only": True,
        "hardware": hardware_snapshot(),
    }
    try:
        parsed_base_url = urlparse(base_url)
        if parsed_base_url.scheme != "http" or parsed_base_url.hostname not in {
            "127.0.0.1",
            "localhost",
            "::1",
        }:
            raise ValueError("Ollama qualification accepts only a loopback HTTP server")

        def ollama_request(path: str, body: Mapping[str, object] | None = None) -> object:
            request = urllib_request.Request(
                base_url.rstrip("/") + path,
                data=(
                    _canonical_json(body).encode("utf-8") if body is not None else None
                ),
                headers={"Content-Type": "application/json"},
                method="POST" if body is not None else "GET",
            )
            with urllib_request.urlopen(request, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))

        version_response = ollama_request("/api/version")
        tags_response = ollama_request("/api/tags")
        if not isinstance(tags_response, Mapping):
            raise ValueError("Ollama tags response was not an object")
        matching_models = [
            item
            for item in tags_response.get("models", [])
            if isinstance(item, Mapping)
            and (item.get("name") == model_tag or item.get("model") == model_tag)
        ]
        if len(matching_models) != 1:
            raise ValueError(
                f"expected one exact installed Ollama tag {model_tag!r}; found "
                f"{len(matching_models)}"
            )
        installed_model = matching_models[0]
        model_digest = str(installed_model.get("digest", "")).strip()
        if not model_digest:
            raise ValueError("installed Ollama model did not report an immutable digest")
        show_response = ollama_request("/api/show", {"model": model_tag, "verbose": False})
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": ["decision", "operation_ids", "confidence", "rationale"],
            "properties": {
                "decision": {"enum": ["reuse", "no_reuse", "abstain"]},
                "operation_ids": {
                    "type": "array",
                    "items": {"enum": sorted(_GENERATOR_ALLOWED)},
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "rationale": {"type": "string"},
            },
        }
        body = {
            "model": model_tag,
            "messages": [
                {"role": "system", "content": _generator_instruction()},
                {"role": "user", "content": _canonical_json(_generator_user_payload())},
            ],
            "stream": False,
            "format": schema,
            "think": thinking,
            "options": {"temperature": 0, "seed": 17},
        }
        result = ollama_request("/api/chat", body)
        if not isinstance(result, Mapping):
            raise ValueError("Ollama chat response was not an object")
        raw_output = str(result.get("message", {}).get("content", ""))
        payload = extract_first_json_object(raw_output)
        valid, reason = _validate_route_payload(payload, _GENERATOR_ALLOWED)
        if not valid:
            raise ValueError(f"Ollama payload validation failed: {reason}")
        receipt.update(
            {
                "status": "measured",
                "raw_output": raw_output,
                "parsed_payload": payload,
                "server_version": version_response,
                "model_digest": model_digest,
                "installed_model": dict(installed_model),
                "show_response_digest": _digest(show_response),
                "server_metrics": {
                    key: result.get(key)
                    for key in (
                        "total_duration",
                        "load_duration",
                        "prompt_eval_count",
                        "prompt_eval_duration",
                        "eval_count",
                        "eval_duration",
                    )
                },
            }
        )
    except Exception as error:
        receipt.update({"status": "failed", "error": repr(error)})
    finally:
        receipt["elapsed_seconds"] = time.perf_counter() - started
        receipt["completed_at"] = _utc_now()
        receipt["receipt_digest"] = _digest(
            {key: value for key, value in receipt.items() if key != "receipt_digest"}
        )
        _write_receipt(output_path, receipt)
    return receipt
