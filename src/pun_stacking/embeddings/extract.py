import gc
import json
from pathlib import Path

import numpy as np
import sentence_transformers
import torch
from sentence_transformers import SentenceTransformer

from pun_stacking.data.audit import load_jsonl
from pun_stacking.embeddings.prompts import input_prompt_for_model
from pun_stacking.graphs.runner import load_experiment_config

SPLITS = ("train", "validation", "test")


def model_alias(model_name: str) -> str:
    return model_name.rsplit("/", 1)[-1].lower()


def load_canonical_corpus(data_root: Path) -> list[dict]:
    rows = []

    for split in SPLITS:
        rows.extend(load_jsonl(data_root / "original" / f"{split}.jsonl"))

    if len(rows) != 5700:
        raise ValueError("Expected 5700 corpus instances")

    identifiers = [row["id"] for row in rows]

    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate identifiers in the original corpus")

    return sorted(rows, key=lambda row: row["id"])


def write_json(path: Path, content: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(path.suffix + ".tmp")

    temporary.write_text(
        json.dumps(content, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def resolve_models(
    configured_models: list[str],
    selected_model: str | None,
) -> list[str]:
    if not selected_model:
        return configured_models

    matches = [name for name in configured_models if selected_model in (name, model_alias(name))]

    if len(matches) != 1:
        raise ValueError(f"Unknown or ambiguous embedding model: {selected_model}")

    return matches


def encode_model(
    model_name: str,
    texts: list[str],
    settings: dict,
) -> tuple[np.ndarray, int]:
    if settings["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    model = SentenceTransformer(
        model_name,
        device=settings["device"],
    )

    try:
        native_max_length = model.max_seq_length
        requested_max_length = settings["max_length"]

        if isinstance(native_max_length, int) and native_max_length > 0:
            effective_max_length = min(
                native_max_length,
                requested_max_length,
            )
        else:
            effective_max_length = requested_max_length

        model.max_seq_length = effective_max_length
        model.eval()

        prefix = input_prompt_for_model(model_name)
        inputs = [f"{prefix}{text}" for text in texts]

        embeddings = model.encode(
            inputs,
            batch_size=settings["batch_size"],
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=settings["normalize"],
            prompt="",
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must form a two-dimensional matrix")

        if embeddings.shape[0] != len(texts):
            raise ValueError("Embedding count does not match input count")

        if not np.isfinite(embeddings).all():
            raise ValueError("Embeddings contain non-finite values")

        norms = np.linalg.norm(embeddings, axis=1)

        if np.any(norms <= 0):
            raise ValueError("Embeddings contain zero vectors")

        return embeddings, effective_max_length

    finally:
        del model
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def extract_single_model(
    root: Path,
    model_name: str,
    rows: list[dict],
    settings: dict,
) -> dict:
    alias = model_alias(model_name)

    output_directory = root / "artifacts" / "embeddings"
    output_directory.mkdir(parents=True, exist_ok=True)

    vectors_path = output_directory / f"{alias}.npy"
    metadata_path = output_directory / f"{alias}.json"

    if vectors_path.is_file() != metadata_path.is_file():
        vectors_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)

    if vectors_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        expected = {
            "model": model_name,
            "batch_size": settings["batch_size"],
            "max_length_requested": settings["max_length"],
            "normalized": settings["normalize"],
            "instances": len(rows),
            "input_prefix": input_prompt_for_model(model_name),
        }

        if any(metadata.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"Incompatible embedding artifacts for {alias}")

        vectors = np.load(vectors_path, mmap_mode="r")

        if (
            vectors.ndim != 2
            or vectors.shape[0] != len(rows)
            or vectors.shape[1] != metadata["dimensions"]
        ):
            raise RuntimeError(f"Invalid stored embedding shape for {alias}")

        print(f"Skipping completed embeddings: {alias}")

        return metadata

    print(f"Extracting embeddings: {model_name}")

    texts = [row["text"] for row in rows]

    vectors, effective_max_length = encode_model(
        model_name,
        texts,
        settings,
    )

    temporary_vectors = vectors_path.with_suffix(".npy.tmp")

    with temporary_vectors.open("wb") as stream:
        np.save(stream, vectors)

    temporary_vectors.replace(vectors_path)

    metadata = {
        "model": model_name,
        "alias": alias,
        "instances": len(rows),
        "dimensions": int(vectors.shape[1]),
        "dtype": "float32",
        "batch_size": settings["batch_size"],
        "max_length_requested": settings["max_length"],
        "max_length_effective": effective_max_length,
        "normalized": settings["normalize"],
        "input_prefix": input_prompt_for_model(model_name),
        "sentence_transformers_version": sentence_transformers.__version__,
        "torch_version": torch.__version__,
        "vectors_file": str(vectors_path.relative_to(root)),
        "indices_file": "artifacts/embeddings/ids.json",
    }

    write_json(metadata_path, metadata)

    print(f"Completed: {alias}, shape={vectors.shape}, max_length={effective_max_length}")

    return metadata


def run_embeddings(
    selected_model: str | None = None,
    project_root: Path | None = None,
) -> None:
    root = (project_root or Path.cwd()).resolve()
    config = load_experiment_config(root)

    settings = config["embeddings"]

    if settings["fine_tune"]:
        raise ValueError("This stage only supports frozen embeddings")

    configured_models = settings["models"]

    if len(configured_models) != len(set(configured_models)):
        raise ValueError("Duplicate embedding models")

    aliases = [model_alias(model) for model in configured_models]

    if len(aliases) != len(set(aliases)):
        raise ValueError("Duplicate embedding aliases")

    models = resolve_models(configured_models, selected_model)

    rows = load_canonical_corpus(root / config["data"].get("root", "data"))

    identifiers = [row["id"] for row in rows]

    output_directory = root / "artifacts" / "embeddings"
    output_directory.mkdir(parents=True, exist_ok=True)

    identifiers_path = output_directory / "ids.json"

    if identifiers_path.is_file():
        stored_identifiers = json.loads(identifiers_path.read_text(encoding="utf-8"))

        if stored_identifiers != identifiers:
            raise RuntimeError("Stored embedding identifiers differ from the corpus")
    else:
        write_json(identifiers_path, identifiers)

    report = {
        "strategy": "frozen_sentence_embeddings",
        "corpus_instances": len(rows),
        "models": [],
    }

    for model_name in configured_models:
        metadata_path = output_directory / f"{model_alias(model_name)}.json"

        if model_name in models:
            metadata = extract_single_model(
                root,
                model_name,
                rows,
                settings,
            )
        elif (
            metadata_path.is_file()
            and (output_directory / f"{model_alias(model_name)}.npy").is_file()
        ):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            continue

        report["models"].append(metadata)

    write_json(
        root / "reports" / "embeddings_summary.json",
        report,
    )

    print("Embedding extraction completed")


def input_prompt_for_model(model_name: str) -> str:
    if model_name == "intfloat/multilingual-e5-large":
        return "query: "

    if model_name == "google/embeddinggemma-300m":
        return "task: classification | query: "

    return ""
