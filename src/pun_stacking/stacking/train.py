import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pun_stacking.data.audit import load_jsonl, pair_id
from pun_stacking.graphs.runner import load_experiment_config
from pun_stacking.models.base import (
    embedding_alias,
    predict_probability,
    read_assignments,
    save_predictions,
    scores,
    write_json,
)


def get_configurations(config: dict) -> dict[str, list[str]]:
    embeddings = [embedding_alias(model) for model in config["embeddings"]["models"]]

    graphs = config["graphs"]["names"]

    if set(graphs) != {"cooccurrence", "ppmi", "pun_context"}:
        raise ValueError("Unexpected graph configuration")

    configurations = {
        "graphs_only": list(graphs),
        "tfidf_graphs": ["tfidf", *graphs],
        "embeddings_only": list(embeddings),
        "all_views": ["tfidf", *embeddings, *graphs],
    }

    for embedding in embeddings:
        configurations[f"{embedding}_graphs"] = [
            embedding,
            *graphs,
        ]

        configurations[f"{embedding}_without_pun_context"] = [
            embedding,
            "cooccurrence",
            "ppmi",
        ]

    if len(configurations) != 12:
        raise ValueError("Unexpected number of stacking configurations")

    for name, views in configurations.items():
        if len(views) != len(set(views)):
            raise ValueError(f"Duplicate views in {name}")

    return configurations


def read_probabilities(
    path: Path,
    rows: list[dict],
    expected_folds: np.ndarray | None = None,
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing predictions: {path}")

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)

        if reader.fieldnames != [
            "id",
            "label",
            "fold",
            "p_non_pun",
            "p_pun",
        ]:
            raise ValueError(f"Unexpected prediction columns: {path}")

        records = list(reader)

    if len(records) != len(rows):
        raise ValueError(f"Prediction count mismatch: {path}")

    probabilities = np.empty(len(rows), dtype=np.float64)

    for index, (record, row) in enumerate(zip(records, rows, strict=True)):
        if record["id"] != row["id"]:
            raise ValueError(f"Prediction ID mismatch: {path}")

        if int(record["label"]) != row["label"]:
            raise ValueError(f"Prediction label mismatch: {path}")

        if expected_folds is None:
            if record["fold"] != "":
                raise ValueError(f"Unexpected fold in validation: {path}")
        elif int(record["fold"]) != int(expected_folds[index]):
            raise ValueError(f"OOF fold mismatch: {path}")

        p_non_pun = float(record["p_non_pun"])
        p_pun = float(record["p_pun"])

        if not np.isfinite([p_non_pun, p_pun]).all():
            raise ValueError(f"Non-finite probabilities: {path}")

        if not 0 <= p_non_pun <= 1 or not 0 <= p_pun <= 1:
            raise ValueError(f"Invalid probabilities: {path}")

        if not np.isclose(
            p_non_pun + p_pun,
            1.0,
            atol=1e-6,
        ):
            raise ValueError(f"Probabilities do not sum to one: {path}")

        probabilities[index] = p_pun

    return probabilities


def load_base_predictions(
    root: Path,
    config: dict,
    split_seed: int,
    train_rows: list[dict],
    validation_rows: list[dict],
    folds: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    directory = root / "reports" / "base_models" / f"split_seed_{split_seed}"

    summary_path = directory / "summary.json"

    if not summary_path.is_file():
        raise FileNotFoundError(f"Base classifiers have not finished: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    expected_protocol = {
        "classifier": "logistic_regression",
        "configuration": config["base_models"],
        "embeddings": config["embeddings"],
        "graphs": config["graphs"],
        "folds": config["cross_validation"]["n_splits"],
        "fold_seed": config["cross_validation"]["random_state"],
    }

    if summary.get("protocol") != expected_protocol:
        raise ValueError(f"Base model protocol differs from current configuration: {summary_path}")

    names = [
        "tfidf",
        *[embedding_alias(model) for model in config["embeddings"]["models"]],
        *config["graphs"]["names"],
    ]

    if set(summary.get("models", {})) != set(names):
        raise ValueError("Missing base model results")

    train_probabilities = {}
    validation_probabilities = {}

    for name in names:
        train_probabilities[name] = read_probabilities(
            directory / f"{name}_oof.csv",
            train_rows,
            expected_folds=folds,
        )

        validation_probabilities[name] = read_probabilities(
            directory / f"{name}_validation.csv",
            validation_rows,
        )

    return train_probabilities, validation_probabilities


def make_meta_classifier(settings: dict) -> Pipeline:
    if settings["meta_classifier"] != "logistic_regression":
        raise ValueError("Unsupported meta-classifier")

    if not settings["use_oof"]:
        raise ValueError("Stacking requires OOF predictions")

    steps = []

    if settings["standardize"]:
        steps.append(("scale", StandardScaler()))

    steps.append(
        (
            "classifier",
            LogisticRegression(
                C=settings["C"],
                max_iter=settings["max_iter"],
                solver="lbfgs",
            ),
        )
    )

    return Pipeline(steps)


def run_single_split(
    root: Path,
    config: dict,
    split_seed: int,
) -> None:
    data_root = root / config["data"].get("root", "data")

    split_directory = data_root / "pair_controlled" / f"seed_{split_seed}"

    train_rows = load_jsonl(split_directory / "train.jsonl")
    validation_rows = load_jsonl(split_directory / "validation.jsonl")

    train_pairs = {pair_id(row["id"]) for row in train_rows}
    validation_pairs = {pair_id(row["id"]) for row in validation_rows}

    if train_pairs & validation_pairs:
        raise ValueError("Training and validation pairs overlap")

    folds = read_assignments(
        data_root / "folds" / f"seed_{split_seed}.csv",
        train_rows,
        config["cross_validation"]["n_splits"],
    )

    configurations = get_configurations(config)
    settings = config["stacking"]

    report_directory = root / "reports" / "stacking" / f"split_seed_{split_seed}"

    model_directory = root / "artifacts" / "stacking" / f"split_seed_{split_seed}"

    summary_path = report_directory / "summary.json"

    protocol = {
        "method": "out_of_fold_stacking",
        "stacking": settings,
        "base_models": config["base_models"],
        "embeddings": config["embeddings"],
        "graphs": config["graphs"],
        "cross_validation": config["cross_validation"],
        "configurations": configurations,
    }

    if summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))

        if previous.get("protocol") != protocol:
            raise RuntimeError(f"Incompatible stacking results: {summary_path}")

        required_files = [
            path
            for name in configurations
            for path in (
                report_directory / f"{name}_validation.csv",
                model_directory / f"{name}.joblib",
            )
        ]

        if all(path.is_file() for path in required_files):
            print(f"Skipping completed stacking for split {split_seed}")
            return

    train_probabilities, validation_probabilities = load_base_predictions(
        root,
        config,
        split_seed,
        train_rows,
        validation_rows,
        folds,
    )

    labels = np.array(
        [row["label"] for row in train_rows],
        dtype=np.int64,
    )

    report_directory.mkdir(parents=True, exist_ok=True)
    model_directory.mkdir(parents=True, exist_ok=True)

    results = {}

    for name, views in configurations.items():
        x_train = np.column_stack([train_probabilities[view] for view in views])

        x_validation = np.column_stack([validation_probabilities[view] for view in views])

        if not np.isfinite(x_train).all() or not np.isfinite(x_validation).all():
            raise ValueError(f"Invalid stacking features: {name}")

        model = make_meta_classifier(settings)
        model.fit(x_train, labels)

        probabilities = predict_probability(
            model,
            x_validation,
        )

        save_predictions(
            report_directory / f"{name}_validation.csv",
            validation_rows,
            probabilities,
        )

        artifact_path = model_directory / f"{name}.joblib"
        temporary_artifact = artifact_path.with_suffix(".joblib.tmp")

        joblib.dump(
            {
                "method": "stacking",
                "views": views,
                "classifier": model,
            },
            temporary_artifact,
        )

        temporary_artifact.replace(artifact_path)

        results[name] = {
            "views": views,
            "validation": scores(
                validation_rows,
                probabilities,
            ),
        }

        print(
            f"Split {split_seed}: "
            f"{name} completed, "
            f"F1-macro={results[name]['validation']['f1_macro']:.4f}",
            flush=True,
        )

    write_json(
        summary_path,
        {
            "split_seed": split_seed,
            "protocol": protocol,
            "models": results,
        },
    )


def run_stacking(
    split_seed: int | None = None,
    project_root: Path | None = None,
) -> None:
    root = (project_root or Path.cwd()).resolve()
    config = load_experiment_config(root)

    if not config["stacking"]["enabled"]:
        raise ValueError("Stacking is disabled")

    seeds = config["data"]["split_seeds"]

    if split_seed is not None:
        if split_seed not in seeds:
            raise ValueError(f"Unknown split seed: {split_seed}")

        seeds = [split_seed]

    for seed in seeds:
        run_single_split(root, config, seed)
