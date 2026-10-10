import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pun_stacking.data.audit import load_jsonl, pair_id
from pun_stacking.graphs.features import GraphFeatureEncoder
from pun_stacking.graphs.runner import (
    build_fold_graphs,
    load_experiment_config,
    load_final_graphs,
)


def embedding_alias(model: str) -> str:
    return model.rsplit("/", 1)[-1].lower()


def read_assignments(
    path: Path,
    rows: list[dict],
    n_folds: int,
) -> np.ndarray:
    assignments = {}

    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)

        if reader.fieldnames != ["pair_id", "fold"]:
            raise ValueError(f"Invalid fold columns: {path}")

        for record in reader:
            group = record["pair_id"]

            if group in assignments:
                raise ValueError(f"Duplicate fold assignment: {group}")

            fold = int(record["fold"])

            if not 0 <= fold < n_folds:
                raise ValueError(f"Invalid fold: {fold}")

            assignments[group] = fold

    groups = {pair_id(row["id"]) for row in rows}

    if set(assignments) != groups:
        raise ValueError(f"Fold assignments do not match training pairs: {path}")

    folds = np.array(
        [assignments[pair_id(row["id"])] for row in rows],
        dtype=np.int64,
    )

    for fold in range(n_folds):
        ids = {pair_id(row["id"]) for i, row in enumerate(rows) if folds[i] == fold}

        if not ids:
            raise ValueError(f"Empty fold: {fold}")

        if int(np.sum(folds == fold)) != 2 * len(ids):
            raise ValueError(f"Incomplete pairs in fold {fold}")

    return folds


def get_embedding_matrices(
    root: Path,
    models: list[str],
    train_rows: list[dict],
    validation_rows: list[dict],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    directory = root / "artifacts" / "embeddings"

    identifiers = json.loads((directory / "ids.json").read_text(encoding="utf-8"))

    if len(identifiers) != 5700 or len(set(identifiers)) != len(identifiers):
        raise ValueError("Invalid canonical embedding IDs")

    index = {identifier: position for position, identifier in enumerate(identifiers)}

    train_positions = [index[row["id"]] for row in train_rows]
    validation_positions = [index[row["id"]] for row in validation_rows]

    results = {}

    for model_name in models:
        alias = embedding_alias(model_name)

        info = json.loads((directory / f"{alias}.json").read_text(encoding="utf-8"))

        if info["model"] != model_name or info["instances"] != len(identifiers):
            raise ValueError(f"Incompatible embedding metadata: {alias}")

        matrix = np.load(directory / f"{alias}.npy", mmap_mode="r")

        if (
            matrix.ndim != 2
            or matrix.shape != (len(identifiers), info["dimensions"])
            or matrix.dtype != np.float32
        ):
            raise ValueError(f"Invalid embedding matrix: {alias}")

        train = np.asarray(
            matrix[train_positions],
            dtype=np.float32,
        )

        validation = np.asarray(
            matrix[validation_positions],
            dtype=np.float32,
        )

        if not np.isfinite(train).all() or not np.isfinite(validation).all():
            raise ValueError(f"Non-finite embeddings: {alias}")

        results[alias] = (train, validation)

    return results


def make_classifier(name: str, settings: dict) -> Pipeline:
    classifier = LogisticRegression(
        C=settings["C"],
        max_iter=settings["max_iter"],
        solver="lbfgs",
    )

    if name == "tfidf":
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=settings["tfidf_min_df"],
            sublinear_tf=True,
        )

        return Pipeline(
            [
                ("tfidf", vectorizer),
                ("classifier", classifier),
            ]
        )

    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def predict_probability(model: Pipeline, features) -> np.ndarray:
    if list(model.classes_) != [0, 1]:
        raise ValueError("Expected binary classes 0 and 1")

    probabilities = np.asarray(
        model.predict_proba(features),
        dtype=np.float64,
    )

    if (
        probabilities.ndim != 2
        or probabilities.shape[1] != 2
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    ):
        raise ValueError("Invalid predicted probabilities")

    return probabilities[:, 1]


def scores(rows: list[dict], p_pun: np.ndarray) -> dict:
    labels = np.array([row["label"] for row in rows])
    predictions = (p_pun >= 0.5).astype(int)

    pairs = {}

    for row, prediction in zip(rows, predictions, strict=True):
        group = pair_id(row["id"])

        pairs.setdefault(group, []).append(int(prediction == row["label"]))

    if any(len(correct) != 2 for correct in pairs.values()):
        raise ValueError("Paired evaluation requires two examples per pair")

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "pair_exact_accuracy": float(np.mean([all(correct) for correct in pairs.values()])),
    }


def save_predictions(
    path: Path,
    rows: list[dict],
    p_pun: np.ndarray,
    folds=None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(rows) != len(p_pun):
        raise ValueError("Prediction count mismatch")

    temporary = path.with_suffix(".csv.tmp")

    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)

        writer.writerow(["id", "label", "fold", "p_non_pun", "p_pun"])

        for position, (row, probability) in enumerate(zip(rows, p_pun, strict=True)):
            writer.writerow(
                [
                    row["id"],
                    row["label"],
                    "" if folds is None else int(folds[position]),
                    float(1 - probability),
                    float(probability),
                ]
            )

    temporary.replace(path)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(".json.tmp")

    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)


def run_single_split(
    root: Path,
    config: dict,
    split_seed: int,
) -> None:
    data_root = root / config["data"].get("root", "data")

    directory = data_root / "pair_controlled" / f"seed_{split_seed}"

    train_rows = load_jsonl(directory / "train.jsonl")
    validation_rows = load_jsonl(directory / "validation.jsonl")

    if {pair_id(r["id"]) for r in train_rows} & {pair_id(r["id"]) for r in validation_rows}:
        raise ValueError("Training and validation pairs overlap")

    n_folds = config["cross_validation"]["n_splits"]

    folds = read_assignments(
        data_root / "folds" / f"seed_{split_seed}.csv",
        train_rows,
        n_folds,
    )

    models = config["embeddings"]["models"]

    names = [
        "tfidf",
        *[embedding_alias(model) for model in models],
        *config["graphs"]["names"],
    ]

    if len(names) != len(set(names)):
        raise ValueError("Duplicate representation names")

    settings = config["base_models"]

    report_dir = root / "reports" / "base_models" / f"split_seed_{split_seed}"

    model_dir = root / "artifacts" / "base_models" / f"split_seed_{split_seed}"

    summary_path = report_dir / "summary.json"

    protocol = {
        "classifier": "logistic_regression",
        "configuration": settings,
        "embeddings": config["embeddings"],
        "graphs": config["graphs"],
        "folds": n_folds,
        "fold_seed": config["cross_validation"]["random_state"],
    }

    if summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))

        if previous.get("protocol") != protocol or set(previous.get("models", {})) != set(names):
            raise RuntimeError(f"Existing result uses another protocol: {summary_path}")

        required = [
            path
            for name in names
            for path in (
                report_dir / f"{name}_oof.csv",
                report_dir / f"{name}_validation.csv",
                model_dir / f"{name}.joblib",
            )
        ]

        if not all(path.is_file() for path in required):
            raise RuntimeError(f"Incomplete completed results: {summary_path}")

        print(f"Skipping completed base models for split {split_seed}")
        return

    matrices = get_embedding_matrices(
        root,
        models,
        train_rows,
        validation_rows,
    )

    oof = {name: np.full(len(train_rows), np.nan, dtype=np.float64) for name in names}

    for fold in range(n_folds):
        train_idx = np.flatnonzero(folds != fold)
        hold_idx = np.flatnonzero(folds == fold)

        train_fold_rows = [train_rows[i] for i in train_idx]
        hold_fold_rows = [train_rows[i] for i in hold_idx]

        y_fold = [row["label"] for row in train_fold_rows]

        fold_graphs, source_rows, held_rows = build_fold_graphs(
            split_seed,
            fold,
            root,
        )

        if [row["id"] for row in source_rows] != [row["id"] for row in train_fold_rows]:
            raise ValueError("OOF graph training mismatch")

        if [row["id"] for row in held_rows] != [row["id"] for row in hold_fold_rows]:
            raise ValueError("OOF graph holdout mismatch")

        views = {
            "tfidf": (
                [row["text"] for row in train_fold_rows],
                [row["text"] for row in hold_fold_rows],
            ),
        }

        for alias, (train_matrix, _) in matrices.items():
            views[alias] = (
                train_matrix[train_idx],
                train_matrix[hold_idx],
            )

        for name, graph in fold_graphs.items():
            encoder = GraphFeatureEncoder.fit(
                graph,
                dimensions=config["graphs"]["svd_dimensions"],
                seed=config["graphs"]["svd_seed"],
            )

            views[name] = (
                encoder.transform([row["tokens"] for row in train_fold_rows]),
                encoder.transform([row["tokens"] for row in hold_fold_rows]),
            )

        for name in names:
            x_train, x_hold = views[name]

            classifier = make_classifier(name, settings)

            classifier.fit(x_train, y_fold)

            oof[name][hold_idx] = predict_probability(
                classifier,
                x_hold,
            )

        print(
            f"Split {split_seed}: completed OOF fold {fold + 1}/{n_folds}",
            flush=True,
        )

    if any(not np.isfinite(probabilities).all() for probabilities in oof.values()):
        raise RuntimeError("Incomplete OOF predictions")

    graphs = load_final_graphs(split_seed, root)

    train_labels = [row["label"] for row in train_rows]

    model_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "split_seed": split_seed,
        "protocol": protocol,
        "models": {},
    }

    for name in names:
        encoder = None

        if name == "tfidf":
            x_train = [row["text"] for row in train_rows]
            x_validation = [row["text"] for row in validation_rows]

        elif name in matrices:
            x_train, x_validation = matrices[name]

        else:
            encoder = GraphFeatureEncoder.fit(
                graphs[name],
                dimensions=config["graphs"]["svd_dimensions"],
                seed=config["graphs"]["svd_seed"],
            )

            x_train = encoder.transform([row["tokens"] for row in train_rows])

            x_validation = encoder.transform([row["tokens"] for row in validation_rows])

        classifier = make_classifier(name, settings)

        classifier.fit(x_train, train_labels)

        valid_prob = predict_probability(
            classifier,
            x_validation,
        )

        save_predictions(
            report_dir / f"{name}_oof.csv",
            train_rows,
            oof[name],
            folds,
        )

        save_predictions(
            report_dir / f"{name}_validation.csv",
            validation_rows,
            valid_prob,
        )

        joblib.dump(
            {
                "representation": name,
                "classifier": classifier,
                "graph_encoder": encoder,
            },
            model_dir / f"{name}.joblib",
        )

        report["models"][name] = {
            "oof": scores(train_rows, oof[name]),
            "validation": scores(validation_rows, valid_prob),
        }

        print(
            f"Split {split_seed}: completed {name}",
            flush=True,
        )

    write_json(summary_path, report)


def run_base_models(
    split_seed: int | None = None,
    project_root: Path | None = None,
) -> None:
    root = (project_root or Path.cwd()).resolve()
    config = load_experiment_config(root)

    seeds = config["data"]["split_seeds"]

    if split_seed is not None:
        if split_seed not in seeds:
            raise ValueError(f"Unknown split seed: {split_seed}")

        seeds = [split_seed]

    for seed in seeds:
        run_single_split(root, config, seed)
