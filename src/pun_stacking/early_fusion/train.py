import json
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from pun_stacking.data.audit import load_jsonl, pair_id
from pun_stacking.graphs.features import GraphFeatureEncoder
from pun_stacking.graphs.runner import (
    build_fold_graphs,
    load_experiment_config,
    load_final_graphs,
)
from pun_stacking.models.base import (
    get_embedding_matrices,
    predict_probability,
    read_assignments,
    save_predictions,
    scores,
    write_json,
)
from pun_stacking.stacking.train import get_configurations


def prepare_numeric_views(
    train_rows: list[dict],
    evaluation_rows: list[dict],
    embeddings: dict[str, tuple[np.ndarray, np.ndarray]],
    graphs: dict,
    graph_settings: dict,
) -> tuple[dict, dict]:
    views = dict(embeddings)
    encoders = {}

    train_tokens = [row["tokens"] for row in train_rows]
    evaluation_tokens = [row["tokens"] for row in evaluation_rows]

    for name in graph_settings["names"]:
        encoder = GraphFeatureEncoder.fit(
            graphs[name],
            dimensions=graph_settings["svd_dimensions"],
            seed=graph_settings["svd_seed"],
        )

        views[name] = (
            encoder.transform(train_tokens),
            encoder.transform(evaluation_tokens),
        )

        encoders[name] = encoder

    return views, encoders


def fit_fusion(
    views: list[str],
    numeric_views: dict,
    train_texts: list[str],
    evaluation_texts: list[str],
    train_labels: np.ndarray,
    fusion_settings: dict,
    base_settings: dict,
) -> tuple[LogisticRegression, object, dict]:
    if fusion_settings["classifier"] != "logistic_regression":
        raise ValueError("Unsupported early fusion classifier")

    numeric_names = [name for name in views if name != "tfidf"]

    scaler = None
    vectorizer = None
    train_blocks = []
    evaluation_blocks = []

    if "tfidf" in views:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=base_settings["tfidf_min_df"],
            sublinear_tf=True,
        )

        train_blocks.append(vectorizer.fit_transform(train_texts))

        evaluation_blocks.append(vectorizer.transform(evaluation_texts))

    if numeric_names:
        missing = [name for name in numeric_names if name not in numeric_views]

        if missing:
            raise ValueError(f"Missing early fusion views: {missing}")

        train_numeric = np.concatenate(
            [numeric_views[name][0] for name in numeric_names],
            axis=1,
        )

        evaluation_numeric = np.concatenate(
            [numeric_views[name][1] for name in numeric_names],
            axis=1,
        )

        if not np.isfinite(train_numeric).all() or not np.isfinite(evaluation_numeric).all():
            raise ValueError("Non-finite early fusion features")

        if fusion_settings["standardize_numeric"]:
            scaler = StandardScaler()

            train_numeric = scaler.fit_transform(train_numeric)

            evaluation_numeric = scaler.transform(evaluation_numeric)

        if "tfidf" in views:
            train_blocks.append(csr_matrix(train_numeric))
            evaluation_blocks.append(csr_matrix(evaluation_numeric))
        else:
            train_blocks.append(train_numeric)
            evaluation_blocks.append(evaluation_numeric)

    if not train_blocks:
        raise ValueError("Early fusion requires at least one view")

    if len(train_blocks) == 1:
        x_train = train_blocks[0]
        x_evaluation = evaluation_blocks[0]
    else:
        x_train = hstack(train_blocks, format="csr")
        x_evaluation = hstack(evaluation_blocks, format="csr")

    if x_train.shape[1] != x_evaluation.shape[1]:
        raise ValueError("Incompatible feature dimensions")

    if not np.isfinite(x_train.data if hasattr(x_train, "data") else x_train).all():
        raise ValueError("Invalid training features")

    if not np.isfinite(x_evaluation.data if hasattr(x_evaluation, "data") else x_evaluation).all():
        raise ValueError("Invalid evaluation features")

    classifier = LogisticRegression(
        C=fusion_settings["C"],
        max_iter=fusion_settings["max_iter"],
        solver="lbfgs",
    )

    classifier.fit(x_train, train_labels)

    probabilities = predict_probability(
        classifier,
        x_evaluation,
    )

    preprocessing = {
        "tfidf_vectorizer": vectorizer,
        "numeric_scaler": scaler,
        "numeric_views": numeric_names,
        "feature_count": int(x_train.shape[1]),
    }

    return classifier, probabilities, preprocessing


def run_single_split(
    root: Path,
    config: dict,
    split_seed: int,
) -> None:
    settings = config["early_fusion"]
    base_settings = config["base_models"]

    data_root = root / config["data"].get("root", "data")
    split_directory = data_root / "pair_controlled" / f"seed_{split_seed}"

    train_rows = load_jsonl(split_directory / "train.jsonl")

    validation_rows = load_jsonl(split_directory / "validation.jsonl")

    train_pairs = {pair_id(row["id"]) for row in train_rows}
    validation_pairs = {pair_id(row["id"]) for row in validation_rows}

    if train_pairs & validation_pairs:
        raise ValueError("Training and validation pairs overlap")

    n_folds = config["cross_validation"]["n_splits"]

    folds = read_assignments(
        data_root / "folds" / f"seed_{split_seed}.csv",
        train_rows,
        n_folds,
    )

    configurations = get_configurations(config)

    report_directory = root / "reports" / "early_fusion" / f"split_seed_{split_seed}"

    model_directory = root / "artifacts" / "early_fusion" / f"split_seed_{split_seed}"

    summary_path = report_directory / "summary.json"

    protocol = {
        "method": "feature_level_early_fusion",
        "early_fusion": settings,
        "base_models": base_settings,
        "embeddings": config["embeddings"],
        "graphs": config["graphs"],
        "cross_validation": config["cross_validation"],
        "configurations": configurations,
    }

    if summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))

        if previous.get("protocol") != protocol:
            raise RuntimeError(f"Incompatible early fusion results: {summary_path}")

        required_files = [
            path
            for name in configurations
            for path in (
                report_directory / f"{name}_oof.csv",
                report_directory / f"{name}_validation.csv",
                model_directory / f"{name}.joblib",
            )
        ]

        if not all(path.is_file() for path in required_files):
            raise RuntimeError(f"Incomplete early fusion results: {summary_path}")

        print(f"Skipping completed early fusion for split {split_seed}")
        return

    embedding_matrices = get_embedding_matrices(
        root,
        config["embeddings"]["models"],
        train_rows,
        validation_rows,
    )

    labels = np.asarray(
        [row["label"] for row in train_rows],
        dtype=np.int64,
    )

    oof = {name: np.full(len(train_rows), np.nan, dtype=np.float64) for name in configurations}

    for fold in range(n_folds):
        train_indices = np.flatnonzero(folds != fold)
        held_indices = np.flatnonzero(folds == fold)

        fold_train = [train_rows[i] for i in train_indices]
        fold_held = [train_rows[i] for i in held_indices]

        fold_graphs, graph_train, graph_held = build_fold_graphs(
            split_seed,
            fold,
            root,
        )

        if [row["id"] for row in fold_train] != [row["id"] for row in graph_train]:
            raise ValueError("OOF graph training mismatch")

        if [row["id"] for row in fold_held] != [row["id"] for row in graph_held]:
            raise ValueError("OOF graph evaluation mismatch")

        fold_embeddings = {
            name: (
                matrices[0][train_indices],
                matrices[0][held_indices],
            )
            for name, matrices in embedding_matrices.items()
        }

        numeric_views, _ = prepare_numeric_views(
            fold_train,
            fold_held,
            fold_embeddings,
            fold_graphs,
            config["graphs"],
        )

        fold_labels = labels[train_indices]

        train_texts = [row["text"] for row in fold_train]
        held_texts = [row["text"] for row in fold_held]

        for name, views in configurations.items():
            _, probabilities, _ = fit_fusion(
                views,
                numeric_views,
                train_texts,
                held_texts,
                fold_labels,
                settings,
                base_settings,
            )

            oof[name][held_indices] = probabilities

        print(
            f"Split {split_seed}: early fusion OOF fold {fold + 1}/{n_folds} completed",
            flush=True,
        )

    if any(not np.isfinite(probabilities).all() for probabilities in oof.values()):
        raise RuntimeError("Incomplete early fusion OOF predictions")

    final_graphs = load_final_graphs(
        split_seed,
        root,
    )

    final_views, final_encoders = prepare_numeric_views(
        train_rows,
        validation_rows,
        embedding_matrices,
        final_graphs,
        config["graphs"],
    )

    report_directory.mkdir(parents=True, exist_ok=True)
    model_directory.mkdir(parents=True, exist_ok=True)

    train_texts = [row["text"] for row in train_rows]
    validation_texts = [row["text"] for row in validation_rows]

    results = {}

    for name, views in configurations.items():
        classifier, validation_probabilities, preprocessing = fit_fusion(
            views,
            final_views,
            train_texts,
            validation_texts,
            labels,
            settings,
            base_settings,
        )

        save_predictions(
            report_directory / f"{name}_oof.csv",
            train_rows,
            oof[name],
            folds,
        )

        save_predictions(
            report_directory / f"{name}_validation.csv",
            validation_rows,
            validation_probabilities,
        )

        selected_encoders = {view: final_encoders[view] for view in views if view in final_encoders}

        artifact_path = model_directory / f"{name}.joblib"
        temporary_artifact = artifact_path.with_suffix(".joblib.tmp")

        joblib.dump(
            {
                "method": "early_fusion",
                "views": views,
                "classifier": classifier,
                "preprocessing": preprocessing,
                "graph_encoders": selected_encoders,
            },
            temporary_artifact,
        )

        temporary_artifact.replace(artifact_path)

        results[name] = {
            "views": views,
            "feature_count": preprocessing["feature_count"],
            "oof": scores(train_rows, oof[name]),
            "validation": scores(
                validation_rows,
                validation_probabilities,
            ),
        }

        print(
            f"Split {split_seed}: early fusion {name} completed, "
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


def run_early_fusion(
    split_seed: int | None = None,
    project_root: Path | None = None,
) -> None:
    root = (project_root or Path.cwd()).resolve()
    config = load_experiment_config(root)

    if not config["early_fusion"]["enabled"]:
        raise ValueError("Early fusion is disabled")

    seeds = config["data"]["split_seeds"]

    if split_seed is not None:
        if split_seed not in seeds:
            raise ValueError(f"Unknown split seed: {split_seed}")

        seeds = [split_seed]

    for seed in seeds:
        run_single_split(root, config, seed)
