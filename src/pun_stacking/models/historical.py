import json
from pathlib import Path

import joblib
import numpy as np
from nltk.corpus import stopwords
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    strip_accents_unicode,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from pun_stacking.data.audit import load_jsonl, pair_id
from pun_stacking.graphs.runner import load_experiment_config
from pun_stacking.models.base import (
    predict_probability,
    read_assignments,
    save_predictions,
    scores,
    write_json,
)


def load_portuguese_stopwords() -> list[str]:
    try:
        words = stopwords.words("portuguese")
    except LookupError as error:
        raise RuntimeError(
            "NLTK Portuguese stopwords are unavailable. "
            "Run: uv run python -m nltk.downloader stopwords"
        ) from error

    return sorted({strip_accents_unicode(str(word).lower()) for word in words})


def make_historical_model(
    training_rows: list[dict],
    settings: dict,
    portuguese_stopwords: list[str],
) -> VotingClassifier:
    labels = np.asarray(
        [row["label"] for row in training_rows],
        dtype=np.int64,
    )

    groups = np.asarray(
        [pair_id(row["id"]) for row in training_rows],
    )

    splitter = StratifiedGroupKFold(
        n_splits=settings["calibration_folds"],
        shuffle=True,
        random_state=settings["random_state"],
    )

    calibration_folds = list(
        splitter.split(
            np.zeros(len(training_rows)),
            labels,
            groups,
        )
    )

    for train_indices, validation_indices in calibration_folds:
        if set(groups[train_indices]) & set(groups[validation_indices]):
            raise ValueError("Pair crossing in SVM calibration")

        if set(labels[train_indices]) != {0, 1}:
            raise ValueError("Missing training classes in calibration")

        if set(labels[validation_indices]) != {0, 1}:
            raise ValueError("Missing validation classes in calibration")

    def make_text_pipeline(estimator):
        vectorizer = TfidfVectorizer(
            ngram_range=(
                settings["ngram_min"],
                settings["ngram_max"],
            ),
            stop_words=portuguese_stopwords,
            lowercase=True,
            strip_accents="unicode",
        )

        return Pipeline(
            [
                ("tfidf", vectorizer),
                ("classifier", estimator),
            ]
        )

    random_forest = make_text_pipeline(
        RandomForestClassifier(
            n_estimators=settings["rf_estimators"],
            criterion="entropy",
            max_depth=settings["rf_max_depth"],
            random_state=settings["rf_seed"],
            n_jobs=1,
        )
    )

    logistic_regression = make_text_pipeline(
        LogisticRegression(
            random_state=settings["random_state"],
            max_iter=settings["lr_max_iter"],
            class_weight="balanced",
        )
    )

    calibrated_svm = CalibratedClassifierCV(
        estimator=make_text_pipeline(
            LinearSVC(
                random_state=settings["random_state"],
                class_weight="balanced",
                max_iter=settings["svm_max_iter"],
            )
        ),
        method="sigmoid",
        cv=calibration_folds,
        n_jobs=1,
    )

    return VotingClassifier(
        estimators=[
            ("rf", random_forest),
            ("lr", logistic_regression),
            ("svm", calibrated_svm),
        ],
        voting="soft",
        n_jobs=1,
    )


def run_historical_split(
    root: Path,
    config: dict,
    split_seed: int,
) -> dict:
    settings = config["historical_baseline"]
    data_root = root / config["data"].get("root", "data")

    split_dir = data_root / "pair_controlled" / f"seed_{split_seed}"

    train_rows = load_jsonl(split_dir / "train.jsonl")
    validation_rows = load_jsonl(split_dir / "validation.jsonl")

    train_groups = {pair_id(row["id"]) for row in train_rows}

    validation_groups = {pair_id(row["id"]) for row in validation_rows}

    if train_groups & validation_groups:
        raise ValueError("Training and validation pairs overlap")

    n_folds = config["cross_validation"]["n_splits"]

    folds = read_assignments(
        data_root / "folds" / f"seed_{split_seed}.csv",
        train_rows,
        n_folds,
    )

    report_dir = root / "reports" / "historical_baseline" / f"split_seed_{split_seed}"

    artifact_dir = root / "artifacts" / "historical_baseline" / f"split_seed_{split_seed}"

    summary_path = report_dir / "summary.json"
    model_path = artifact_dir / "tfidf_soft_voting.joblib"

    oof_path = report_dir / "tfidf_soft_voting_oof.csv"
    validation_path = report_dir / "tfidf_soft_voting_validation.csv"

    protocol = {
        "method": "tfidf_soft_voting",
        "settings": settings,
        "external_split_seed": split_seed,
        "oof_folds": n_folds,
        "oof_seed": config["cross_validation"]["random_state"],
        "calibration": "stratified_group_kfold",
    }

    if summary_path.is_file():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))

        if previous.get("protocol") != protocol:
            raise RuntimeError(f"Incompatible historical baseline: {summary_path}")

        if all(
            path.is_file()
            for path in (
                model_path,
                oof_path,
                validation_path,
            )
        ):
            print(f"Skipping completed historical baseline: split={split_seed}")
            return previous

    portuguese_stopwords = load_portuguese_stopwords()

    labels = np.asarray(
        [row["label"] for row in train_rows],
        dtype=np.int64,
    )

    probabilities_oof = np.full(
        len(train_rows),
        np.nan,
        dtype=np.float64,
    )

    for fold in range(n_folds):
        train_indices = np.flatnonzero(folds != fold)
        held_indices = np.flatnonzero(folds == fold)

        fold_train = [train_rows[i] for i in train_indices]
        fold_held = [train_rows[i] for i in held_indices]

        model = make_historical_model(
            fold_train,
            settings,
            portuguese_stopwords,
        )

        model.fit(
            [row["text"] for row in fold_train],
            labels[train_indices],
        )

        probabilities_oof[held_indices] = predict_probability(
            model,
            [row["text"] for row in fold_held],
        )

        print(
            f"Split {split_seed}: historical OOF fold {fold + 1}/{n_folds} completed",
            flush=True,
        )

    if not np.isfinite(probabilities_oof).all():
        raise RuntimeError("Incomplete historical OOF probabilities")

    final_model = make_historical_model(
        train_rows,
        settings,
        portuguese_stopwords,
    )

    final_model.fit(
        [row["text"] for row in train_rows],
        labels,
    )

    validation_probabilities = predict_probability(
        final_model,
        [row["text"] for row in validation_rows],
    )

    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    save_predictions(
        oof_path,
        train_rows,
        probabilities_oof,
        folds,
    )

    save_predictions(
        validation_path,
        validation_rows,
        validation_probabilities,
    )

    temporary_model = model_path.with_suffix(".joblib.tmp")

    joblib.dump(
        {
            "method": "historical_tfidf_soft_voting",
            "classifier": final_model,
            "protocol": protocol,
        },
        temporary_model,
    )

    temporary_model.replace(model_path)

    report = {
        "split_seed": split_seed,
        "protocol": protocol,
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "oof": scores(train_rows, probabilities_oof),
        "validation": scores(
            validation_rows,
            validation_probabilities,
        ),
    }

    write_json(summary_path, report)

    print(
        f"Split {split_seed}: historical baseline completed, "
        f"validation_f1_macro="
        f"{report['validation']['f1_macro']:.4f}",
        flush=True,
    )

    return report


def run_historical_baseline(
    split_seed: int | None = None,
    project_root: Path | None = None,
) -> None:
    root = (project_root or Path.cwd()).resolve()
    config = load_experiment_config(root)

    if not config["historical_baseline"]["enabled"]:
        raise ValueError("Historical baseline is disabled")

    seeds = config["data"]["split_seeds"]

    if split_seed is not None:
        if split_seed not in seeds:
            raise ValueError(f"Unknown split seed: {split_seed}")

        seeds = [split_seed]

    for seed in seeds:
        run_historical_split(root, config, seed)
