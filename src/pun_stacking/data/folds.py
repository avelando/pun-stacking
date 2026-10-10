import csv
import json
import tomllib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from pun_stacking.data.audit import inspect_splits, load_jsonl, pair_id


def create_group_folds(
    rows: list[dict],
    n_splits: int,
    random_state: int,
) -> tuple[dict[str, int], list[dict]]:
    if n_splits < 2:
        raise ValueError("At least two folds are required")

    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate example IDs")

    groups = np.array([pair_id(row["id"]) for row in rows])
    labels = np.array([row["label"] for row in rows])

    pair_labels = defaultdict(list)

    for group, label in zip(groups, labels, strict=True):
        pair_labels[group].append(int(label))

    if any(sorted(values) != [0, 1] for values in pair_labels.values()):
        raise ValueError("Every pair must contain one example of each class")

    if len(pair_labels) < n_splits:
        raise ValueError("Not enough pairs for the requested folds")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    assignments = {}
    summaries = []

    for fold, (train_indices, validation_indices) in enumerate(
        splitter.split(
            np.zeros(len(rows)),
            labels,
            groups,
        )
    ):
        train_groups = set(groups[train_indices])
        validation_groups = set(groups[validation_indices])

        if train_groups & validation_groups:
            raise ValueError(f"Pair crossing detected in fold {fold}")

        if len(validation_indices) != 2 * len(validation_groups):
            raise ValueError(f"Incomplete validation pairs in fold {fold}")

        validation_labels = Counter(labels[validation_indices].tolist())

        if validation_labels[0] != validation_labels[1]:
            raise ValueError(f"Class imbalance detected in fold {fold}")

        for group in validation_groups:
            if group in assignments:
                raise ValueError(f"Pair assigned to multiple folds: {group}")

            assignments[group] = fold

        summaries.append(
            {
                "fold": fold,
                "train_examples": len(train_indices),
                "validation_examples": len(validation_indices),
                "train_pairs": len(train_groups),
                "validation_pairs": len(validation_groups),
                "validation_class_counts": {
                    "0": validation_labels[0],
                    "1": validation_labels[1],
                },
            }
        )

    if set(assignments) != set(pair_labels):
        raise ValueError("Some pairs were not assigned to a fold")

    return assignments, summaries


def build_folds(project_root: Path | None = None) -> Path:
    root = (project_root or Path.cwd()).resolve()

    with (root / "configs" / "experiments.toml").open("rb") as stream:
        config = tomllib.load(stream)

    data_config = config["data"]
    cv_config = config["cross_validation"]

    if not cv_config["group_by_pair"]:
        raise ValueError("Pair-grouped cross-validation must be enabled")

    seeds = data_config["split_seeds"]
    n_splits = cv_config["n_splits"]
    random_state = cv_config["random_state"]

    data_root = root / data_config.get("root", "data")
    output_directory = data_root / "folds"
    output_directory.mkdir(parents=True, exist_ok=True)

    reports = []

    for seed in seeds:
        split_directory = data_root / "pair_controlled" / f"seed_{seed}"

        inspect_splits(split_directory, require_pairs=True)

        rows = load_jsonl(split_directory / "train.jsonl")

        assignments, summaries = create_group_folds(
            rows=rows,
            n_splits=n_splits,
            random_state=random_state,
        )

        output_path = output_directory / f"seed_{seed}.csv"

        with output_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["pair_id", "fold"])

            for group in sorted(assignments):
                writer.writerow([group, assignments[group]])

        reports.append(
            {
                "split_seed": seed,
                "total_training_examples": len(rows),
                "total_training_pairs": len(assignments),
                "folds": summaries,
                "assignment_file": str(output_path.relative_to(root)),
            }
        )

        print(f"Seed {seed}: {len(assignments)} pairs assigned to {n_splits} folds")

    report = {
        "strategy": "stratified_group_kfold",
        "n_splits": n_splits,
        "random_state": random_state,
        "partitions": reports,
    }

    output_path = root / "reports" / "oof_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Report: {output_path}")

    return output_path
