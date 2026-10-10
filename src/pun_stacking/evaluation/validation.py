import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from pun_stacking.graphs.runner import load_experiment_config
from pun_stacking.models.base import embedding_alias, write_json
from pun_stacking.stacking.train import get_configurations

METRICS = ("accuracy", "f1_macro", "pair_exact_accuracy")


def read_report(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing validation report: {path}")

    report = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(report, dict):
        raise ValueError(f"Invalid report: {path}")

    return report


def make_record(
    family: str,
    model: str,
    split_seed: int,
    metrics: dict,
    model_seed: int | None = None,
) -> dict:
    record = {
        "family": family,
        "model": model,
        "split_seed": split_seed,
        "model_seed": model_seed,
    }

    for metric in METRICS:
        if metric not in metrics:
            raise ValueError(f"Missing metric {metric}: {family}/{model}")

        value = float(metrics[metric])

        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid metric {metric}: {family}/{model}")

        record[metric] = value

    return record


def collect_validation_results(root: Path, config: dict) -> list[dict]:
    records = []

    split_seeds = config["data"]["split_seeds"]
    model_seeds = config["training"]["model_seeds"]

    base_names = [
        "tfidf",
        *[embedding_alias(model) for model in config["embeddings"]["models"]],
        *config["graphs"]["names"],
    ]

    fusion_names = list(get_configurations(config))

    for split_seed in split_seeds:
        base_dir = root / "reports" / "base_models" / f"split_seed_{split_seed}"

        base_report = read_report(base_dir / "summary.json")

        if set(base_report.get("models", {})) != set(base_names):
            raise ValueError(f"Incomplete base results for split {split_seed}")

        for model in base_names:
            records.append(
                make_record(
                    "base",
                    model,
                    split_seed,
                    base_report["models"][model]["validation"],
                )
            )

        historical_dir = root / "reports" / "historical_baseline" / f"split_seed_{split_seed}"

        historical_report = read_report(historical_dir / "summary.json")

        records.append(
            make_record(
                "historical",
                "tfidf_soft_voting",
                split_seed,
                historical_report["validation"],
            )
        )

        for family in ("early_fusion", "stacking"):
            directory = root / "reports" / family / f"split_seed_{split_seed}"

            report = read_report(directory / "summary.json")

            if set(report.get("models", {})) != set(fusion_names):
                raise ValueError(f"Incomplete {family} results for split {split_seed}")

            for model in fusion_names:
                records.append(
                    make_record(
                        family,
                        model,
                        split_seed,
                        report["models"][model]["validation"],
                    )
                )

        for model_seed in model_seeds:
            path = (
                root
                / "reports"
                / "bertimbau"
                / f"split_seed_{split_seed}_model_seed_{model_seed}.json"
            )

            report = read_report(path)

            if report.get("split_seed") != split_seed or report.get("model_seed") != model_seed:
                raise ValueError(f"Invalid BERTimbau report: {path}")

            records.append(
                make_record(
                    "bertimbau",
                    "bertimbau-large",
                    split_seed,
                    report["validation_metrics"],
                    model_seed=model_seed,
                )
            )

    return records


def aggregate_results(
    records: list[dict],
    split_seeds: list[int],
    model_seeds: list[int],
) -> list[dict]:
    grouped = defaultdict(list)

    for record in records:
        grouped[(record["family"], record["model"])].append(record)

    results = []

    for (family, model), model_records in sorted(grouped.items()):
        by_split = defaultdict(list)

        for record in model_records:
            by_split[record["split_seed"]].append(record)

        if set(by_split) != set(split_seeds):
            raise ValueError(f"Missing partitions for {family}/{model}")

        result = {
            "family": family,
            "model": model,
            "split_count": len(split_seeds),
            "runs": len(model_records),
        }

        for split_seed, split_records in by_split.items():
            observed_seeds = [record["model_seed"] for record in split_records]

            expected_seeds = model_seeds if family == "bertimbau" else [None]

            if len(observed_seeds) != len(expected_seeds) or set(observed_seeds) != set(
                expected_seeds
            ):
                raise ValueError(f"Unexpected model seeds for {family}/{model}, split {split_seed}")

        for metric in METRICS:
            split_means = [
                mean(record[metric] for record in by_split[seed]) for seed in split_seeds
            ]

            result[f"{metric}_mean"] = mean(split_means)

            result[f"{metric}_std_between_splits"] = (
                stdev(split_means) if len(split_means) > 1 else 0.0
            )

            if family == "bertimbau":
                within_split_stds = [
                    stdev(record[metric] for record in by_split[seed]) for seed in split_seeds
                ]

                result[f"{metric}_mean_seed_std"] = mean(within_split_stds)

        results.append(result)

    return results


def save_csv(path: Path, records: list[dict]) -> None:
    if not records:
        raise ValueError("Cannot write an empty report")

    path.parent.mkdir(parents=True, exist_ok=True)

    columns = list(dict.fromkeys(key for record in records for key in record))

    temporary = path.with_suffix(".csv.tmp")

    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)

    temporary.replace(path)


def run_validation_report(
    project_root: Path | None = None,
) -> None:
    root = (project_root or Path.cwd()).resolve()
    config = load_experiment_config(root)

    records = collect_validation_results(root, config)

    summary = aggregate_results(
        records,
        config["data"]["split_seeds"],
        config["training"]["model_seeds"],
    )

    directory = root / "reports" / "validation"

    save_csv(directory / "runs.csv", records)
    save_csv(directory / "summary.csv", summary)

    write_json(
        directory / "summary.json",
        {
            "evaluation_split": "validation",
            "split_strategy": "pair_controlled",
            "split_seeds": config["data"]["split_seeds"],
            "model_seeds": config["training"]["model_seeds"],
            "results": summary,
        },
    )

    print(f"Validation report saved to {directory}")
