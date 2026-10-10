import csv
import json
import tomllib
from pathlib import Path

import joblib
import networkx as nx

from pun_stacking.data.audit import inspect_splits, load_jsonl, pair_id
from pun_stacking.graphs.builders import GRAPH_NAMES, build_graph_set


def load_experiment_config(root: Path) -> dict:
    path = root / "configs" / "experiments.toml"

    with path.open("rb") as stream:
        return tomllib.load(stream)


def graph_directory(root: Path, split_seed: int) -> Path:
    return root / "artifacts" / "graphs" / "pair_controlled" / f"seed_{split_seed}" / "final"


def save_graph_set(
    graphs: dict[str, nx.Graph],
    directory: Path,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)

    for name, graph in graphs.items():
        joblib.dump(graph, directory / f"{name}.joblib")


def load_final_graphs(
    split_seed: int,
    project_root: Path | None = None,
) -> dict[str, nx.Graph]:
    root = (project_root or Path.cwd()).resolve()
    directory = graph_directory(root, split_seed)

    graphs = {}

    for name in GRAPH_NAMES:
        path = directory / f"{name}.joblib"

        if not path.is_file():
            raise FileNotFoundError(f"Missing graph for split seed {split_seed}: {path}")

        graphs[name] = joblib.load(path)

    return graphs


def graph_statistics(
    graphs: dict[str, nx.Graph],
) -> dict:
    return {
        name: {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "components": nx.number_connected_components(graph),
            "density": nx.density(graph),
        }
        for name, graph in graphs.items()
    }


def build_final_graphs(
    project_root: Path | None = None,
) -> Path:
    root = (project_root or Path.cwd()).resolve()
    config = load_experiment_config(root)

    data_root = root / config["data"].get("root", "data")
    split_seeds = config["data"]["split_seeds"]
    graph_config = config["graphs"]

    reports = []

    for seed in split_seeds:
        split_directory = data_root / "pair_controlled" / f"seed_{seed}"

        inspect_splits(split_directory, require_pairs=True)

        train_rows = load_jsonl(split_directory / "train.jsonl")

        graphs = build_graph_set(
            train_rows,
            graph_config,
        )

        directory = graph_directory(root, seed)
        save_graph_set(graphs, directory)

        statistics = graph_statistics(graphs)

        reports.append(
            {
                "split_seed": seed,
                "training_examples": len(train_rows),
                "graphs": statistics,
                "artifact_directory": str(directory.relative_to(root)),
            }
        )

        print(f"Split seed {seed}")

        for name, values in statistics.items():
            print(f"  {name}: {values['nodes']} nodes, {values['edges']} edges")

    report = {
        "strategy": "pair_controlled",
        "construction_scope": "external_training_only",
        "graph_configuration": graph_config,
        "partitions": reports,
    }

    output_path = root / "reports" / "graphs_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Report: {output_path}")

    return output_path


def build_fold_graphs(
    split_seed: int,
    fold: int,
    project_root: Path | None = None,
) -> tuple[dict[str, nx.Graph], list[dict], list[dict]]:
    root = (project_root or Path.cwd()).resolve()
    config = load_experiment_config(root)

    if split_seed not in config["data"]["split_seeds"]:
        raise ValueError(f"Unknown split seed: {split_seed}")

    n_splits = config["cross_validation"]["n_splits"]

    if fold not in range(n_splits):
        raise ValueError(f"Invalid fold: {fold}")

    data_root = root / config["data"].get("root", "data")

    train_path = data_root / "pair_controlled" / f"seed_{split_seed}" / "train.jsonl"

    fold_path = data_root / "folds" / f"seed_{split_seed}.csv"

    rows = load_jsonl(train_path)

    assignments = {}

    with fold_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)

        if reader.fieldnames != ["pair_id", "fold"]:
            raise ValueError("Unexpected fold assignment columns")

        for item in reader:
            group = item["pair_id"]

            if group in assignments:
                raise ValueError(f"Duplicate fold assignment: {group}")

            assigned_fold = int(item["fold"])

            if assigned_fold not in range(n_splits):
                raise ValueError(f"Invalid assigned fold: {assigned_fold}")

            assignments[group] = assigned_fold

    expected_groups = {pair_id(row["id"]) for row in rows}

    if set(assignments) != expected_groups:
        raise ValueError("Fold assignments do not match the external training pairs")

    fold_train = [row for row in rows if assignments[pair_id(row["id"])] != fold]

    fold_validation = [row for row in rows if assignments[pair_id(row["id"])] == fold]

    if not fold_train or not fold_validation:
        raise ValueError(f"Empty OOF partition: {fold}")

    if len(fold_train) + len(fold_validation) != len(rows):
        raise ValueError("OOF partition sizes are inconsistent")

    graphs = build_graph_set(
        fold_train,
        config["graphs"],
    )

    return graphs, fold_train, fold_validation
