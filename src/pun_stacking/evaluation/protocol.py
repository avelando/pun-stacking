import json
from pathlib import Path

from pun_stacking.evaluation.validation import (
    aggregate_results,
    collect_validation_results,
)
from pun_stacking.graphs.runner import load_experiment_config
from pun_stacking.models.base import embedding_alias
from pun_stacking.stacking.train import get_configurations


def build_evaluation_plan(config: dict) -> dict:
    split_seeds = config["data"]["split_seeds"]
    model_seeds = config["training"]["model_seeds"]

    embeddings = [embedding_alias(model) for model in config["embeddings"]["models"]]

    graphs = list(config["graphs"]["names"])
    configurations = get_configurations(config)

    base_models = ["tfidf", *embeddings, *graphs]

    comparisons = []

    for embedding in embeddings:
        comparisons.append(
            {
                "reference": f"stacking/{embedding}_only",
                "candidate": f"stacking/{embedding}_graphs",
            }
        )

        comparisons.append(
            {
                "reference": f"early_fusion/{embedding}_only",
                "candidate": f"early_fusion/{embedding}_graphs",
            }
        )

    if len(comparisons) != 8:
        raise ValueError("Expected eight primary comparisons")

    if len(split_seeds) != 6 or len(set(split_seeds)) != 6:
        raise ValueError("Expected six unique split seeds")

    if len(model_seeds) != 6 or len(set(model_seeds)) != 6:
        raise ValueError("Expected six unique model seeds")

    if config["evaluation"]["primary_metric"] != "accuracy":
        raise ValueError("Unexpected primary evaluation metric")

    return {
        "split_strategy": "pair_controlled",
        "split_seeds": split_seeds,
        "model_seeds": model_seeds,
        "selection_rule": "all_predeclared_configurations",
        "primary_metric": "accuracy",
        "secondary_metrics": [
            "f1_macro",
            "pair_exact_accuracy",
        ],
        "decision_threshold": 0.5,
        "bertimbau_checkpoint_metric": "eval_f1_macro",
        "bertimbau_seed_aggregation": "mean_within_split",
        "models": {
            "base": base_models,
            "historical": ["tfidf_soft_voting"],
            "early_fusion": list(configurations),
            "stacking": list(configurations),
            "bertimbau": ["bertimbau-large"],
        },
        "primary_comparisons": comparisons,
        "multiple_comparison_correction": "holm",
        "test_policy": "separate_explicit_execution",
    }


def validate_completed_experiments(
    root: Path,
    config: dict,
) -> None:
    validation_path = root / "reports" / "validation" / "summary.json"

    if not validation_path.is_file():
        raise FileNotFoundError(
            "Validation summary is missing. Run the validation-report stage first."
        )

    saved = json.loads(validation_path.read_text(encoding="utf-8"))

    records = collect_validation_results(root, config)

    expected_summary = aggregate_results(
        records,
        config["data"]["split_seeds"],
        config["training"]["model_seeds"],
    )

    if (
        saved.get("evaluation_split") != "validation"
        or saved.get("split_seeds") != config["data"]["split_seeds"]
        or saved.get("model_seeds") != config["training"]["model_seeds"]
        or saved.get("results") != expected_summary
    ):
        raise RuntimeError(
            "Validation summary is inconsistent with current experiments. "
            "Regenerate it before freezing the protocol."
        )


def freeze_protocol(
    project_root: Path | None = None,
) -> Path:
    root = (project_root or Path.cwd()).resolve()
    config = load_experiment_config(root)

    validate_completed_experiments(root, config)

    frozen = {
        "status": "frozen",
        "experiment_config": config,
        "evaluation_plan": build_evaluation_plan(config),
    }

    destination = root / "configs" / "frozen_protocol.json"

    if destination.exists():
        previous = json.loads(destination.read_text(encoding="utf-8"))

        if previous != frozen:
            raise RuntimeError(
                "A different protocol is already frozen. "
                "Do not overwrite it after final evaluation."
            )

        print("Experimental protocol is already frozen")
        return destination

    temporary = destination.with_suffix(".json.tmp")

    temporary.write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    temporary.replace(destination)

    print(f"Experimental protocol frozen: {destination}")

    return destination
