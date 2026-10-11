from collections.abc import Callable
from dataclasses import dataclass

from pun_stacking.data.audit import audit_data
from pun_stacking.data.folds import build_folds
from pun_stacking.early_fusion.train import run_early_fusion
from pun_stacking.embeddings.extract import run_embeddings
from pun_stacking.evaluation.protocol import freeze_protocol
from pun_stacking.evaluation.validation import run_validation_report
from pun_stacking.graphs.runner import build_final_graphs
from pun_stacking.models.base import run_base_models
from pun_stacking.models.bertimbau import run_bertimbau
from pun_stacking.models.historical import run_historical_baseline
from pun_stacking.stacking.train import run_stacking


@dataclass(frozen=True)
class Stage:
    name: str
    description: str
    function: Callable[[], None]


def prepare_data() -> None:
    audit_data()


def prepare_folds() -> None:
    build_folds()


def build_graphs() -> None:
    build_final_graphs()


def train_historical_baseline(split_seed: int | None = None) -> None:
    run_historical_baseline(split_seed=split_seed)


def train_bertimbau(
    split_seed: int | None = None,
    model_seed: int | None = None,
) -> None:
    run_bertimbau(split_seed=split_seed, model_seed=model_seed)


def extract_embeddings(model_name: str | None = None) -> None:
    run_embeddings(selected_model=model_name)


def train_base_models(split_seed: int | None = None) -> None:
    run_base_models(split_seed=split_seed)


def train_early_fusion(split_seed: int | None = None) -> None:
    run_early_fusion(split_seed=split_seed)


def train_stacking(split_seed: int | None = None) -> None:
    run_stacking(split_seed=split_seed)


def generate_validation_report() -> None:
    run_validation_report()


def freeze_experimental_protocol() -> None:
    freeze_protocol()


def evaluate() -> None:
    raise NotImplementedError("Final test evaluation is not implemented")


STAGES = (
    Stage("data", "Prepare and validate dataset", prepare_data),
    Stage("folds", "Create pair-grouped OOF folds", prepare_folds),
    Stage(
        "historical-baseline",
        "Train historical TF-IDF soft-voting ensemble",
        train_historical_baseline,
    ),
    Stage("graphs", "Build knowledge graphs", build_graphs),
    Stage("bertimbau", "Train BERTimbau Large", train_bertimbau),
    Stage("embeddings", "Extract sentence embeddings", extract_embeddings),
    Stage("base-models", "Train base classifiers", train_base_models),
    Stage("early-fusion", "Run early fusion experiments", train_early_fusion),
    Stage("stacking", "Run stacking experiments", train_stacking),
    Stage(
        "validation-report",
        "Consolidate validation results",
        generate_validation_report,
    ),
    Stage("evaluation", "Run final test evaluation", evaluate),
    Stage(
        "freeze-protocol",
        "Freeze experimental evaluation protocol",
        freeze_experimental_protocol,
    ),
)


def run_stage(
    name: str,
    split_seed: int | None = None,
    model_seed: int | None = None,
    embedding: str | None = None,
) -> None:
    for stage in STAGES:
        if stage.name != name:
            continue

        print(f"Running: {stage.description}")

        if name == "bertimbau":
            if embedding is not None:
                raise ValueError("Embedding filter is not supported")

            train_bertimbau(
                split_seed=split_seed,
                model_seed=model_seed,
            )
            return

        if name == "embeddings":
            if split_seed is not None or model_seed is not None:
                raise ValueError("Embedding extraction does not use seeds")

            extract_embeddings(model_name=embedding)
            return

        split_filtered_stages = {
            "historical-baseline": train_historical_baseline,
            "base-models": train_base_models,
            "early-fusion": train_early_fusion,
            "stacking": train_stacking,
        }

        if name in split_filtered_stages:
            if model_seed is not None or embedding is not None:
                raise ValueError("Unsupported filters for this stage")

            split_filtered_stages[name](split_seed=split_seed)
            return

        if split_seed is not None or model_seed is not None or embedding is not None:
            raise ValueError("Unsupported filters for this stage")

        stage.function()
        return

    raise ValueError(f"Unknown stage: {name}")


def run_pipeline() -> None:
    excluded = {"freeze-protocol", "evaluation"}

    for stage in STAGES:
        if stage.name not in excluded:
            run_stage(stage.name)
