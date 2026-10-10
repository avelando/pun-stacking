from collections.abc import Callable
from dataclasses import dataclass

from pun_stacking.data.audit import audit_data
from pun_stacking.data.folds import build_folds
from pun_stacking.graphs.runner import build_final_graphs


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


def train_bertimbau(
    split_seed: int | None = None,
    model_seed: int | None = None,
) -> None:
    from pun_stacking.models.bertimbau import run_bertimbau

    run_bertimbau(
        split_seed=split_seed,
        model_seed=model_seed,
    )


def extract_embeddings(model_name: str | None = None) -> None:
    from pun_stacking.embeddings.extract import run_embeddings

    run_embeddings(selected_model=model_name)


def train_base_models(split_seed: int | None = None) -> None:
    from pun_stacking.models.base import run_base_models

    run_base_models(split_seed=split_seed)


def train_early_fusion() -> None:
    raise NotImplementedError("Early fusion is not implemented")


def train_stacking() -> None:
    raise NotImplementedError("Stacking is not implemented")


def evaluate() -> None:
    raise NotImplementedError("Evaluation is not implemented")


STAGES = (
    Stage("data", "Prepare and validate dataset", prepare_data),
    Stage("folds", "Create pair-grouped OOF folds", prepare_folds),
    Stage("graphs", "Build knowledge graphs", build_graphs),
    Stage("bertimbau", "Train BERTimbau Large", train_bertimbau),
    Stage("embeddings", "Extract sentence embeddings", extract_embeddings),
    Stage("base-models", "Train base classifiers", train_base_models),
    Stage("early-fusion", "Run early fusion experiments", train_early_fusion),
    Stage("stacking", "Run stacking experiments", train_stacking),
    Stage("evaluation", "Evaluate models and generate reports", evaluate),
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
                raise ValueError("Embedding selection is not supported for BERTimbau")

            train_bertimbau(
                split_seed=split_seed,
                model_seed=model_seed,
            )
            return

        if name == "embeddings":
            if split_seed is not None or model_seed is not None:
                raise ValueError("Embedding extraction does not use split or model seeds")

            extract_embeddings(model_name=embedding)
            return

        if name == "base-models":
            if model_seed is not None or embedding is not None:
                raise ValueError("Base models support only the split seed filter")

            train_base_models(split_seed=split_seed)
            return

        if split_seed is not None or model_seed is not None or embedding is not None:
            raise ValueError("Unsupported filters for this pipeline stage")

        stage.function()
        return

    raise ValueError(f"Unknown stage: {name}")


def run_pipeline() -> None:
    for stage in STAGES:
        run_stage(stage.name)
