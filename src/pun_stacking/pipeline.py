from dataclasses import dataclass
from collections.abc import Callable

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


def train_bertimbau() -> None:
    raise NotImplementedError("BERTimbau training is not implemented")


def extract_embeddings() -> None:
    raise NotImplementedError("Embedding extraction is not implemented")


def train_base_models() -> None:
    raise NotImplementedError("Base model training is not implemented")


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


def run_stage(name: str) -> None:
    for stage in STAGES:
        if stage.name == name:
            print(f"Running: {stage.description}")
            stage.function()
            return

    raise ValueError(f"Unknown stage: {name}")


def run_pipeline() -> None:
    for stage in STAGES:
        run_stage(stage.name)
