import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import transformers
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

from pun_stacking.data.audit import load_jsonl, pair_id
from pun_stacking.graphs.runner import load_experiment_config


class PunDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        tokenizer,
        max_length: int,
    ):
        self.encodings = tokenizer(
            [row["text"] for row in rows],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.labels = [row["label"] for row in rows]

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        item = {key: values[index] for key, values in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def classification_metrics(logits, labels) -> dict:
    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = np.argmax(logits, axis=1)

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def compute_metrics(prediction) -> dict:
    return classification_metrics(
        prediction.predictions,
        prediction.label_ids,
    )


def calculate_pair_accuracy(
    rows: list[dict],
    predictions: np.ndarray,
) -> float:
    if len(rows) != len(predictions):
        raise ValueError("Prediction count does not match dataset size")

    groups = {}

    for row, prediction in zip(rows, predictions, strict=True):
        group = pair_id(row["id"])

        groups.setdefault(group, []).append(int(prediction) == row["label"])

    if not groups or any(len(values) != 2 for values in groups.values()):
        raise ValueError("Expected exactly two instances per pair")

    return float(np.mean([all(values) for values in groups.values()]))


def save_validation_predictions(
    path: Path,
    rows: list[dict],
    probabilities: np.ndarray,
) -> None:
    predictions = np.argmax(probabilities, axis=1)

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)

        writer.writerow(["id", "label", "prediction", "p_non_pun", "p_pun"])

        for row, prediction, probability in zip(
            rows,
            predictions,
            probabilities,
            strict=True,
        ):
            writer.writerow(
                [
                    row["id"],
                    row["label"],
                    int(prediction),
                    float(probability[0]),
                    float(probability[1]),
                ]
            )


def train_single_run(
    root: Path,
    config: dict,
    split_seed: int,
    model_seed: int,
) -> dict:
    settings = config["bertimbau"]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for BERTimbau training")

    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is not supported by the active GPU")

    data_root = root / config["data"].get("root", "data")

    split_dir = data_root / "pair_controlled" / f"seed_{split_seed}"

    train_rows = load_jsonl(split_dir / "train.jsonl")
    validation_rows = load_jsonl(split_dir / "validation.jsonl")

    if len(train_rows) != 3990 or len(validation_rows) != 570:
        raise ValueError("Unexpected external partition sizes")

    train_pairs = {pair_id(row["id"]) for row in train_rows}
    validation_pairs = {pair_id(row["id"]) for row in validation_rows}

    if train_pairs & validation_pairs:
        raise ValueError("Training and validation contain crossing pairs")

    run_dir = (
        root / "artifacts" / "bertimbau" / f"split_seed_{split_seed}" / f"model_seed_{model_seed}"
    )

    checkpoint_dir = run_dir / "checkpoints"
    best_model_dir = run_dir / "best_model"

    report_dir = root / "reports" / "bertimbau"
    report_path = report_dir / f"split_seed_{split_seed}_model_seed_{model_seed}.json"

    prediction_path = report_dir / f"split_seed_{split_seed}_model_seed_{model_seed}_validation.csv"

    if report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))

        if (
            previous.get("training_config") != settings
            or previous.get("split_seed") != split_seed
            or previous.get("model_seed") != model_seed
        ):
            raise RuntimeError(f"Previous run has incompatible configuration: {report_path}")

        if not (best_model_dir / "config.json").is_file():
            raise RuntimeError(f"Completed run is missing its model: {best_model_dir}")

        if not prediction_path.is_file():
            raise RuntimeError(f"Completed run is missing predictions: {prediction_path}")

        print(f"Skipping completed run: split={split_seed}, model={model_seed}")

        return previous

    run_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    run_config_path = run_dir / "run_config.json"

    run_config = {
        "split_seed": split_seed,
        "model_seed": model_seed,
        "training_config": settings,
    }

    if run_config_path.exists():
        previous_config = json.loads(run_config_path.read_text(encoding="utf-8"))

        if previous_config != run_config:
            raise RuntimeError(f"Incompatible interrupted run: {run_dir}")
    else:
        if any(checkpoint_dir.glob("checkpoint-*")):
            raise RuntimeError(f"Checkpoints exist without run configuration: {run_dir}")

        run_config_path.write_text(
            json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    set_seed(model_seed)

    tokenizer = AutoTokenizer.from_pretrained(
        settings["model"],
        use_fast=True,
    )

    train_dataset = PunDataset(
        train_rows,
        tokenizer,
        settings["max_length"],
    )

    validation_dataset = PunDataset(
        validation_rows,
        tokenizer,
        settings["max_length"],
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        settings["model"],
        num_labels=2,
        id2label={0: "non_pun", 1: "pun"},
        label2id={"non_pun": 0, "pun": 1},
    )

    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=settings["epochs"],
        learning_rate=settings["learning_rate"],
        per_device_train_batch_size=settings["batch_size"],
        per_device_eval_batch_size=settings["evaluation_batch_size"],
        gradient_accumulation_steps=settings["gradient_accumulation_steps"],
        weight_decay=settings["weight_decay"],
        warmup_ratio=settings["warmup_ratio"],
        max_grad_norm=settings["max_grad_norm"],
        bf16=True,
        fp16=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,
        save_total_limit=settings["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        seed=model_seed,
        data_seed=model_seed,
        optim="adamw_torch",
        lr_scheduler_type="linear",
        report_to="none",
        save_safetensors=True,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
        ),
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=settings["early_stopping_patience"])
        ],
    )

    checkpoint = get_last_checkpoint(str(checkpoint_dir))

    print(f"Training BERTimbau: split_seed={split_seed}, model_seed={model_seed}")

    if checkpoint:
        print(f"Resuming from {checkpoint}")

    trainer.train(resume_from_checkpoint=checkpoint)

    best_checkpoint = trainer.state.best_model_checkpoint

    if best_checkpoint is None:
        raise RuntimeError("Training finished without a best checkpoint")

    result = trainer.predict(validation_dataset)

    logits = result.predictions

    if isinstance(logits, tuple):
        logits = logits[0]

    probabilities = softmax(logits, axis=1)
    predictions = np.argmax(probabilities, axis=1)

    labels = np.array(
        [row["label"] for row in validation_rows],
        dtype=np.int64,
    )

    metrics = classification_metrics(logits, labels)

    metrics["pair_exact_accuracy"] = calculate_pair_accuracy(
        validation_rows,
        predictions,
    )

    trainer.save_model(str(best_model_dir))

    if not (best_model_dir / "config.json").is_file():
        raise RuntimeError("Best model was not saved correctly")

    save_validation_predictions(
        prediction_path,
        validation_rows,
        probabilities,
    )

    report = {
        "model": settings["model"],
        "split_strategy": "pair_controlled",
        "split_seed": split_seed,
        "model_seed": model_seed,
        "training_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "training_config": settings,
        "validation_metrics": metrics,
        "best_validation_f1_macro": float(trainer.state.best_metric),
        "best_checkpoint_step": int(Path(best_checkpoint).name.rsplit("-", 1)[1]),
        "training_global_step": int(trainer.state.global_step),
        "pytorch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "model_directory": str(best_model_dir.relative_to(root)),
        "validation_predictions": str(prediction_path.relative_to(root)),
    }

    temporary_report = report_path.with_suffix(".json.tmp")

    temporary_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    temporary_report.replace(report_path)

    if settings["cleanup_checkpoints_after_success"]:
        shutil.rmtree(checkpoint_dir)

    print(
        f"Finished: split={split_seed}, model={model_seed}, "
        f"validation_accuracy={metrics['accuracy']:.4f}, "
        f"validation_f1_macro={metrics['f1_macro']:.4f}, "
        f"pair_exact={metrics['pair_exact_accuracy']:.4f}"
    )

    del trainer
    del model

    torch.cuda.empty_cache()

    return report


def run_bertimbau(
    split_seed: int | None = None,
    model_seed: int | None = None,
    project_root: Path | None = None,
) -> None:
    root = (project_root or Path.cwd()).resolve()
    config = load_experiment_config(root)

    if not config["bertimbau"]["enabled"]:
        raise ValueError("BERTimbau is disabled in the configuration")

    split_seeds = config["data"]["split_seeds"]
    model_seeds = config["training"]["model_seeds"]

    if split_seed is not None:
        if split_seed not in split_seeds:
            raise ValueError(f"Unknown split seed: {split_seed}")

        split_seeds = [split_seed]

    if model_seed is not None:
        if model_seed not in model_seeds:
            raise ValueError(f"Unknown model seed: {model_seed}")

        model_seeds = [model_seed]

    for current_split_seed in split_seeds:
        for current_model_seed in model_seeds:
            train_single_run(
                root,
                config,
                current_split_seed,
                current_model_seed,
            )
