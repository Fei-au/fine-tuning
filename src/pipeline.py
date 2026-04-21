from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.metrics import build_compute_metrics


@dataclass
class RunArtifacts:
    trainer: Trainer
    tokenized_datasets: DatasetDict
    label_names: list[str]


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def load_and_prepare_dataset(cfg: dict[str, Any]) -> tuple[DatasetDict, list[str]]:
    data_cfg = cfg["data"]
    raw = load_dataset(data_cfg["dataset_name"], data_cfg.get("dataset_config"))

    if "validation" not in raw:
        split = raw["train"].train_test_split(
            test_size=data_cfg.get("validation_split", 0.1),
            seed=cfg["seed"],
        )
        raw = DatasetDict(
            {
                "train": split["train"],
                "validation": split["test"],
                "test": raw.get("test", split["test"]),
            }
        )

    text_col = data_cfg["text_column"]
    label_col = data_cfg["label_column"]

    train_features = raw["train"].features[label_col]
    if hasattr(train_features, "names"):
        label_names = list(train_features.names)
    else:
        all_labels = sorted(set(raw["train"][label_col]))
        label_names = [str(label) for label in all_labels]

    for split_name in ("train", "validation", "test"):
        if split_name not in raw:
            continue
        required = {text_col, label_col}
        missing = required - set(raw[split_name].column_names)
        if missing:
            raise ValueError(f"Missing required columns {missing} in split {split_name}")

    return raw, label_names


def build_trainer(cfg: dict[str, Any]) -> RunArtifacts:
    seed_everything(cfg["seed"])

    raw, label_names = load_and_prepare_dataset(cfg)
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_name"])

    text_col = data_cfg["text_column"]
    label_col = data_cfg["label_column"]

    def preprocess(batch: dict[str, Any]) -> dict[str, Any]:
        tokens = tokenizer(
            batch[text_col],
            truncation=True,
            max_length=data_cfg["max_length"],
        )
        tokens["labels"] = batch[label_col]
        return tokens

    tokenized = raw.map(
        preprocess,
        batched=True,
        remove_columns=raw["train"].column_names,
        desc="Tokenizing dataset",
    )

    id2label = {idx: name for idx, name in enumerate(label_names)}
    label2id = {name: idx for idx, name in id2label.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["model_name"],
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
    )

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=train_cfg["learning_rate"],
        per_device_train_batch_size=train_cfg["train_batch_size"],
        per_device_eval_batch_size=train_cfg["eval_batch_size"],
        num_train_epochs=train_cfg["epochs"],
        weight_decay=train_cfg["weight_decay"],
        warmup_ratio=train_cfg["warmup_ratio"],
        logging_steps=train_cfg["logging_steps"],
        eval_strategy=train_cfg["evaluation_strategy"],
        save_strategy=train_cfg["save_strategy"],
        save_total_limit=train_cfg["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model=train_cfg["metric_for_best_model"],
        greater_is_better=True,
        report_to=train_cfg["report_to"],
        fp16=bool(train_cfg.get("fp16", False) and torch.cuda.is_available()),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=build_compute_metrics(),
    )

    # Keep preprocessing metadata next to checkpoints for reproducible inference.
    with open(output_dir / "label_names.json", "w", encoding="utf-8") as f:
        json.dump(label_names, f, ensure_ascii=True, indent=2)

    tokenizer.save_pretrained(output_dir / "tokenizer")

    return RunArtifacts(trainer=trainer, tokenized_datasets=tokenized, label_names=label_names)


def get_device_summary() -> dict[str, Any]:
    if torch.cuda.is_available():
        return {
            "device": "cuda",
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_name": torch.cuda.get_device_name(0),
        }
    return {"device": "cpu"}
