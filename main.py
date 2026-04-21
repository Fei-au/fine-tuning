from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoModelForSequenceClassification

from src.inference import predict_texts
from src.metrics import build_confusion_matrix
from src.pipeline import build_trainer, get_device_summary, load_config


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Modern text classification training pipeline")
	parser.add_argument("--config", default="config.yaml", help="Path to YAML configuration file")

	subparsers = parser.add_subparsers(dest="command", required=True)

	train_parser = subparsers.add_parser("train", help="Train and evaluate model")
	train_parser.add_argument(
		"--smoke",
		action="store_true",
		help="Run a tiny smoke training (1 epoch, small subset)",
	)

	eval_parser = subparsers.add_parser("eval", help="Evaluate model checkpoint")
	eval_parser.add_argument(
		"--checkpoint",
		default=None,
		help="Optional checkpoint path; defaults to best model in output dir",
	)

	pred_parser = subparsers.add_parser("predict", help="Run inference on texts")
	pred_parser.add_argument(
		"--model-dir",
		default=None,
		help="Path to trained model directory; defaults to training.output_dir",
	)
	pred_parser.add_argument(
		"--text",
		action="append",
		required=True,
		help="Input text (repeat --text for multiple samples)",
	)

	return parser.parse_args()


def train(cfg: dict, smoke: bool = False) -> None:
	if smoke:
		cfg["training"]["epochs"] = 1
		cfg["training"]["logging_steps"] = 5

	artifacts = build_trainer(cfg)
	trainer = artifacts.trainer

	eval_dataset = artifacts.tokenized_datasets["validation"]
	test_dataset = artifacts.tokenized_datasets["test"]

	if smoke:
		train_size = min(256, len(artifacts.tokenized_datasets["train"]))
		val_size = min(256, len(artifacts.tokenized_datasets["validation"]))
		test_size = min(256, len(artifacts.tokenized_datasets["test"]))
		train_subset = artifacts.tokenized_datasets["train"].select(range(train_size))
		val_subset = artifacts.tokenized_datasets["validation"].select(range(val_size))
		test_subset = artifacts.tokenized_datasets["test"].select(range(test_size))
		trainer.train_dataset = train_subset
		trainer.eval_dataset = val_subset
		eval_dataset = val_subset
		test_dataset = test_subset

	train_result = trainer.train()
	trainer.save_model()

	eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)
	test_metrics = trainer.evaluate(eval_dataset=test_dataset)

	preds_output = trainer.predict(test_dataset)
	preds = np.argmax(preds_output.predictions, axis=-1)
	labels = preds_output.label_ids

	report = {
		"device": get_device_summary(),
		"train_metrics": train_result.metrics,
		"validation_metrics": eval_metrics,
		"test_metrics": test_metrics,
		"confusion_matrix": build_confusion_matrix(preds, labels),
	}

	output_dir = Path(cfg["training"]["output_dir"])
	with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
		json.dump(report, f, ensure_ascii=True, indent=2)

	print(json.dumps(report, ensure_ascii=True, indent=2))


def evaluate(cfg: dict, checkpoint: str | None) -> None:
	artifacts = build_trainer(cfg)
	trainer = artifacts.trainer

	if checkpoint:
		trainer.model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
		metrics = trainer.evaluate(eval_dataset=artifacts.tokenized_datasets["test"], metric_key_prefix="test")
		print(json.dumps({"checkpoint": checkpoint, "metrics": metrics}, ensure_ascii=True, indent=2))
		return

	metrics = trainer.evaluate(eval_dataset=artifacts.tokenized_datasets["test"], metric_key_prefix="test")
	print(json.dumps({"metrics": metrics}, ensure_ascii=True, indent=2))


def predict(cfg: dict, model_dir: str | None, texts: list[str]) -> None:
	model_path = model_dir or cfg["training"]["output_dir"]
	results = predict_texts(model_path, texts)
	print(json.dumps({"predictions": results}, ensure_ascii=True, indent=2))


def main() -> None:
	args = parse_args()
	cfg = load_config(args.config)

	if args.command == "train":
		train(cfg, smoke=args.smoke)
		return

	if args.command == "eval":
		evaluate(cfg, checkpoint=args.checkpoint)
		return

	if args.command == "predict":
		predict(cfg, model_dir=args.model_dir, texts=args.text)
		return

	raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
	main()
