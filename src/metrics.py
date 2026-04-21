from __future__ import annotations

from typing import Any

import evaluate
import numpy as np
from sklearn.metrics import confusion_matrix


def build_compute_metrics():
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        result_accuracy = accuracy.compute(predictions=preds, references=labels)
        result_f1 = f1.compute(predictions=preds, references=labels, average="macro")
        return {
            "accuracy": float(result_accuracy["accuracy"]),
            "macro_f1": float(result_f1["f1"]),
        }

    return compute_metrics


def build_confusion_matrix(preds: np.ndarray, labels: np.ndarray) -> list[list[int]]:
    matrix = confusion_matrix(labels, preds)
    return matrix.tolist()
