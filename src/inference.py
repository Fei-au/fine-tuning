from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def predict_texts(model_dir: str, texts: list[str]) -> list[dict[str, float | str]]:
    path = Path(model_dir)
    tokenizer_path = path / "tokenizer"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path if tokenizer_path.exists() else path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()

    encoded = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**encoded).logits

    probs = torch.softmax(logits, dim=-1)
    confs, labels = torch.max(probs, dim=-1)

    results: list[dict[str, float | str]] = []
    for idx, (label_id, confidence) in enumerate(zip(labels.tolist(), confs.tolist())):
        label_name = model.config.id2label.get(label_id, str(label_id))
        results.append(
            {
                "text": texts[idx],
                "label": label_name,
                "confidence": float(confidence),
            }
        )
    return results
