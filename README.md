# Fine-Tuning Practice Project

This project gives you a modern, common workflow for training an NLP model yourself with minimal boilerplate. It uses PyTorch + Hugging Face Trainer and covers the full path:

1. dataset loading and split handling
2. tokenization and dynamic padding
3. pretrained embeddings/model fine-tuning
4. optimization and scheduling
5. validation/test evaluation
6. checkpointing best model
7. inference on custom text

Default task: text classification on AG News with `distilbert-base-uncased`.

## Project Structure

- `main.py`: CLI entrypoint (`train`, `eval`, `predict`)
- `config.yaml`: training/data/model configuration
- `src/pipeline.py`: dataset + tokenizer + model + Trainer assembly
- `src/metrics.py`: accuracy/F1 and confusion matrix helpers
- `src/inference.py`: standalone prediction helper

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Train

Run a full baseline training:

```powershell
python main.py --config config.yaml train
```

Run a smoke test (tiny subset, 1 epoch):

```powershell
python main.py --config config.yaml train --smoke
```

## Evaluate

Evaluate using model state from output directory:

```powershell
python main.py --config config.yaml eval
```

Evaluate a specific checkpoint:

```powershell
python main.py --config config.yaml eval --checkpoint outputs/ag_news_distilbert/checkpoint-1000
```

## Predict

```powershell
python main.py --config config.yaml predict --text "The stock market rallied today." --text "The team won the final match."
```

Optionally pick a different model directory:

```powershell
python main.py --config config.yaml predict --model-dir outputs/ag_news_distilbert --text "NASA announced a new mission."
```

## Output Artifacts

Under `training.output_dir` (default `outputs/ag_news_distilbert`), you will get:

- model checkpoints
- tokenizer files
- `label_names.json`
- `metrics_summary.json` with train/validation/test metrics and confusion matrix

## Notes

- `fp16` is enabled by config and is only used when CUDA is available.
- To switch datasets or models, edit `config.yaml`.
