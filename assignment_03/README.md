# Assignment 3: End-to-End HuggingFace Model Training & Docker Deployment

## Overview

This project fine-tunes **DistilBERT** (`distilbert-base-cased`) on Goodreads book reviews to classify them into **8 genres** using the HuggingFace `transformers` library and `Trainer` API.

**🔗 Links:**
- **HuggingFace Model**: [Prakhar54-byte/distilbert-goodreads-genre-classifier](https://huggingface.co/Prakhar54-byte/distilbert-goodreads-genre-classifier)

---

## Model Selection Rationale

**DistilBERT** (`distilbert-base-cased`) was selected because:

1. **Efficiency**: It's a distilled version of BERT — 40% smaller and 60% faster while retaining 97% of BERT's language understanding.
2. **Suitable scale**: The dataset (~6400 train / ~1600 test samples) is small enough that a full BERT model would risk overfitting. DistilBERT provides a good balance.
3. **Cased variant**: Preserves capitalization information, which can be relevant for distinguishing genres in book reviews.
4. **Proven track record**: Widely used for text classification tasks in both research and production environments.

---

## Dataset

- **Source**: [UCSD Goodreads Book Graph](https://mengtingwan.github.io/data/goodreads.html)
- **Genres** (8 classes): poetry, children, comics & graphic, fantasy & paranormal, history & biography, mystery/thriller/crime, romance, young adult
- **Split**: 800 train + 200 test per genre → **6,400 training / 1,600 test** samples
- **Preprocessing**: Reviews streamed from gzip, sampled randomly, tokenized with `DistilBertTokenizerFast` (max 512 tokens)

---

## Training Summary

| Parameter | Value |
|-----------|-------|
| Model | `distilbert-base-cased` |
| Epochs | 3 |
| Batch Size | 2 (effective 10 with gradient accumulation) |
| Gradient Accumulation | 5 steps |
| Learning Rate | 5e-5 |
| Weight Decay | 0.01 |
| Warmup Steps | 100 |
| Precision | FP16 (mixed precision) |
| Training Time | ~18 minutes (NVIDIA GPU) |

---

## Evaluation Results

### Local Model vs HuggingFace Model Comparison

| Metric | Local Model | HuggingFace Model |
|--------|-------------|-------------------|
| Accuracy | 0.6081 | 0.6081 |
| Weighted F1 | 0.6054 | 0.6054 |
| Weighted Precision | 0.6034 | 0.6034 |
| Loss | 1.2691 | 1.2691 |

> ✅ Metrics are identical — the uploaded model on HuggingFace Hub is an exact copy of the locally trained model.

### Per-Genre Classification Report

| Genre | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| children | 0.62 | 0.66 | 0.64 |
| comics_graphic | 0.76 | 0.77 | 0.77 |
| fantasy_paranormal | 0.43 | 0.43 | 0.43 |
| history_biography | 0.59 | 0.60 | 0.60 |
| mystery_thriller_crime | 0.60 | 0.62 | 0.61 |
| poetry | 0.78 | 0.81 | 0.79 |
| romance | 0.62 | 0.61 | 0.61 |
| young_adult | 0.43 | 0.36 | 0.39 |

### Comparison with Baseline

| Model | Accuracy |
|-------|----------|
| TF-IDF + Logistic Regression | ~0.55 |
| **DistilBERT (fine-tuned)** | **0.61** |

The fine-tuned DistilBERT outperforms the TF-IDF baseline by ~6 percentage points.

---

## Project Structure

```
├── src/
│   ├── __init__.py
│   ├── data.py          # Data loading & preprocessing
│   ├── utils.py         # Dataset class, metrics, label maps
│   ├── train.py         # Training pipeline
│   └── eval.py          # Evaluation script
├── Dockerfile           # Training Docker image
├── Dockerfile.eval      # Evaluation-only Docker image
├── requirements.txt     # Python dependencies
├── results/
│   ├── train_metrics.json
│   ├── eval_results_local.json
│   ├── eval_results_hf.json
│   └── eval_comparison.json
└── README.md
```

---

## Docker Instructions

### Build & Run Training Image

```bash
docker build -t ml-assignment-train .
docker run --rm ml-assignment-train
```

### Build & Run Evaluation-Only Image (pulls from HuggingFace)

```bash
docker build -f Dockerfile.eval -t ml-assignment-eval .
docker run --rm ml-assignment-eval
```

---

## Local Setup

```bash
# Create virtual environment (Python 3.10)
python3.10 -m venv .venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Train the model
python -m src.train

# Evaluate locally
python -m src.eval --model_path ./distilbert-goodreads-genre-classifier --source local

# Evaluate from HuggingFace
python -m src.eval --model_path Prakhar54-byte/distilbert-goodreads-genre-classifier --source hf
```

---

## Challenges

1. **GPU Memory**: The NVIDIA GPU had only 3.68 GiB VRAM, causing OOM with the original batch size (10). Solved by using batch_size=2 with gradient_accumulation_steps=5 and FP16 mixed precision.
2. **CPU Training Speed**: CPU training was estimated at ~4 hours. GPU with optimized settings completed in ~18 minutes.
3. **Data Access**: The Goodreads dataset is hosted externally and streamed via HTTP, requiring network access during data preparation.
