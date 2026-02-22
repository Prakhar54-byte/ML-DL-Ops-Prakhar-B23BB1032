# Short Report — Assignment 3

**Student**: Prakhar (B23BB1032)  
**Date**: February 2026  
**HuggingFace Model**: [Prakhar54-byte/distilbert-goodreads-genre-classifier](https://huggingface.co/Prakhar54-byte/distilbert-goodreads-genre-classifier)  
**GitHub**: [ML-DL-Ops-Prakhar-B23BB1032 (Assignment_03 branch)](https://github.com/Prakhar54-byte/ML-DL-Ops-Prakhar-B23BB1032/tree/Assignment_03)

---

## 1. Model Selection

**Model**: `distilbert-base-cased`  
**Task**: Multi-class text classification (8 book genres)

**Why DistilBERT?**
- **Compact & fast** — 40% smaller and 60% faster than BERT, while retaining 97% of its language understanding capability.
- **Right-sized for the data** — With only ~6,400 training samples, a full BERT would risk overfitting. DistilBERT provides a good balance between capacity and generalization.
- **Cased variant** — Preserves capitalization, which can carry genre-relevant signals (e.g., proper nouns in history/biography reviews).
- **Widely supported** — Strong ecosystem support via HuggingFace `transformers` and `Trainer` API.

---

## 2. Training Summary

| Parameter | Value |
|-----------|-------|
| Base Model | `distilbert-base-cased` |
| Dataset | Goodreads book reviews (UCSD Book Graph) |
| Genres | 8 classes (poetry, children, comics/graphic, fantasy/paranormal, history/biography, mystery/thriller/crime, romance, young adult) |
| Train / Test Split | 6,400 / 1,600 samples (800/200 per genre) |
| Epochs | 3 |
| Effective Batch Size | 10 (batch=2 × grad_accum=5) |
| Optimizer | AdamW (lr=5e-5, weight_decay=0.01) |
| Warmup Steps | 100 |
| Precision | FP16 mixed precision |
| Training Time | ~18 minutes (NVIDIA GPU) |
| Training Loss | 4.86 → converged |

---

## 3. Evaluation Comparison

### Local Model vs HuggingFace Hub Model

| Metric | Local Model | HuggingFace Model | Match? |
|--------|:-----------:|:-----------------:|:------:|
| Accuracy | 0.6081 | 0.6081 | ✅ |
| Weighted F1 | 0.6054 | 0.6054 | ✅ |
| Weighted Precision | 0.6034 | 0.6034 | ✅ |
| Loss | 1.2691 | 1.2691 | ✅ |

> **Conclusion**: Metrics are identical — the HuggingFace Hub model is a verified exact copy of the locally trained model.

### Per-Genre Performance

| Genre | Precision | Recall | F1 |
|-------|:---------:|:------:|:--:|
| Poetry | 0.78 | 0.81 | **0.79** |
| Comics & Graphic | 0.76 | 0.77 | **0.77** |
| Children | 0.62 | 0.66 | 0.64 |
| Mystery/Thriller/Crime | 0.60 | 0.62 | 0.61 |
| Romance | 0.62 | 0.61 | 0.61 |
| History & Biography | 0.59 | 0.60 | 0.60 |
| Fantasy & Paranormal | 0.43 | 0.43 | 0.43 |
| Young Adult | 0.43 | 0.36 | 0.39 |

**Best performing**: Poetry (F1=0.79) and Comics/Graphic (F1=0.77) — these genres have distinctive vocabulary.  
**Worst performing**: Young Adult (F1=0.39) and Fantasy/Paranormal (F1=0.43) — these genres overlap significantly with each other and with Romance.

### vs. Baseline

| Model | Accuracy |
|-------|:--------:|
| TF-IDF + Logistic Regression | ~0.55 |
| **DistilBERT (fine-tuned)** | **0.61** |

Fine-tuned DistilBERT outperforms the TF-IDF baseline by ~6 percentage points.

---

## 4. Docker Images

Two Docker images were created:

1. **`Dockerfile`** (Training) — Builds an image that trains the model from scratch.
   ```bash
   docker build -t ml-assignment-train .
   docker run --rm ml-assignment-train
   ```

2. **`Dockerfile.eval`** (Evaluation Only) — Pulls the model from HuggingFace Hub and runs evaluation automatically.
   ```bash
   docker build -f Dockerfile.eval -t ml-assignment-eval .
   docker run --rm ml-assignment-eval
   ```

Both images use `python:3.10-slim` as the base image with minimal system dependencies.

---

## 5. Challenges Faced

1. **GPU Memory Constraints** — The NVIDIA GPU had only 3.68 GiB VRAM, causing OOM with the original batch size of 10. Solved by using `batch_size=2` with `gradient_accumulation_steps=5` and FP16 mixed precision to maintain the same effective batch size.

2. **CPU vs GPU Speed** — CPU training would have taken ~4 hours. GPU with optimized settings completed in ~18 minutes — a 13x speedup.

3. **External Data Dependency** — The Goodreads dataset is hosted externally and streamed via HTTP, requiring stable network access during data preparation. A caching mechanism (`genre_reviews_dict.pickle`) was implemented to avoid re-downloading.

4. **Genre Overlap** — Some genres (Young Adult, Fantasy/Paranormal, Romance) share significant vocabulary overlap, making classification harder. This is reflected in the lower F1 scores for these categories.

---

## 6. Project Structure

```
assignment_03/
├── src/
│   ├── __init__.py        # Package init
│   ├── data.py            # Data loading & preprocessing
│   ├── utils.py           # Dataset class, metrics, label maps
│   ├── train.py           # Training pipeline (Trainer API)
│   └── eval.py            # Evaluation script
├── results/
│   ├── train_metrics.json
│   ├── eval_results_local.json
│   ├── eval_results_hf.json
│   └── eval_comparison.json
├── Dockerfile             # Training image
├── Dockerfile.eval        # Evaluation-only image
├── requirements.txt
├── README.md
└── REPORT.md              # This report
```
