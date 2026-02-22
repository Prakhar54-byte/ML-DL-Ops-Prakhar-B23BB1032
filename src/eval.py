"""
Evaluation script for DistilBERT Goodreads genre classifier.

Usage:
    python -m src.eval --model_path ./distilbert-goodreads-genre-classifier
    python -m src.eval --model_path Prakhar54-byte/distilbert-goodreads-genre-classifier --source hf

This script:
  1. Loads a trained model from a local path or HuggingFace repo
  2. Loads test data
  3. Runs evaluation and prints classification report
  4. Saves results to JSON
"""

import argparse
import json
import os

import torch
from sklearn.metrics import classification_report
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from src.data import get_encoded_labels, load_all_reviews, prepare_splits
from src.utils import ID2LABEL, LABEL2ID, GenreDataset, compute_metrics

os.environ["WANDB_DISABLED"] = "true"

RESULTS_DIR = "./results"
MAX_LENGTH = 512


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DistilBERT genre classifier")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Local model directory or HuggingFace repo ID",
    )
    parser.add_argument(
        "--source",
        type=str,
        choices=["local", "hf"],
        default="local",
        help="Source of the model: 'local' or 'hf' (HuggingFace Hub)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Load data ---
    print("\n=== Loading test data ===")
    genre_reviews = load_all_reviews(seed=args.seed)
    _, _, test_texts, test_labels = prepare_splits(genre_reviews, seed=args.seed)
    print(f"Test samples: {len(test_texts)}")

    # --- Load model & tokenizer ---
    print(f"\n=== Loading model from: {args.model_path} (source: {args.source}) ===")
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_path)
    model = DistilBertForSequenceClassification.from_pretrained(args.model_path).to(device)

    # --- Tokenize test data ---
    print("\n=== Tokenizing test data ===")
    test_encodings = tokenizer(
        test_texts, truncation=True, padding=True, max_length=MAX_LENGTH
    )
    test_labels_enc = get_encoded_labels(test_labels)
    test_dataset = GenreDataset(test_encodings, test_labels_enc)

    # --- Evaluate ---
    print("\n=== Running evaluation ===")
    eval_args = TrainingArguments(
        output_dir="./eval_tmp",
        per_device_eval_batch_size=16,
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        compute_metrics=compute_metrics,
    )

    eval_results = trainer.evaluate(eval_dataset=test_dataset)
    print(f"\nEvaluation results: {json.dumps(eval_results, indent=2)}")

    # --- Classification report ---
    print("\n=== Detailed Classification Report ===")
    predicted = trainer.predict(test_dataset)
    predicted_labels = predicted.predictions.argmax(-1).flatten().tolist()
    predicted_labels_str = [ID2LABEL[lid] for lid in predicted_labels]

    report = classification_report(test_labels, predicted_labels_str)
    print(report)

    # --- Save results ---
    os.makedirs(RESULTS_DIR, exist_ok=True)

    suffix = "hf" if args.source == "hf" else "local"
    result_file = os.path.join(RESULTS_DIR, f"eval_results_{suffix}.json")

    save_data = {
        "model_path": args.model_path,
        "source": args.source,
        "eval_loss": eval_results.get("eval_loss"),
        "eval_accuracy": eval_results.get("eval_accuracy"),
        "eval_precision": eval_results.get("eval_precision"),
        "eval_recall": eval_results.get("eval_recall"),
        "eval_f1": eval_results.get("eval_f1"),
        "classification_report": report,
    }

    with open(result_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {result_file}")

    # --- Compare if both local and HF results exist ---
    local_file = os.path.join(RESULTS_DIR, "eval_results_local.json")
    hf_file = os.path.join(RESULTS_DIR, "eval_results_hf.json")

    if os.path.exists(local_file) and os.path.exists(hf_file):
        print("\n=== Comparison: Local vs HuggingFace Model ===")
        with open(local_file, "r") as f:
            local_data = json.load(f)
        with open(hf_file, "r") as f:
            hf_data = json.load(f)

        comparison = {
            "local_model": {
                "accuracy": local_data.get("eval_accuracy"),
                "f1": local_data.get("eval_f1"),
                "loss": local_data.get("eval_loss"),
            },
            "hf_model": {
                "accuracy": hf_data.get("eval_accuracy"),
                "f1": hf_data.get("eval_f1"),
                "loss": hf_data.get("eval_loss"),
            },
        }

        print(f"{'Metric':<12} {'Local':>10} {'HF':>10}")
        print("-" * 34)
        for metric in ["accuracy", "f1", "loss"]:
            local_val = comparison["local_model"].get(metric, "N/A")
            hf_val = comparison["hf_model"].get(metric, "N/A")
            if isinstance(local_val, float):
                local_val = f"{local_val:.4f}"
            if isinstance(hf_val, float):
                hf_val = f"{hf_val:.4f}"
            print(f"{metric:<12} {str(local_val):>10} {str(hf_val):>10}")

        comp_file = os.path.join(RESULTS_DIR, "eval_comparison.json")
        with open(comp_file, "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison saved to {comp_file}")


if __name__ == "__main__":
    main()
