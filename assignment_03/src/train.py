"""
Training script for DistilBERT Goodreads genre classifier.

Usage:
    python -m src.train [--push_to_hub] [--hf_repo_id REPO_ID] [--output_dir DIR]

This script:
  1. Downloads and preprocesses Goodreads review data
  2. Tokenizes with DistilBertTokenizerFast
  3. Fine-tunes DistilBertForSequenceClassification (8 genres)
  4. Saves model locally (and optionally pushes to HuggingFace Hub)
"""

import argparse
import json
import os

import torch
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from src.data import get_encoded_labels, load_all_reviews, prepare_splits
from src.utils import LABEL2ID, ID2LABEL, GenreDataset, compute_metrics

# Disable wandb logging
os.environ["WANDB_DISABLED"] = "true"

# === Defaults ===
MODEL_NAME = "distilbert-base-cased"
MAX_LENGTH = 512
OUTPUT_DIR = "./distilbert-goodreads-genre-classifier"
RESULTS_DIR = "./results"


def parse_args():
    parser = argparse.ArgumentParser(description="Train DistilBERT genre classifier")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help="Directory to save the fine-tuned model",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the trained model to HuggingFace Hub",
    )
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        default="Prakhar54-byte/distilbert-goodreads-genre-classifier",
        help="HuggingFace repo ID for pushing the model",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Training batch size per device")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Evaluation batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--fp16", action="store_true", default=False, help="Use mixed precision training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Step 1: Load data ---
    print("\n=== Step 1: Loading data ===")
    genre_reviews = load_all_reviews(seed=args.seed)
    train_texts, train_labels, test_texts, test_labels = prepare_splits(
        genre_reviews, seed=args.seed
    )
    print(f"Train: {len(train_texts)} | Test: {len(test_texts)}")

    # --- Step 2: Tokenize ---
    print("\n=== Step 2: Tokenizing ===")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    train_encodings = tokenizer(
        train_texts, truncation=True, padding=True, max_length=MAX_LENGTH
    )
    test_encodings = tokenizer(
        test_texts, truncation=True, padding=True, max_length=MAX_LENGTH
    )

    train_labels_enc = get_encoded_labels(train_labels)
    test_labels_enc = get_encoded_labels(test_labels)

    train_dataset = GenreDataset(train_encodings, train_labels_enc)
    test_dataset = GenreDataset(test_encodings, test_labels_enc)

    # --- Step 3: Load model ---
    print("\n=== Step 3: Loading pre-trained model ===")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    # --- Step 4: Training arguments ---
    print("\n=== Step 4: Configuring training ===")
    training_args = TrainingArguments(
        output_dir="./training_results",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="epoch",
        load_best_model_at_end=False,
        fp16=args.fp16 and torch.cuda.is_available(),
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # --- Step 5: Train ---
    print("\n=== Step 5: Training ===")
    train_result = trainer.train()

    # Log training metrics
    train_metrics = train_result.metrics
    print(f"\nTraining metrics: {json.dumps(train_metrics, indent=2)}")

    # --- Step 6: Save model ---
    print(f"\n=== Step 6: Saving model to {args.output_dir} ===")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save training metrics
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "train_metrics.json"), "w") as f:
        json.dump(train_metrics, f, indent=2)
    print("Training metrics saved to results/train_metrics.json")

    # --- Step 7: Push to Hub (optional) ---
    if args.push_to_hub:
        print(f"\n=== Step 7: Pushing to HuggingFace Hub ({args.hf_repo_id}) ===")
        model.push_to_hub(args.hf_repo_id)
        tokenizer.push_to_hub(args.hf_repo_id)
        print(f"Model pushed to: https://huggingface.co/{args.hf_repo_id}")

    print("\n=== Training complete! ===")


if __name__ == "__main__":
    main()
