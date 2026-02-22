"""
Utility functions for the DistilBERT genre classification project.
Includes dataset class, metrics computation, and label mappings.
"""

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# Genre labels used in the Goodreads dataset
GENRES = [
    "poetry",
    "children",
    "comics_graphic",
    "fantasy_paranormal",
    "history_biography",
    "mystery_thriller_crime",
    "romance",
    "young_adult",
]

# Deterministic label-to-id mapping (sorted for reproducibility)
LABEL2ID = {label: idx for idx, label in enumerate(sorted(GENRES))}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


class GenreDataset(torch.utils.data.Dataset):
    """PyTorch Dataset wrapping tokenized encodings and integer labels."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def compute_metrics(pred):
    """Compute accuracy, precision, recall, and F1 for Trainer evaluation."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
