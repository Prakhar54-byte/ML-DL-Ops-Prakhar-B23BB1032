"""
Data loading and preprocessing for Goodreads genre classification.
Downloads reviews from the UCSD Book Graph dataset, samples per genre,
and splits into train/test sets.
"""

import gzip
import json
import os
import pickle
import random

import requests

from src.utils import GENRES, LABEL2ID

# URLs for each genre's review data
GENRE_URLS = {
    "poetry": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz",
    "children": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz",
    "comics_graphic": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz",
    "fantasy_paranormal": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz",
    "history_biography": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz",
    "mystery_thriller_crime": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz",
    "romance": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz",
    "young_adult": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz",
}


def load_reviews(url, head=10000, sample_size=2000):
    """Stream reviews from URL and return a random sample."""
    reviews = []
    count = 0

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with gzip.open(response.raw, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            reviews.append(d["review_text"])
            count += 1
            if head is not None and count >= head:
                break

    return random.sample(reviews, min(sample_size, len(reviews)))


def load_all_reviews(cache_path="genre_reviews_dict.pickle", seed=42):
    """Load reviews for all genres, using cache if available."""
    random.seed(seed)

    if os.path.exists(cache_path):
        print(f"Loading cached reviews from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    genre_reviews = {}
    for genre, url in GENRE_URLS.items():
        print(f"Downloading reviews for genre: {genre}")
        genre_reviews[genre] = load_reviews(url, head=10000, sample_size=2000)

    with open(cache_path, "wb") as f:
        pickle.dump(genre_reviews, f)
    print(f"Cached reviews to {cache_path}")

    return genre_reviews


def prepare_splits(
    genre_reviews,
    samples_per_genre=1000,
    train_per_genre=800,
    seed=42,
):
    """Split genre reviews into train and test sets.
    
    Returns:
        train_texts, train_labels, test_texts, test_labels
    """
    random.seed(seed)

    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    for genre, reviews in genre_reviews.items():
        sampled = random.sample(reviews, min(samples_per_genre, len(reviews)))
        for review in sampled[:train_per_genre]:
            train_texts.append(review)
            train_labels.append(genre)
        for review in sampled[train_per_genre:]:
            test_texts.append(review)
            test_labels.append(genre)

    return train_texts, train_labels, test_texts, test_labels


def get_encoded_labels(labels):
    """Convert string labels to integer IDs."""
    return [LABEL2ID[label] for label in labels]


if __name__ == "__main__":
    # Quick test
    genre_reviews = load_all_reviews()
    train_texts, train_labels, test_texts, test_labels = prepare_splits(genre_reviews)
    print(f"Train: {len(train_texts)} samples, Test: {len(test_texts)} samples")
    print(f"Genres: {sorted(set(train_labels))}")
