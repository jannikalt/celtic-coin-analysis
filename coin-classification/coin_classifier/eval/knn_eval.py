"""kNN evaluation on learned embeddings."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


def knn_evaluate(embeddings: np.ndarray, labels: np.ndarray, test_size: float = 0.2, n_neighbors: int = 5) -> Dict:
    """Run kNN evaluation and return weighted metrics."""
    unique, counts = np.unique(labels, return_counts=True)
    keep = set(unique[counts >= 2])
    mask = np.array([y in keep for y in labels])
    embeddings = embeddings[mask]
    labels = labels[mask]

    if len(np.unique(labels)) < 2 or len(labels) < 4:
        return {
            "accuracy": None,
            "precision_weighted": None,
            "recall_weighted": None,
            "f1_weighted": None,
            "warning": "Not enough samples per class for stratified split",
        }

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=test_size, random_state=42, stratify=labels
    )
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric="cosine")
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }
