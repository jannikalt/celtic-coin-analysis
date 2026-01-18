"""Evaluation metrics for classification outputs."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Dict:
    """Compute accuracy, weighted scores, per-class metrics, and confusion matrix."""
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)

    per_class = {}
    for idx, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(per_class_precision[idx]) if idx < len(per_class_precision) else 0.0,
            "recall": float(per_class_recall[idx]) if idx < len(per_class_recall) else 0.0,
            "support": int(np.sum(y_true == idx)),
        }

    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": float(acc),
        "f1_weighted": float(f1),
        "precision_weighted": float(precision),
        "recall_weighted": float(recall),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }
