"""HDBSCAN evaluation on learned embeddings."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import silhouette_score


def hdbscan_evaluate(
    embeddings: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int = 5,
) -> Dict:
    """Cluster embeddings with HDBSCAN and return cluster statistics."""
    try:
        import hdbscan  # type: ignore
    except Exception as exc:
        raise RuntimeError("HDBSCAN is not installed. Install with: pip install hdbscan") from exc

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(embeddings)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    noise_pct = float(n_noise / len(labels)) if len(labels) else 0.0

    metrics: Dict = {
        "n_clusters": int(n_clusters),
        "n_noise": n_noise,
        "noise_pct": noise_pct,
    }

    valid_mask = labels != -1
    if np.sum(valid_mask) > 1 and n_clusters > 1:
        metrics["silhouette"] = float(silhouette_score(embeddings[valid_mask], labels[valid_mask]))

    return {"metrics": metrics, "labels": labels}
