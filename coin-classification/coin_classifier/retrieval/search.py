"""Embedding-based similarity search utilities."""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

import numpy as np


@dataclass
class SearchResult:
    """Single similarity search result entry."""
    image_path: Path
    similarity: float
    metadata: Optional[Dict] = None


class EmbeddingSearch:
    """Cosine-similarity search over embedding collections."""
    def __init__(self, embeddings: np.ndarray, image_paths: List[Path], metadata: Optional[List[Dict]] = None):
        if len(embeddings) != len(image_paths):
            raise ValueError("Embeddings count must match image_paths")
        self.embeddings = embeddings
        self.image_paths = image_paths
        self.metadata = metadata or [{}] * len(image_paths)

        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings_norm = self.embeddings / np.where(norms == 0, 1, norms)

    def search(self, query_embedding: np.ndarray, top_k: int = 10, threshold: float = 0.0) -> List[SearchResult]:
        """Return top-k most similar results above threshold."""
        qn = np.linalg.norm(query_embedding)
        if qn > 0:
            query_embedding = query_embedding / qn
        similarities = np.dot(self.embeddings_norm, query_embedding)
        valid = similarities >= threshold
        idx = np.where(valid)[0]
        if len(idx) == 0:
            return []
        sorted_idx = idx[np.argsort(similarities[idx])[::-1][:top_k]]

        results = []
        for i in sorted_idx:
            results.append(SearchResult(
                image_path=self.image_paths[i],
                similarity=float(similarities[i]),
                metadata=self.metadata[i] if i < len(self.metadata) else {},
            ))
        return results
