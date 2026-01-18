"""Metric embedding model for DINOv3 features."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import DinoV3Backbone


class MetricEmbeddingModel(nn.Module):
    """Projection head for metric learning embeddings."""
    def __init__(
        self,
        model_name: str,
        embedding_dim: int = 512,
        views: str = "rev",
        proj_type: str = "linear",
        proj_hidden: int = 512,
        proj_dropout: float = 0.1,
    ):
        super().__init__()
        self.views = views
        self.backbone = DinoV3Backbone(model_name)

        in_features = self.backbone.hidden_size * 2 if views == "both_concat" else self.backbone.hidden_size
        if proj_type == "mlp":
            self.proj = nn.Sequential(
                nn.Linear(in_features, proj_hidden),
                nn.GELU(),
                nn.Dropout(proj_dropout),
                nn.Linear(proj_hidden, embedding_dim),
            )
        elif proj_type == "linear":
            self.proj = nn.Linear(in_features, embedding_dim)
        else:
            raise ValueError(f"Unknown proj_type: {proj_type}")

        for p in self.backbone.parameters():
            p.requires_grad = False

    def _encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images into DINOv3 CLS embeddings."""
        return self.backbone(pixel_values)

    def forward(self, pixel_values_rev=None, pixel_values_obv=None) -> torch.Tensor:
        """Return L2-normalized embeddings for metric learning."""
        if self.views == "rev":
            z = self._encode(pixel_values_rev)
        elif self.views == "obv":
            z = self._encode(pixel_values_obv)
        elif self.views == "both_avg":
            z_rev = self._encode(pixel_values_rev)
            z_obv = self._encode(pixel_values_obv)
            z = 0.5 * (z_rev + z_obv)
        elif self.views == "both_concat":
            z_rev = self._encode(pixel_values_rev)
            z_obv = self._encode(pixel_values_obv)
            z = torch.cat([z_obv, z_rev], dim=1)
        else:
            raise ValueError(f"Unknown views mode: {self.views}")

        emb = self.proj(z)
        emb = F.normalize(emb, dim=1)
        return emb
