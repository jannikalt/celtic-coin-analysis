"""Linear classifier on top of DINOv3 features."""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import DinoV3Backbone


class DinoV3Classifier(nn.Module):
    """Classifier model supporting single or multi-view fusion."""
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        views: str = "rev",
        head_type: str = "linear",
        head_hidden: int = 512,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.views = views
        self.backbone = DinoV3Backbone(model_name)

        head_in = self.backbone.hidden_size * 2 if views == "both_concat" else self.backbone.hidden_size
        if head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(head_in, head_hidden),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden, num_classes),
            )
        elif head_type == "linear":
            self.head = nn.Linear(head_in, num_classes)
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

        for p in self.backbone.parameters():
            p.requires_grad = False

    def _encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images into DINOv3 CLS embeddings."""
        return self.backbone(pixel_values)

    def forward(self, pixel_values_rev=None, pixel_values_obv=None):
        """Forward pass returning class logits."""
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

        logits = self.head(z)
        return logits

    def extract_features(self, pixel_values_rev=None, pixel_values_obv=None) -> torch.Tensor:
        """Extract fused feature vectors without classification head."""
        if self.views == "rev":
            return self._encode(pixel_values_rev)
        if self.views == "obv":
            return self._encode(pixel_values_obv)
        if self.views == "both_avg":
            z_rev = self._encode(pixel_values_rev)
            z_obv = self._encode(pixel_values_obv)
            return 0.5 * (z_rev + z_obv)
        if self.views == "both_concat":
            z_rev = self._encode(pixel_values_rev)
            z_obv = self._encode(pixel_values_obv)
            return torch.cat([z_obv, z_rev], dim=1)
        raise ValueError(f"Unknown views mode: {self.views}")
