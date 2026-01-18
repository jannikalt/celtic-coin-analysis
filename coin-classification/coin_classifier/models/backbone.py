"""DINOv3 backbone wrapper."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class DinoV3Backbone(nn.Module):
    """Wrapper around HuggingFace DINOv3 to extract CLS embeddings."""
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_size = int(self.model.config.hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return CLS token embeddings for a batch of images."""
        outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state[:, 0, :]
