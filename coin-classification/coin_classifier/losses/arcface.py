"""ArcFace margin-based classification head for metric learning."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFace(nn.Module):
    """ArcFace layer producing margin-adjusted logits."""
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Compute ArcFace logits for given embeddings and labels."""
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        phi = torch.cos(theta + self.m)
        phi = torch.where(theta > math.pi - self.m, cosine - self.m * math.sin(math.pi - self.m), phi)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output
