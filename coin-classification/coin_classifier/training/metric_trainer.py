"""Training utilities for metric learning mode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from tqdm import tqdm

from ..losses.triplet import BatchHardTripletLoss


@dataclass
class MetricConfig:
    """Configuration for metric-learning losses and optimizer settings."""
    device: str = "cuda"
    epochs: int = 10
    lr: float = 1e-4
    arcface_s: float = 30.0
    arcface_m: float = 0.5
    triplet_margin: float = 0.3
    triplet_weight: float = 1.0
    arcface_weight: float = 1.0


def resolve_device(device: str) -> str:
    """Resolve requested device, falling back to CPU if CUDA is unavailable."""
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def train_metric_epoch(model, loader, optimizer, arcface, cfg: MetricConfig) -> Dict:
    """Train metric model for one epoch with ArcFace + triplet loss."""
    model.train()
    total_loss = 0.0

    triplet = BatchHardTripletLoss(margin=cfg.triplet_margin).to(cfg.device)

    for batch in tqdm(loader, desc="Train", leave=False):
        labels = batch["labels"].to(cfg.device)
        pixel_values_rev = batch.get("pixel_values_rev")
        pixel_values_obv = batch.get("pixel_values_obv")

        if pixel_values_rev is not None:
            pixel_values_rev = pixel_values_rev.to(cfg.device)
        if pixel_values_obv is not None:
            pixel_values_obv = pixel_values_obv.to(cfg.device)

        embeddings = model(pixel_values_rev=pixel_values_rev, pixel_values_obv=pixel_values_obv)

        logits = arcface(embeddings, labels)
        loss_arc = torch.nn.functional.cross_entropy(logits, labels)
        loss_tri = triplet(embeddings, labels)
        loss = cfg.arcface_weight * loss_arc + cfg.triplet_weight * loss_tri

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    avg_loss = total_loss / max(1, len(loader))
    return {"loss": avg_loss}


@torch.no_grad()
def eval_metric_embeddings(model, loader, device: str) -> Dict:
    """Extract embeddings and labels for evaluation."""
    model.eval()
    feats = []
    labels = []
    ids = []

    for batch in tqdm(loader, desc="Val", leave=False):
        y = batch["labels"].to(device)
        pixel_values_rev = batch.get("pixel_values_rev")
        pixel_values_obv = batch.get("pixel_values_obv")

        if pixel_values_rev is not None:
            pixel_values_rev = pixel_values_rev.to(device)
        if pixel_values_obv is not None:
            pixel_values_obv = pixel_values_obv.to(device)

        emb = model(pixel_values_rev=pixel_values_rev, pixel_values_obv=pixel_values_obv)
        feats.append(emb.detach().cpu().numpy())
        labels.append(y.detach().cpu().numpy())
        ids.extend(batch["ids"])

    feats = np.vstack(feats)
    labels = np.concatenate(labels)
    return {"embeddings": feats, "labels": labels, "ids": np.array(ids)}
