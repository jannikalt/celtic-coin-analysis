"""Training utilities for classifier mode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ..eval.metrics import classification_metrics


@dataclass
class TrainConfig:
    """Basic training configuration for classifier mode."""
    device: str = "cuda"
    epochs: int = 10
    lr: float = 1e-4


def resolve_device(device: str) -> str:
    """Resolve requested device, falling back to CPU if CUDA is unavailable."""
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


def train_epoch_classifier(model, loader, optimizer, criterion, device: str) -> Dict:
    """Train classifier for one epoch and return loss and predictions."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in tqdm(loader, desc="Train", leave=False):
        labels = batch["labels"].to(device)
        pixel_values_rev = batch.get("pixel_values_rev")
        pixel_values_obv = batch.get("pixel_values_obv")

        if pixel_values_rev is not None:
            pixel_values_rev = pixel_values_rev.to(device)
        if pixel_values_obv is not None:
            pixel_values_obv = pixel_values_obv.to(device)

        logits = model(pixel_values_rev=pixel_values_rev, pixel_values_obv=pixel_values_obv)
        loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        preds = logits.argmax(dim=1).detach().cpu().numpy().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(loader))
    return {
        "loss": avg_loss,
        "y_true": np.array(all_labels),
        "y_pred": np.array(all_preds),
    }


@torch.no_grad()
def eval_classifier(model, loader, criterion, device: str) -> Dict:
    """Evaluate classifier on a validation set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_ids = []
    all_probs = []

    for batch in tqdm(loader, desc="Val", leave=False):
        labels = batch["labels"].to(device)
        ids = batch["ids"]
        pixel_values_rev = batch.get("pixel_values_rev")
        pixel_values_obv = batch.get("pixel_values_obv")

        if pixel_values_rev is not None:
            pixel_values_rev = pixel_values_rev.to(device)
        if pixel_values_obv is not None:
            pixel_values_obv = pixel_values_obv.to(device)

        logits = model(pixel_values_rev=pixel_values_rev, pixel_values_obv=pixel_values_obv)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()

        total_loss += float(loss.item())
        preds = logits.argmax(dim=1).detach().cpu().numpy().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_ids.extend(ids)
        all_probs.append(probs)

    avg_loss = total_loss / max(1, len(loader))
    return {
        "loss": avg_loss,
        "y_true": np.array(all_labels),
        "y_pred": np.array(all_preds),
        "ids": np.array(all_ids),
        "probs": np.vstack(all_probs),
    }


@torch.no_grad()
def extract_embeddings(model, loader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    """Extract feature embeddings and labels from a dataloader."""
    model.eval()
    all_feats = []
    all_labels = []

    for batch in tqdm(loader, desc="Embed", leave=False):
        labels = batch["labels"].to(device)
        pixel_values_rev = batch.get("pixel_values_rev")
        pixel_values_obv = batch.get("pixel_values_obv")

        if pixel_values_rev is not None:
            pixel_values_rev = pixel_values_rev.to(device)
        if pixel_values_obv is not None:
            pixel_values_obv = pixel_values_obv.to(device)

        feats = model.extract_features(pixel_values_rev=pixel_values_rev, pixel_values_obv=pixel_values_obv)
        all_feats.append(feats.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    return np.vstack(all_feats), np.concatenate(all_labels)
