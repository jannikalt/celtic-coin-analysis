"""Training utilities for side classification mode."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def train_epoch_side_classifier(model, loader, optimizer, criterion, device: str) -> Dict:
    """Train side classifier for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(loader, desc="Train", leave=False):
        labels = batch["labels"].to(device)
        pixel_values_a = batch["pixel_values_a"].to(device)
        pixel_values_b = batch["pixel_values_b"].to(device)
        masks_a = batch["masks_a"].to(device)
        masks_b = batch["masks_b"].to(device)
        
        logits = model(
            pixel_values_a=pixel_values_a,
            masks_a=masks_a,
            pixel_values_b=pixel_values_b,
            masks_b=masks_b
        )
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
def eval_side_classifier(model, loader, criterion, device: str) -> Dict:
    """Evaluate side classifier on validation set."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_ids = []
    all_flips = []
    all_probs = []
    
    for batch in tqdm(loader, desc="Val", leave=False):
        labels = batch["labels"].to(device)
        ids = batch["ids"]
        flips = batch["flips"].detach().cpu().numpy().tolist()
        
        pixel_values_a = batch["pixel_values_a"].to(device)
        pixel_values_b = batch["pixel_values_b"].to(device)
        masks_a = batch["masks_a"].to(device)
        masks_b = batch["masks_b"].to(device)
        
        logits = model(
            pixel_values_a=pixel_values_a,
            masks_a=masks_a,
            pixel_values_b=pixel_values_b,
            masks_b=masks_b
        )
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        
        total_loss += float(loss.item())
        preds = logits.argmax(dim=1).detach().cpu().numpy().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_ids.extend(ids)
        all_flips.extend(flips)
        all_probs.append(probs)
    
    avg_loss = total_loss / max(1, len(loader))
    return {
        "loss": avg_loss,
        "y_true": np.array(all_labels),
        "y_pred": np.array(all_preds),
        "ids": np.array(all_ids),
        "flips": np.array(all_flips),
        "probs": np.vstack(all_probs),
    }
