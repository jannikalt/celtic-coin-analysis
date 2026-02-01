"""Plotting utilities for evaluation artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import List
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image


def save_confusion_matrix(cm: np.ndarray, class_names: List[str], out_path: Path) -> None:
    """Save a confusion matrix heatmap to disk."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_high_confidence_mistakes(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    ids: np.ndarray,
    class_names: List[str],
    out_path: Path,
    views: str = "both_concat",
    top_k: int = 12,
    confidence_threshold: float = 0.7,
) -> None:
    """Visualize top-k misclassified examples with high confidence.
    
    Args:
        df: Validation dataframe with image paths
        y_true: Ground truth labels
        y_pred: Predicted labels
        probs: Prediction probabilities (n_samples, n_classes)
        ids: Sample IDs matching df
        class_names: List of class names
        out_path: Output path for the visualization
        views: View configuration ('rev', 'obv', 'both_concat', 'both_avg')
        top_k: Number of examples to visualize
        confidence_threshold: Minimum confidence for misclassified examples
    """
    # Find misclassified samples
    misclassified_mask = y_true != y_pred
    misclassified_indices = np.where(misclassified_mask)[0]
    
    if len(misclassified_indices) == 0:
        print("No misclassified examples found.")
        return
    
    # Get predicted probabilities for misclassified samples
    misclass_probs = probs[misclassified_indices]
    misclass_confidence = np.max(misclass_probs, axis=1)
    
    # Filter by confidence threshold
    high_conf_mask = misclass_confidence >= confidence_threshold
    high_conf_indices = misclassified_indices[high_conf_mask]
    high_conf_scores = misclass_confidence[high_conf_mask]
    
    if len(high_conf_indices) == 0:
        print(f"No high-confidence (>={confidence_threshold}) misclassifications found.")
        return
    
    # Sort by confidence (descending) and take top_k
    sorted_idx = np.argsort(-high_conf_scores)
    top_indices = high_conf_indices[sorted_idx[:top_k]]
    
    # Create visualization
    n_examples = min(top_k, len(top_indices))
    n_cols = 3
    n_rows = (n_examples + n_cols - 1) // n_cols
    
    # Determine image layout based on views
    if views in {"both_concat", "both_avg"}:
        fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(6 * n_cols, 4 * n_rows))
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1 or n_cols == 1:
        axes = axes.reshape(n_rows, -1)
    
    # Create ID to dataframe row mapping
    df_indexed = df.set_index("id")
    
    for plot_idx, val_idx in enumerate(top_indices):
        row_idx = plot_idx // n_cols
        col_idx = plot_idx % n_cols
        
        sample_id = ids[val_idx]
        true_label_idx = y_true[val_idx]
        pred_label_idx = y_pred[val_idx]
        confidence = np.max(probs[val_idx])
        
        true_label = class_names[true_label_idx]
        pred_label = class_names[pred_label_idx]
        
        # Get image paths from dataframe
        if sample_id not in df_indexed.index:
            continue
            
        row = df_indexed.loc[sample_id]
        
        if views in {"both_concat", "both_avg"}:
            # Show both obverse and reverse
            ax_obv = axes[row_idx, col_idx * 2] if n_cols > 1 else axes[row_idx, 0]
            ax_rev = axes[row_idx, col_idx * 2 + 1] if n_cols > 1 else axes[row_idx, 1]
            
            try:
                obv_img = Image.open(row["obverse_path"]).convert("RGB")
                ax_obv.imshow(obv_img)
                ax_obv.axis("off")
                ax_obv.set_title("Obverse", fontsize=10)
            except Exception:
                ax_obv.text(0.5, 0.5, "Image not found", ha="center", va="center")
                ax_obv.axis("off")
            
            try:
                rev_img = Image.open(row["reverse_path"]).convert("RGB")
                ax_rev.imshow(rev_img)
                ax_rev.axis("off")
                ax_rev.set_title("Reverse", fontsize=10)
            except Exception:
                ax_rev.text(0.5, 0.5, "Image not found", ha="center", va="center")
                ax_rev.axis("off")
            
            # Add text below the pair
            fig.text(
                (col_idx * 2 + 1) / (n_cols * 2),
                1 - (row_idx + 0.95) / n_rows,
                f"ID: {sample_id}\nTrue: {true_label}\nPred: {pred_label}\nConf: {confidence:.2%}",
                ha="center",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3),
            )
        else:
            # Show single view
            ax = axes[row_idx, col_idx] if n_rows > 1 or n_cols > 1 else axes
            
            img_path = row["reverse_path"] if views == "rev" else row["obverse_path"]
            
            try:
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, "Image not found", ha="center", va="center")
            
            ax.axis("off")
            ax.set_title(
                f"ID: {sample_id}\nTrue: {true_label}\nPred: {pred_label}\nConf: {confidence:.2%}",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3),
            )
    
    # Hide unused subplots
    total_subplots = n_rows * n_cols * (2 if views in {"both_concat", "both_avg"} else 1)
    for idx in range(n_examples * (2 if views in {"both_concat", "both_avg"} else 1), total_subplots):
        if views in {"both_concat", "both_avg"}:
            row_idx = idx // (n_cols * 2)
            col_idx = idx % (n_cols * 2)
        else:
            row_idx = idx // n_cols
            col_idx = idx % n_cols
        
        if n_rows > 1 or n_cols > 1:
            axes[row_idx, col_idx].axis("off")
    
    plt.suptitle(
        f"High-Confidence Misclassifications (threshold: {confidence_threshold:.0%})",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"Saved {n_examples} high-confidence misclassifications to {out_path}")
