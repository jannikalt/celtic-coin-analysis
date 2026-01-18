"""Dataset split utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def stratified_split(df: pd.DataFrame, val_ratio: float = 0.2, seed: int = 0):
    """Create a stratified train/val split by label."""
    rng = np.random.RandomState(seed)
    train_idx = []
    val_idx = []
    for lab in sorted(df["label"].unique()):
        ids = df[df["label"] == lab].index.values
        rng.shuffle(ids)
        n_val = max(1, int(val_ratio * len(ids)))
        val_idx.extend(ids[:n_val].tolist())
        train_idx.extend(ids[n_val:].tolist())

    df_train = df.loc[train_idx].reset_index(drop=True)
    df_val = df.loc[val_idx].reset_index(drop=True)
    return df_train, df_val
