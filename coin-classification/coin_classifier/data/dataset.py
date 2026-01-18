"""Dataset utilities and collator for DINOv3 coin classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


REQUIRED_COLUMNS = {"id", "label", "obverse_path", "reverse_path"}


def load_dataframe(path: str) -> pd.DataFrame:
    """Load TSV/CSV with required columns for coin classification."""
    df = pd.read_csv(path, sep="\t")
    if not REQUIRED_COLUMNS.issubset(set(df.columns)):
        df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return df


@dataclass
class DatasetConfig:
    """Configuration for dataset views selection."""
    views: str = "rev"  # rev, obv, both_avg, both_concat


class CoinImagesDataset(Dataset):
    """Dataset returning obverse/reverse images and labels."""
    def __init__(
        self,
        df: pd.DataFrame,
        label_encoder,
        input_size: int = 224,
        views: str = "rev",
    ):
        self.df = df.reset_index(drop=True)
        self.label_encoder = label_encoder
        self.input_size = input_size
        self.views = views

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path: str) -> Image.Image:
        """Load a single image as RGB."""
        img = Image.open(path).convert("RGB")
        if self.input_size:
            img = img.resize((self.input_size, self.input_size), resample=Image.BILINEAR)
        return img

    def __getitem__(self, idx: int) -> Dict:
        """Return a sample dict with images, label, and id."""
        row = self.df.iloc[idx]
        label = int(self.label_encoder.transform([row["label"]])[0])

        sample = {
            "label": label,
            "id": row["id"],
        }

        if self.views in {"obv", "both_concat", "both_avg"}:
            sample["obv_image"] = self._load_image(row["obverse_path"])
        if self.views in {"rev", "both_concat", "both_avg"}:
            sample["rev_image"] = self._load_image(row["reverse_path"])

        return sample


class DinoV3Collator:
    """Collator converting PIL images to DINOv3 tensor batches."""
    def __init__(self, processor, views: str = "rev"):
        self.processor = processor
        self.views = views

    def __call__(self, batch):
        import torch
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        ids = [b["id"] for b in batch]

        out = {
            "labels": labels,
            "ids": ids,
        }

        if self.views in {"obv", "both_concat", "both_avg"}:
            obv_images = [b["obv_image"] for b in batch]
            obv_inputs = self.processor(images=obv_images, return_tensors="pt")
            out["pixel_values_obv"] = obv_inputs["pixel_values"]

        if self.views in {"rev", "both_concat", "both_avg"}:
            rev_images = [b["rev_image"] for b in batch]
            rev_inputs = self.processor(images=rev_images, return_tensors="pt")
            out["pixel_values_rev"] = rev_inputs["pixel_values"]

        return out
