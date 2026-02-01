"""Dataset and collator for side classification training."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch


LABEL_MAPPING = {"obv-rev": 0, "rev-obv": 1}


def find_mask_path(
    coin_id: str,
    image_path: str,
    mask_dir: Optional[str],
    mask_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
) -> Path:
    """Find mask file for a given coin ID and image path."""
    base_dir = Path(mask_dir) if mask_dir else Path(image_path).parent
    for ext in mask_exts:
        candidate = base_dir / f"{coin_id}_mask{ext}"
        if candidate.exists():
            return candidate
    expected = ", ".join(str(base_dir / f"{coin_id}_mask{ext}") for ext in mask_exts)
    raise FileNotFoundError(f"Missing mask for id={coin_id}. Tried: {expected}")


def load_and_preprocess_mask(mask_path: Path, input_size: int) -> np.ndarray:
    """Load and preprocess a binary mask."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Missing/invalid mask: {mask_path}")
    mask = cv2.resize(mask, (input_size, input_size), interpolation=cv2.INTER_NEAREST)
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask_bin


def preprocess_image(img: np.ndarray, input_size: int = 224) -> np.ndarray:
    """Resize image to target size."""
    if img is None:
        return None
    img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_AREA)
    return img


class CoinSideDataset(Dataset):
    """
    Dataset for side classification.
    
    Each coin pair yields TWO training samples:
    - (obv, rev) -> label 0 (obv-rev)
    - (rev, obv) -> label 1 (rev-obv)
    """
    
    def __init__(
        self,
        df_pairs: pd.DataFrame,
        input_size: int = 224,
        mask_dir: Optional[str] = None,
    ):
        self.df = df_pairs.reset_index(drop=True)
        self.input_size = int(input_size)
        self.mask_dir = mask_dir
        
        # Check if explicit mask columns exist
        self.has_mask_cols = (
            "obv_mask_path" in self.df.columns and 
            "rev_mask_path" in self.df.columns
        )
    
    def __len__(self) -> int:
        # Each coin pair yields 2 samples
        return 2 * len(self.df)
    
    def _load_image_pil(self, image_path: str) -> Image.Image:
        """Load and preprocess image as PIL Image."""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Missing image: {image_path}")
        img = preprocess_image(img, input_size=self.input_size)
        if img is None:
            raise RuntimeError(f"Preprocess failed for: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)
    
    def _load_mask(
        self,
        coin_id: str,
        image_path: str,
        explicit_mask_path: Optional[str]
    ) -> np.ndarray:
        """Load binary mask for an image."""
        if explicit_mask_path:
            return load_and_preprocess_mask(
                Path(explicit_mask_path),
                input_size=self.input_size
            )
        mask_path = find_mask_path(coin_id, image_path, self.mask_dir)
        return load_and_preprocess_mask(mask_path, input_size=self.input_size)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample.
        
        For each coin pair at row_idx, we generate two samples:
        - idx = 2*row_idx + 0: (obv, rev) -> label 0
        - idx = 2*row_idx + 1: (rev, obv) -> label 1
        """
        row_idx: int = idx // 2
        flip = (idx % 2) == 1
        
        row = self.df.iloc[row_idx]
        coin_id = str(row["id"])
        
        obv_img_path = str(row["obverse_path"])
        rev_img_path = str(row["reverse_path"])
        
        obv_mask_path = str(row["obv_mask_path"]) if self.has_mask_cols else None
        rev_mask_path = str(row["rev_mask_path"]) if self.has_mask_cols else None
        
        # Base order: (obv, rev) -> label 0
        img_a_path, img_b_path = obv_img_path, rev_img_path
        mask_a_path, mask_b_path = obv_mask_path, rev_mask_path
        label = LABEL_MAPPING["obv-rev"]
        
        # Flipped order: (rev, obv) -> label 1
        if flip:
            img_a_path, img_b_path = img_b_path, img_a_path
            mask_a_path, mask_b_path = mask_b_path, mask_a_path
            label = LABEL_MAPPING["rev-obv"]
        
        img_a = self._load_image_pil(img_a_path)
        img_b = self._load_image_pil(img_b_path)
        
        mask_a = self._load_mask(coin_id, img_a_path, mask_a_path)
        mask_b = self._load_mask(coin_id, img_b_path, mask_b_path)
        
        return {
            "id": coin_id,
            "label": int(label),
            "image_a": img_a,
            "image_b": img_b,
            "mask_a": mask_a,
            "mask_b": mask_b,
            "flip": int(flip),
        }


class DinoV3SideCollator:
    """Collator for side classification batches."""
    
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, batch):
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        ids = [b["id"] for b in batch]
        flips = torch.tensor([b["flip"] for b in batch], dtype=torch.long)
        
        images_a = [b["image_a"] for b in batch]
        images_b = [b["image_b"] for b in batch]
        
        inputs_a = self.processor(images=images_a, return_tensors="pt")
        inputs_b = self.processor(images=images_b, return_tensors="pt")
        
        masks_a = torch.from_numpy(
            np.stack([b["mask_a"] for b in batch], axis=0)
        ).float() / 255.0
        masks_b = torch.from_numpy(
            np.stack([b["mask_b"] for b in batch], axis=0)
        ).float() / 255.0
        
        return {
            "labels": labels,
            "ids": ids,
            "flips": flips,
            "pixel_values_a": inputs_a["pixel_values"],
            "pixel_values_b": inputs_b["pixel_values"],
            "masks_a": masks_a,
            "masks_b": masks_b,
        }
