"""Side classification module for predicting coin side order."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch
import streamlit as st
from transformers import AutoImageProcessor

from coin_classifier.models.side_classifier import DinoV3SideClassifier


def load_side_classifier_model(
    model_dir: str,
    device: str = "cuda",
) -> tuple[DinoV3SideClassifier, AutoImageProcessor, dict]:
    """
    Load a trained side classifier from output directory.
    
    Returns:
        (model, processor, config)
    """
    model_dir = Path(model_dir)
    
    # Load run config
    config_path = model_dir / "run_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"run_config.json not found in {model_dir}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        run_config = json.load(f)
    
    # Load label mapping
    label_mapping_path = model_dir / "label_mapping.json"
    if label_mapping_path.exists():
        with open(label_mapping_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)
            # Handle both {"label_mapping": {...}} and direct {...} formats
            if "label_mapping" in label_data:
                label_mapping = label_data["label_mapping"]
            else:
                label_mapping = label_data
    else:
        label_mapping = {"obv-rev": 0, "rev-obv": 1}
    
    # Load model checkpoint
    ckpt_path = model_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"best_model.pt not found in {model_dir}")
    
    model_name = run_config.get("model_name", "facebook/dinov3-vits16-pretrain-lvd1689m")
    input_size = run_config.get("input_size", 224)
    mask_pooling = not run_config.get("disable_mask_pooling", False)
    
    # Initialize model
    model = DinoV3SideClassifier(
        model_name=model_name,
        input_size=input_size,
        mask_pooling=mask_pooling,
    )
    
    # Load weights
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Load processor
    processor = AutoImageProcessor.from_pretrained(model_name)
    try:
        processor.size = {"height": input_size, "width": input_size}
    except Exception:
        pass
    
    config = {
        "model_name": model_name,
        "input_size": input_size,
        "label_mapping": label_mapping,
        "mask_pooling": mask_pooling,
    }
    
    return model, processor, config


class SideClassifier:
    """Wrapper for side classification inference."""
    
    def __init__(
        self,
        model: DinoV3SideClassifier,
        processor: AutoImageProcessor,
        config: dict,
        device: str = "cuda",
    ):
        self.model = model
        self.processor = processor
        self.device = device
        self.input_size = config["input_size"]
        # Store label mapping as tuple of tuples for hashability
        # Format: ((label_str, idx), (label_str, idx), ...)
        self.label_mapping = tuple(sorted(config["label_mapping"].items()))
        # Create reverse mapping: ((idx, label_str), (idx, label_str), ...)
        self.idx_to_label = tuple(sorted((v, k) for k, v in config["label_mapping"].items()))
    
    def _get_label_from_idx(self, idx: int) -> str:
        """Get label from index."""
        for stored_idx, label in self.idx_to_label:
            if stored_idx == idx:
                return label
        return "unknown"
    
    def _get_label_mapping_dict(self) -> dict:
        """Get label mapping as dict."""
        return dict(self.label_mapping)
    
    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Resize image to model input size."""
        img = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
        return img
    
    def preprocess_mask(self, mask: np.ndarray) -> np.ndarray:
        """Resize and binarize mask."""
        mask = cv2.resize(mask, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
        _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask_bin
    
    def predict_side_order(
        self,
        img_a: np.ndarray,
        img_b: np.ndarray,
        mask_a: np.ndarray,
        mask_b: np.ndarray,
    ) -> dict[str, Any]:
        """
        Predict the order of two coin images.
        
        Args:
            img_a: First image (BGR)
            img_b: Second image (BGR)
            mask_a: Binary mask for first image
            mask_b: Binary mask for second image
        
        Returns:
            Dictionary with:
            - predicted_order: "obv-rev" or "rev-obv"
            - confidence: float between 0 and 1
            - probabilities: dict with probabilities for each class
        """
        # Preprocess images
        img_a_resized = self.preprocess_image(img_a)
        img_b_resized = self.preprocess_image(img_b)
        
        # Preprocess masks
        mask_a_proc = self.preprocess_mask(mask_a)
        mask_b_proc = self.preprocess_mask(mask_b)
        
        # Convert to RGB PIL images
        img_a_rgb = cv2.cvtColor(img_a_resized, cv2.COLOR_BGR2RGB)
        img_b_rgb = cv2.cvtColor(img_b_resized, cv2.COLOR_BGR2RGB)
        
        img_a_pil = Image.fromarray(img_a_rgb)
        img_b_pil = Image.fromarray(img_b_rgb)
        
        # Process with transformers
        inputs_a = self.processor(images=[img_a_pil], return_tensors="pt")
        inputs_b = self.processor(images=[img_b_pil], return_tensors="pt")
        
        pixel_values_a = inputs_a["pixel_values"].to(self.device)
        pixel_values_b = inputs_b["pixel_values"].to(self.device)
        
        # Convert masks to tensors
        masks_a_t = torch.from_numpy(mask_a_proc).float().unsqueeze(0).to(self.device) / 255.0
        masks_b_t = torch.from_numpy(mask_b_proc).float().unsqueeze(0).to(self.device) / 255.0
        
        # Predict
        with torch.no_grad():
            preds, probs = self.model.predict_side_order(
                pixel_values_a, masks_a_t, pixel_values_b, masks_b_t
            )
        
        pred_idx = int(preds[0].cpu().item())
        pred_label = self._get_label_from_idx(pred_idx)
        confidence = float(probs[0, pred_idx].cpu().item())
        
        prob_dict = {
            self._get_label_from_idx(i): float(probs[0, i].cpu().item())
            for i in range(probs.shape[1])
        }
        
        return {
            "predicted_order": pred_label,
            "confidence": confidence,
            "probabilities": prob_dict,
            "is_correct_order": pred_label == "obv-rev",
        }


@st.cache_resource(show_spinner=False)
def build_side_classifier_from_checkpoint(model_dir: str, device: str = "cuda") -> SideClassifier:
    """
    Build a SideClassifier instance from a trained model directory.
    
    Args:
        model_dir: Path to model output directory containing best_model.pt, run_config.json
        device: Device to load model on
    
    Returns:
        SideClassifier instance
    """
    model, processor, config = load_side_classifier_model(model_dir, device)
    return SideClassifier(model, processor, config, device)
