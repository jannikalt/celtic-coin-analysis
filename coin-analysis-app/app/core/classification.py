"""Classification module for coin type prediction using trained classifiers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import streamlit as st
from transformers import AutoImageProcessor

from coin_classifier.models.classifier import DinoV3Classifier


@st.cache_resource(show_spinner=False)
def load_classifier_model(
    model_dir: str,
    device: str = "cuda",
) -> tuple[DinoV3Classifier, dict, AutoImageProcessor, str]:
    """
    Load a trained classifier from output directory.
    
    Returns:
        (model, label_encoder_dict, processor, views)
    """
    model_dir = Path(model_dir)
    
    # Load run config
    config_path = model_dir / "run_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"run_config.json not found in {model_dir}")
    
    with open(config_path, "r") as f:
        run_config = json.load(f)
    
    # Load label encoder
    le_path = model_dir / "label_encoder.json"
    if not le_path.exists():
        raise FileNotFoundError(f"label_encoder.json not found in {model_dir}")
    
    with open(le_path, "r") as f:
        le_data = json.load(f)
    
    # Load model checkpoint
    ckpt_path = model_dir / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"best_model.pt not found in {model_dir}")
    
    model_name = run_config.get("model_name", "facebook/dinov3-vits16-pretrain-lvd1689m")
    views = run_config.get("views", "both_concat")
    head_type = run_config.get("head_type", "linear")
    head_hidden = run_config.get("head_hidden", 512)
    head_dropout = run_config.get("head_dropout", 0.1)
    num_classes = len(le_data["classes"])
    
    # Initialize model
    model = DinoV3Classifier(
        model_name=model_name,
        num_classes=num_classes,
        views=views,
        head_type=head_type,
        head_hidden=head_hidden,
        head_dropout=head_dropout,
    )
    
    # Load weights
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    # Load processor
    processor = AutoImageProcessor.from_pretrained(model_name)
    
    return model, le_data, processor, views


class CoinClassifier:
    """Wrapper for coin classification inference."""
    
    def __init__(
        self,
        model: DinoV3Classifier,
        label_encoder: dict,
        processor: AutoImageProcessor,
        views: str,
        device: str,
    ):
        self.model = model
        self.label_encoder = label_encoder
        self.processor = processor
        self.views = views
        self.device = device
        self.classes = label_encoder["classes"]
    
    @torch.no_grad()
    def predict(
        self,
        obverse: Image.Image | None = None,
        reverse: Image.Image | None = None,
    ) -> dict[str, Any]:
        """
        Predict coin class from obverse and/or reverse images.
        
        Returns:
            {
                "predicted_class": str,
                "confidence": float,
                "probabilities": dict[str, float],
            }
        """
        # Validate inputs
        if self.views == "rev" and reverse is None:
            raise ValueError("Model requires reverse image but none provided")
        if self.views == "obv" and obverse is None:
            raise ValueError("Model requires obverse image but none provided")
        if self.views in ["both_concat", "both_avg"] and (obverse is None or reverse is None):
            raise ValueError(f"Model requires both images (views={self.views})")
        
        # Prepare inputs
        pixel_values_obv = None
        pixel_values_rev = None
        
        if obverse is not None and self.views in ["obv", "both_concat", "both_avg"]:
            if obverse.mode != "RGB":
                obverse = obverse.convert("RGB")
            inputs = self.processor(images=[obverse], return_tensors="pt")
            pixel_values_obv = inputs["pixel_values"].to(self.device)
        
        if reverse is not None and self.views in ["rev", "both_concat", "both_avg"]:
            if reverse.mode != "RGB":
                reverse = reverse.convert("RGB")
            inputs = self.processor(images=[reverse], return_tensors="pt")
            pixel_values_rev = inputs["pixel_values"].to(self.device)
        
        # Forward pass
        logits = self.model(pixel_values_rev=pixel_values_rev, pixel_values_obv=pixel_values_obv)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        pred_idx = int(np.argmax(probs))
        pred_class = self.classes[pred_idx]
        confidence = float(probs[pred_idx])
        
        # Get top-k probabilities
        top_k = min(10, len(self.classes))
        top_indices = np.argsort(probs)[::-1][:top_k]
        
        return {
            "predicted_class": pred_class,
            "confidence": confidence,
            "probabilities": {
                self.classes[i]: float(probs[i])
                for i in top_indices
            },
            "all_probabilities": {
                self.classes[i]: float(probs[i])
                for i in range(len(self.classes))
            },
        }


@st.cache_resource(show_spinner=False)
def build_classifier_from_checkpoint(
    model_dir: str,
    device: str = "cuda",
) -> CoinClassifier:
    """Build classifier from checkpoint directory."""
    if not torch.cuda.is_available() and device == "cuda":
        device = "cpu"
    
    model, label_encoder, processor, views = load_classifier_model(model_dir, device)
    
    return CoinClassifier(
        model=model,
        label_encoder=label_encoder,
        processor=processor,
        views=views,
        device=device,
    )
