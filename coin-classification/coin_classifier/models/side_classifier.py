"""Side classifier model for predicting coin side order using DINOv3 with token-mask encoding."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class DinoV3TokenMaskEncoder(nn.Module):
    """Encoder that pools patch tokens using binary masks."""
    
    def __init__(self, backbone: nn.Module, input_size: int, eps: float = 1e-6, mask_pooling: bool = True):
        super().__init__()
        self.backbone = backbone
        self.input_size = int(input_size)
        self.eps = float(eps)
        self.mask_pooling = mask_pooling

    def _encode_pooled_tokens(self, pixel_values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Encode using mask-weighted pooling of patch tokens."""
        outputs = self.backbone(pixel_values=pixel_values)
        tokens = outputs.last_hidden_state  # [B, 1 + R + N, D]

        num_register = int(getattr(self.backbone.config, "num_register_tokens", 0) or 0)
        patch_size = getattr(self.backbone.config, "patch_size", None)
        if isinstance(patch_size, (tuple, list)):
            patch_size = patch_size[0]
        if not isinstance(patch_size, int) or patch_size <= 0:
            raise ValueError(f"Invalid patch_size in model config: {patch_size}")

        b, _, img_h, img_w = pixel_values.shape
        if (img_h % patch_size) != 0 or (img_w % patch_size) != 0:
            raise ValueError(
                f"pixel_values spatial size must be divisible by patch_size. "
                f"Got HxW={img_h}x{img_w}, patch_size={patch_size}"
            )

        gh, gw = img_h // patch_size, img_w // patch_size
        num_patches_flat = gh * gw

        patch_start = 1 + num_register
        patch_tokens_flat = tokens[:, patch_start:, :]  # [B, N, D]
        if patch_tokens_flat.shape[1] != num_patches_flat:
            raise ValueError(
                "Patch token count mismatch. "
                f"Expected {num_patches_flat} (from HxW={img_h}x{img_w}, patch_size={patch_size}) "
                f"but got {patch_tokens_flat.shape[1]} from last_hidden_state "
                f"with num_register_tokens={num_register}."
            )

        # Downsample mask to patch grid
        mask = masks.unsqueeze(1)  # [B, 1, H, W]
        if mask.shape[-2:] != (img_h, img_w):
            mask = F.interpolate(mask, size=(img_h, img_w), mode="nearest")
        mask_patches = F.interpolate(mask, size=(gh, gw), mode="nearest").flatten(1)  # [B, N]

        # Handle empty masks
        denom = mask_patches.sum(dim=1, keepdim=True)
        is_empty = denom <= self.eps
        if is_empty.any():
            mask_patches = torch.where(is_empty, torch.ones_like(mask_patches), mask_patches)
            denom = mask_patches.sum(dim=1, keepdim=True)

        # Weighted pooling
        pooled = (patch_tokens_flat * mask_patches.unsqueeze(-1)).sum(dim=1) / (denom + self.eps)
        return pooled

    def forward(self, pixel_values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Encode images using mask pooling or CLS token."""
        if self.mask_pooling:
            return self._encode_pooled_tokens(pixel_values, masks)
        else:
            # Standard CLS token
            outputs = self.backbone(pixel_values=pixel_values)
            return outputs.last_hidden_state[:, 0, :]


class DinoV3SideClassifier(nn.Module):
    """Binary classifier for coin side order prediction (obv-rev vs rev-obv)."""
    
    def __init__(
        self,
        model_name: str,
        input_size: int = 224,
        mask_pooling: bool = True,
    ):
        super().__init__()
        
        self.mask_pooling = mask_pooling
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = int(self.backbone.config.hidden_size)
        
        self.encoder = DinoV3TokenMaskEncoder(
            self.backbone,
            input_size=input_size,
            mask_pooling=mask_pooling
        )
        
        # Binary classification: 0=obv-rev, 1=rev-obv
        self.head = nn.Linear(hidden_size * 2, 2)
    
    def forward(
        self,
        pixel_values_a: torch.Tensor,
        masks_a: torch.Tensor,
        pixel_values_b: torch.Tensor,
        masks_b: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with two images and their masks."""
        fa = self.encoder(pixel_values_a, masks_a)  # [B, D]
        fb = self.encoder(pixel_values_b, masks_b)  # [B, D]
        z = torch.cat([fa, fb], dim=1)  # [B, 2D]
        return self.head(z)
    
    def predict_side_order(
        self,
        pixel_values_a: torch.Tensor,
        masks_a: torch.Tensor,
        pixel_values_b: torch.Tensor,
        masks_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict side order and return class indices and probabilities.
        
        Returns:
            (predicted_class, probabilities) where:
            - predicted_class: 0=obv-rev, 1=rev-obv
            - probabilities: [B, 2] tensor of class probabilities
        """
        logits = self.forward(pixel_values_a, masks_a, pixel_values_b, masks_b)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        return preds, probs
