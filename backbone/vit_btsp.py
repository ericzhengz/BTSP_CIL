# -*- coding: utf-8 -*-
"""
ViT backbone for BTSP: wraps timm ViTs to explicitly return (logits, features).
- Supports pretrained timm models (e.g., "vit_base_patch16_224").
- Always exposes self.feature_dim for downstream modules (inc_net / learners).
- This module does NOT own a classification head (managed by inc_net).
"""

from typing import Tuple, Optional, Union
import torch
import torch.nn as nn

try:
    import timm
except Exception as e:
    timm = None


def _ensure_timm():
    if timm is None:
        raise ImportError(
            "timm is not installed. Please install with `pip install timm` "
            "or add it to your environment to use ViT_BTSP."
        )


def _infer_feature_dim(vit_model: nn.Module) -> int:
    # timm ViT models usually expose num_features or embed_dim
    if hasattr(vit_model, "num_features"):
        return int(vit_model.num_features)
    if hasattr(vit_model, "embed_dim"):
        return int(vit_model.embed_dim)
    # fallback: try to run a tiny fake forward (unsafe without device), so avoid.
    raise AttributeError("Cannot infer feature_dim from the timm model. "
                         "Please ensure the model exposes `num_features` or `embed_dim`.")


class ViT_BTSP(nn.Module):
    """
    A thin wrapper over timm Vision Transformers that:
      - creates the model with num_classes=0 (feature extractor)
      - returns both (logits, feat) in forward()
      - optionally attaches a small linear head if num_classes > 0
    """
    def __init__(
        self,
        backbone_type: str = "vit_base_patch16_224",
        pretrained: bool = True,
        num_classes: int = 0,
        global_pool: str = "token",  # "token" | "avg"
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        _ensure_timm()
        self.vit = timm.create_model(
            backbone_type,
            pretrained=pretrained,
            num_classes=0,  # 统一无分类头
            drop_path_rate=drop_path_rate,
        )
        self.global_pool = global_pool
        self.feature_dim = _infer_feature_dim(self.vit)
        self._has_forward_head = hasattr(self.vit, "forward_head")
        self._has_forward_features = hasattr(self.vit, "forward_features")

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get a pooled feature vector [B, D] from timm ViT.
        Handles different return conventions across timm versions.
        """
        if self._has_forward_features:
            feats = self.vit.forward_features(x)
        else:
            # Fallback to standard forward then intercept before head
            feats = self.vit(x)

        # timm sometimes returns dict with keys like "x", "pool"
        if isinstance(feats, dict):
            if "x" in feats and feats["x"] is not None:
                t = feats["x"]  # [B, D] CLS-token or pooled
            elif "pool" in feats and feats["pool"] is not None:
                t = feats["pool"]
            elif "features" in feats and feats["features"] is not None:
                t = feats["features"]
            else:
                # try to stack all and pool
                vals = [v for v in feats.values() if v is not None and isinstance(v, torch.Tensor)]
                if len(vals) == 0:
                    raise RuntimeError("forward_features returned empty dict.")
                t = vals[0]
        else:
            t = feats

        # If features are [B, N, D] token maps, pool them as requested
        if t.dim() == 3:
            if self.global_pool == "avg":
                t = t.mean(dim=1)  # average over tokens
            else:
                # token: pick cls token (index 0) if present
                t = t[:, 0, :]
        return t  # [B, D]

    def forward(self, x: torch.Tensor, return_feat: bool = True) -> Union[torch.Tensor, Tuple[Optional[torch.Tensor], torch.Tensor]]:
        feat = self._forward_features(x)  # [B, D]
        logits = None  # 由inc_net管理分类头
        return (logits, feat) if return_feat else logits
