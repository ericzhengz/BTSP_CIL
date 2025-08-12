# -*- coding: utf-8 -*-
"""
Incremental network for BTSP on top of ViT backbone.
Provides:
  - feature extraction (extract_vector)
  - classifier head (linear or cosine) with update_fc for class increments
  - encode_to_bits / encode_feats_to_bits for sparse binarization

Drop-in for models/btsp_mp.py without touching other methods in PILOT.
Author: your_name
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone.vit_btsp import ViT_BTSP


# ---------------------------
#  Cosine classifier (optional)
# ---------------------------
class CosineLinear(nn.Module):
    """
    Cosine classifier with learnable scale.
    logits = s * cos(theta) = s * (normalize(W) · normalize(x))
    """
    def __init__(self, in_features: int, out_features: int, learnable_scale: bool = True, init_scale: float = 10.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_normal_(self.weight, nonlinearity="linear")
        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(float(init_scale)))
        else:
            self.register_buffer("scale", torch.tensor(float(init_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n = F.normalize(x, dim=1)
        w_n = F.normalize(self.weight, dim=1)
        logits = torch.matmul(x_n, w_n.t()) * self.scale
        return logits


# ---------------------------
#  Incremental Net for BTSP
# ---------------------------
class BTSPIncrementalNet(nn.Module):
    """
    Minimal incremental network used by BTSP learner:
      - backbone: ViT_BTSP
      - head: linear or cosine (optional)
      - extract_vector / update_fc
      - encode_to_bits (Top-k binarization) for BTSP memory writing
    """
    def __init__(self, args: Dict[str, Any], with_fc: bool = True):
        super().__init__()
        self.args = args or {}
        self.with_fc = with_fc

        # ---- Backbone ----
        bb_type = self.args.get("backbone_type", "vit_base_patch16_224")
        self.backbone_name = bb_type
        pretrained = bool(self.args.get("pretrained", True))
        init_classes = int(self.args.get("init_cls", self.args.get("num_classes", 0)))

        # 兼容常用ViT命名（如 in21k），统一让ViT仅产出特征（num_classes=0），分类头由inc_net接管
        if bb_type.startswith("vit_base_patch16_224") or bb_type in ["vit_btsp"]:
            self.backbone = ViT_BTSP(
                backbone_type=bb_type if bb_type != "vit_btsp" else "vit_base_patch16_224",
                pretrained=pretrained,
                num_classes=0,  # 关键：始终为0，避免骨干头与inc_net头重复
                global_pool=self.args.get("global_pool", "token"),
                drop_path_rate=float(self.args.get("drop_path", 0.0)),
            )
        else:
            raise ValueError(f"BTSPIncrementalNet currently supports ViT backbones, got: {bb_type}")

        self.feature_dim: int = int(self.backbone.feature_dim)

        # ---- Memory projection space for BTSP ----
        self.memory_dim: int = int(self.args.get("btsp_memory_dim", self.feature_dim))
        if self.memory_dim != self.feature_dim:
            self.feature_proj = nn.Linear(self.feature_dim, self.memory_dim, bias=False)
            nn.init.orthogonal_(self.feature_proj.weight)
            proj_trainable = bool(self.args.get("btsp_proj_trainable", True))
            for p in self.feature_proj.parameters():
                p.requires_grad = proj_trainable
        else:
            self.feature_proj = nn.Identity()

        # ---- Head ----
        self.head_type = str(self.args.get("head_type", "linear"))  # "linear" | "cosine" | "none"
        self.normalize = bool(self.args.get("normalize", (self.head_type == "cosine")))
        self.learnable_scale = bool(self.args.get("learnable_scale", True))
        self.init_scale = float(self.args.get("init_scale", 10.0))

        if self.with_fc and self.head_type != "none":
            if self.head_type == "cosine":
                self.fc = CosineLinear(self.feature_dim, init_classes, self.learnable_scale, self.init_scale)
            else:
                self.fc = nn.Linear(self.feature_dim, init_classes, bias=True)
        else:
            self.fc = None

    # --------------- Basic APIs ---------------
    @torch.no_grad()
    def extract_vector(self, x: torch.Tensor) -> torch.Tensor:
        """Return features [B, D] without grads (for evaluation or memory writing)."""
        _ = self.training
        self.eval()
        logits, feat = self.backbone(x, return_feat=True)
        return feat

    def update_fc(self, num_classes: int):
        """Expand classifier head to `num_classes` outputs while preserving old weights."""
        if not self.with_fc or self.head_type == "none":
            return

        if isinstance(self.fc, CosineLinear):
            new_fc = CosineLinear(self.feature_dim, num_classes,
                                  learnable_scale=isinstance(self.fc.scale, nn.Parameter),
                                  init_scale=float(self.fc.scale.detach().cpu().item()))
            with torch.no_grad():
                old = self.fc.weight.data
                new_fc.weight.data[: old.size(0)] = old
            # keep on same device
            new_fc = new_fc.to(self.fc.weight.device)
            self.fc = new_fc
        elif isinstance(self.fc, nn.Linear):
            new_fc = nn.Linear(self.feature_dim, num_classes, bias=True).to(self.fc.weight.device)
            with torch.no_grad():
                old_w = self.fc.weight.data
                new_fc.weight.data[: old_w.size(0)] = old_w
                if self.fc.bias is not None:
                    old_b = self.fc.bias.data
                    new_fc.bias.data[: old_b.size(0)] = old_b
            self.fc = new_fc
        else:
            raise TypeError("Unsupported head type for update_fc.")

    def forward(self, x: torch.Tensor):
        """
        Returns:
          {"logits": logits, "feat": feat}
        """
        _, feat = self.backbone(x, return_feat=True)  # ViT无头

        logits = None
        if self.with_fc and self.head_type != "none":
            if isinstance(self.fc, CosineLinear):
                logits = self.fc(feat)
            elif isinstance(self.fc, nn.Linear):
                if self.normalize:
                    logits = self.fc(F.normalize(feat, dim=1))
                else:
                    logits = self.fc(feat)
            else:
                raise TypeError("Unsupported head in BTSPIncrementalNet.")
        return {"logits": logits, "feat": feat}

    # --------------- Extended APIs ---------------
    def extract_memory_vector(self, x: torch.Tensor, with_grad: bool = False) -> torch.Tensor:
        """
        返回映射到记忆空间的向量 [B, N]，用于BTSP编码/分析。
        with_grad=False时不追踪梯度。
        """
        if with_grad:
            _, feat = self.backbone(x, return_feat=True)
            return self.feature_proj(feat)
        else:
            with torch.no_grad():
                self.eval()
                _, feat = self.backbone(x, return_feat=True)
                return self.feature_proj(feat)

    def freeze_backbone(self, eval_mode: bool = True):
        for p in self.backbone.parameters():
            p.requires_grad = False
        if eval_mode:
            self.backbone.eval()
        return self

    def unfreeze_backbone(self, train_mode: bool = True):
        for p in self.backbone.parameters():
            p.requires_grad = True
        if train_mode:
            self.backbone.train()
        return self

    @property
    def memory_space_dim(self) -> int:
        return self.memory_dim

    def get_trainable_parameters(self):
        """便于外部构建优化器，返回当前requires_grad=True的参数迭代器。"""
        return (p for p in self.parameters() if p.requires_grad)

    def head_forward(self, feat: torch.Tensor):
        """只用已抽好的 feat 过投影与头，避免再次前向骨干"""
        # 投影到记忆空间
        mem = self.feature_proj(feat) if not isinstance(self.feature_proj, nn.Identity) else feat
        
        if not self.with_fc or self.head_type == "none":
            return None
            
        if self.head_type == "cosine":
            return self.fc(feat)  # CosineLinear 内部会归一化
        elif self.head_type == "linear":
            if self.normalize:
                return self.fc(F.normalize(feat, dim=1))
            else:
                return self.fc(feat)
        else:
            raise TypeError("Unsupported head type in head_forward.")

    # --------------- Binarization APIs ---------------
    @torch.no_grad()
    def encode_feats_to_bits(
        self,
        feat: torch.Tensor,
        sparsity: float = 0.04,
        k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        将连续特征映射到记忆空间后做Top-k二值化：
          - 输入feat形状 [B, D_feature]
          - 线性映射到 [B, N_memory] 后进行Top-k
        返回：bool张量 [B, N_memory]
        """
        # 映射到记忆空间
        mem = self.feature_proj(feat)  # [B, N]
        # 轻量归一化，提升阈值稳健性
        mem = F.layer_norm(mem, (mem.size(1),))
        B, N = mem.shape
        if k is None:
            k = max(1, int(N * float(sparsity)))
        # 确定性微扰（哈希索引），避免随机噪声导致不可复现
        i = torch.arange(N, device=mem.device).float()
        phi = ((i * 2654435761) % 2**32 - 2**31) / (2**31)  # ~[-1,1]
        eps = 1e-12 * phi.unsqueeze(0).expand_as(mem)
        idx = torch.topk(mem + eps, k, dim=1).indices
        code = torch.zeros(B, N, dtype=torch.bool, device=mem.device)
        code.scatter_(1, idx, True)
        return code  # bool tensor

    @torch.no_grad()
    def encode_to_bits(
        self,
        x: torch.Tensor,
        sparsity: float = 0.04,
        k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Convenience: compute features then binarize.
        """
        _ = self.training
        self.eval()
        _, feat = self.backbone(x, return_feat=True)
        return self.encode_feats_to_bits(feat, sparsity=sparsity, k=k)
