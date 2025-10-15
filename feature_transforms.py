import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional


def flatten_last_dims(x: torch.Tensor) -> torch.Tensor:
    """Flatten all non-batch dimensions to a single vector per sample.
    Returns a contiguous view with shape [B, -1].
    """
    if x.dim() == 2:
        return x
    return x.reshape(x.size(0), -1)


def global_pool_vec(x: torch.Tensor) -> torch.Tensor:
    """Return a vector per sample using global average pooling.
    - 4D input [B, C, H, W] -> [B, C]
    - 3D input [B, C, T]    -> [B, C]
    - 2D input [B, D]       -> [B, D]
    Otherwise flattens to [B, -1].
    """
    if x.dim() == 4:
        pooled = F.adaptive_avg_pool2d(x, 1)
        return pooled.reshape(x.size(0), -1)
    if x.dim() == 3:
        return x.mean(dim=-1)
    if x.dim() == 2:
        return x
    return x.reshape(x.size(0), -1)


class Projector1D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        if in_dim == out_dim:
            self.net = nn.Identity()
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(inplace=True),
                nn.Linear(out_dim, out_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LKDTransform(nn.Module):
    """Local knowledge distillation transform.

    Splits the feature map into a k x k grid via adaptive pooling, computes
    intra-image patch similarities, and projects the flattened matrix.
    """

    def __init__(self, in_channels: int, k: int = 4, mode: str = "cosine", out_dim: int = 256, gamma: float = 1.0):
        super().__init__()
        self.k = int(k)
        self.mode = mode
        self.out_dim = out_dim
        self.gamma = float(gamma)
        self.projector: Optional[Projector1D] = None

    @staticmethod
    def _cosine_sim(patches: torch.Tensor) -> torch.Tensor:
        # patches: [B, S, C]
        patches = F.normalize(patches, dim=-1)
        return torch.bmm(patches, patches.transpose(1, 2))

    def _gaussian_sim(self, patches: torch.Tensor) -> torch.Tensor:
        # patches: [B, S, C]; compute exp(-||xi-xj||^2 * gamma)
        xx = (patches.pow(2).sum(dim=-1, keepdim=True))  # [B, S, 1]
        yy = xx.transpose(1, 2)                          # [B, 1, S]
        xy = torch.bmm(patches, patches.transpose(1, 2)) # [B, S, S]
        dist_sq = (xx + yy - 2.0 * xy).clamp_min(0.0)
        return torch.exp(-self.gamma * dist_sq)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, C, H, W = feat.shape
        pooled = F.adaptive_avg_pool2d(feat, (self.k, self.k))          # [B, C, k, k]
        patches = pooled.permute(0, 2, 3, 1).reshape(B, self.k * self.k, C)  # [B, S, C]

        if self.mode == "cosine":
            intra = self._cosine_sim(patches)
        else:
            intra = self._gaussian_sim(patches)

        flat = flatten_last_dims(intra)
        if self.projector is None:
            self.projector = Projector1D(flat.size(1), self.out_dim).to(flat.device)
        vec = self.projector(flat)
        return vec, {"intra": intra}


class LRTransform(nn.Module):
    """Logit relation transform that fuses teacher and student global vectors."""

    def __init__(self, in_channels_list: Optional[List[int]] = None, out_dim: int = 256):
        super().__init__()
        self.out_dim = out_dim
        self.projector: Optional[Projector1D] = None
        # in_channels_list kept for backward-compat but not used directly

    def forward(self, teacher_feats: List[torch.Tensor], student_feat: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        teacher_vecs = [global_pool_vec(f) for f in teacher_feats]
        fused_vec = torch.stack(teacher_vecs, dim=0).sum(dim=0)
        student_vec = global_pool_vec(student_feat)
        cat = torch.cat([fused_vec, student_vec], dim=1)
        if self.projector is None:
            self.projector = Projector1D(cat.size(1), self.out_dim).to(cat.device)
        vec = self.projector(cat)
        return vec, {}


class ICKDTransform(nn.Module):
    """Intra-channel correlation transform (C x C correlation per sample)."""

    def __init__(self, in_channels: int, out_dim: int = 256):
        super().__init__()
        self.out_dim = out_dim
        self.projector: Optional[Projector1D] = None

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, C, H, W = feat.shape
        f = F.normalize(feat.reshape(B, C, -1), dim=2)
        ICC = torch.bmm(f, f.transpose(1, 2))
        flat = flatten_last_dims(ICC)
        if self.projector is None:
            self.projector = Projector1D(flat.size(1), self.out_dim).to(flat.device)
        vec = self.projector(flat)
        return vec, {"ICC": ICC}


class CWDTransform(nn.Module):
    """Channel-wise distillation transform using softened spatial distributions per channel."""

    def __init__(self, in_channels: int, out_dim: int = 256, temperature: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.temperature, self.eps = float(temperature), float(eps)
        self.projector = Projector1D(in_channels, out_dim)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, C, H, W = feat.shape
        logits = feat.reshape(B * C, -1) / self.temperature
        soft = F.softmax(logits, dim=1)
        soft = soft.reshape(B, C, H, W).clamp(min=self.eps)
        vec = self.projector(soft.reshape(B, C, -1).mean(dim=2))
        return vec, {"probs": soft}


class SPTransform(nn.Module):
    """Similarity preserving transform based on per-sample channel Gram matrix.

    This avoids dependence on the batch size by computing a [C x C] Gram matrix
    for each sample, then projecting the flattened matrix.
    """

    def __init__(self, in_channels: int, out_dim: int = 256):
        super().__init__()
        self.out_dim = out_dim
        self.projector: Optional[Projector1D] = None
        self.in_channels = int(in_channels)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, C, H, W = feat.shape
        f = feat.reshape(B, C, -1)
        f = f - f.mean(dim=2, keepdim=True)
        G = torch.bmm(f, f.transpose(1, 2)) / (H * W)
        G = F.normalize(G, p=2, dim=2)
        flat = flatten_last_dims(G)
        if self.projector is None:
            self.projector = Projector1D(flat.size(1), self.out_dim).to(G.device)
        vec = self.projector(flat)
        return vec, {"G": G}


class MaskedTransform(nn.Module):
    """Masked reconstruction transform aligning channels then reconstructing masked inputs."""

    def __init__(self, in_channels: int, out_dim: int = 256, mask_ratio: float = 0.5):
        super().__init__()
        self.mask_ratio = float(mask_ratio)
        self.align = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.recon = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
        )
        self.proj = Projector1D(in_channels, out_dim)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, C, H, W = feat.shape
        mask = (torch.rand(B, 1, H, W, device=feat.device) > self.mask_ratio).float()
        aligned = self.align(feat)
        recon = self.recon(aligned * mask)
        vec = self.proj(global_pool_vec(recon))
        return vec, {}


class AttentionTransform(nn.Module):
    """Attention energy pooling transform."""

    def __init__(self, p: int = 2, out_dim: int = 256):
        super().__init__()
        self.p = int(p)
        self.out_dim = out_dim
        self.projector: Optional[Projector1D] = None

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # att: [B, H, W]
        att = feat.abs().pow(self.p).sum(dim=1)
        flat = flatten_last_dims(att)
        if self.projector is None:
            self.projector = Projector1D(flat.size(1), self.out_dim).to(flat.device)
        vec = self.projector(flat)
        return vec, {}


__all__ = [
    "flatten_last_dims",
    "global_pool_vec",
    "Projector1D",
    "LKDTransform",
    "LRTransform",
    "ICKDTransform",
    "CWDTransform",
    "SPTransform",
    "MaskedTransform",
    "AttentionTransform",
]
