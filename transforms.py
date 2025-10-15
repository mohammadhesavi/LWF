import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Sequence, Tuple, Optional


def flatten_last_dims(x: torch.Tensor) -> torch.Tensor:
    """Flatten all dimensions after the batch dimension.
    Uses Tensor.flatten(1) which is safe for non-contiguous tensors.
    """
    return x.flatten(1)


def global_pool_vec(x: torch.Tensor) -> torch.Tensor:
    """Return a vector per sample using global pooling or flattening.
    - 4D: B x C x H x W -> adaptive avg pool to 1x1, then flatten -> B x C
    - 3D: B x C x L -> mean over L -> B x C
    - else: flatten all but batch -> B x *
    """
    if x.dim() == 4:
        return F.adaptive_avg_pool2d(x, output_size=1).flatten(1)
    if x.dim() == 3:
        return x.mean(dim=-1)
    return x.flatten(1)


class Projector1D(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
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

    Splits a feature map into k x k pooled patches, computes intra-patch
    similarity for each sample, flattens and projects to a fixed-size vector.

    mode: 'cosine' for cosine similarity, 'rbf' for exp(-||x-y||^2).
    """

    def __init__(self, in_channels: int, k: int = 4, mode: str = 'cosine', out_dim: int = 256) -> None:
        super().__init__()
        self.k = int(k)
        if mode not in {'cosine', 'rbf'}:
            raise ValueError("mode must be 'cosine' or 'rbf'")
        self.mode = mode
        self.out_dim = out_dim
        self.projector: Optional[Projector1D] = None

    def _sim(self, patch_feats: torch.Tensor) -> torch.Tensor:
        """Compute similarity between patches per sample.
        patch_feats: B x P x C
        returns: B x P x P
        """
        if self.mode == 'cosine':
            x = F.normalize(patch_feats, dim=-1)
            return x @ x.transpose(-1, -2)
        # rbf with batched cdist
        # torch.cdist supports batched leading dimensions
        dist = torch.cdist(patch_feats, patch_feats, p=2)
        return torch.exp(-(dist.pow(2)))

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        if feat.dim() != 4:
            raise ValueError('LKDTransform expects 4D tensor (B,C,H,W)')
        B, C, H, W = feat.shape
        # Average pool to k x k cells to avoid divisibility issues
        pooled = F.adaptive_avg_pool2d(feat, output_size=(self.k, self.k))  # B x C x k x k
        patches = pooled.flatten(2).transpose(1, 2)  # B x P x C where P=k*k
        intra = self._sim(patches)
        flat = flatten_last_dims(intra)
        if self.projector is None or (isinstance(self.projector, Projector1D) and getattr(self.projector.net, 'in_features', None) is None and flat.size(1) != self.out_dim):
            # Initialize projector on first use based on computed dimension
            self.projector = Projector1D(flat.size(1), self.out_dim).to(flat.device)
        vec = self.projector(flat)
        return vec, {'intra': intra}


class LRTransform(nn.Module):
    """Linear projection over concatenated pooled vectors from teacher(s) and student.

    Pass a list of in-channel sizes (for the concatenated pooled vectors) at init
    time so the internal projector can be sized deterministically.
    """

    def __init__(self, in_channels_list: Sequence[int], out_dim: int = 256) -> None:
        super().__init__()
        total = int(sum(in_channels_list))
        self.expected_in_dim = total
        self.projector = Projector1D(total, out_dim)

    def forward(self, teacher_feats: Sequence[torch.Tensor], student_feat: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        teacher_vecs = [global_pool_vec(t) for t in teacher_feats]
        student_vec = global_pool_vec(student_feat)
        cat = torch.cat(teacher_vecs + [student_vec], dim=1)
        if cat.size(1) != self.expected_in_dim:
            raise ValueError(f'LRTransform: concatenated dim {cat.size(1)} != expected {self.expected_in_dim}')
        vec = self.projector(cat)
        return vec, {}


class ICKDTransform(nn.Module):
    """Intra-channel correlation transform.

    Computes channel-channel correlation per sample and projects the flattened
    CxC matrix to a fixed-size vector.
    """

    def __init__(self, in_channels: int, out_dim: int = 256) -> None:
        super().__init__()
        self.projector: Optional[Projector1D] = None
        self.out_dim = out_dim

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        if feat.dim() != 4:
            raise ValueError('ICKDTransform expects 4D tensor (B,C,H,W)')
        B, C, H, W = feat.shape
        f = F.normalize(feat.flatten(2), dim=2)  # B x C x HW
        ICC = torch.matmul(f, f.transpose(1, 2))  # B x C x C
        flat = flatten_last_dims(ICC)
        if self.projector is None:
            self.projector = Projector1D(flat.size(1), self.out_dim).to(flat.device)
        vec = self.projector(flat)
        return vec, {'ICC': ICC}


class CWDTransform(nn.Module):
    """Channel-wise distillation transform.

    Applies spatial softmax per channel, then averages over spatial dimension
    and projects to a fixed-size vector.
    """

    def __init__(self, in_channels: int, out_dim: int = 256, temperature: float = 1.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.temperature, self.eps = float(temperature), float(eps)
        self.projector = Projector1D(in_channels, out_dim)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        if feat.dim() != 4:
            raise ValueError('CWDTransform expects 4D tensor (B,C,H,W)')
        B, C, H, W = feat.shape
        logits = feat.flatten(2) / self.temperature  # B x C x HW
        soft = torch.softmax(logits, dim=2).clamp(min=self.eps)
        vec = self.projector(soft.mean(dim=2))  # B x C -> projector -> B x out_dim
        soft_spatial = soft.view(B, C, H, W)
        return vec, {'probs': soft_spatial}


class SPTransform(nn.Module):
    """Similarity-preserving transform (per-sample channel Gram matrix).

    Computes per-sample Gram matrix across channels (B x C x C), then projects
    the flattened representation to a fixed-size vector. This avoids dependency
    on the batch size.
    """

    def __init__(self, in_channels: int, out_dim: int = 256) -> None:
        super().__init__()
        self.projector: Optional[Projector1D] = None
        self.out_dim = out_dim
        self.in_channels = in_channels

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        if feat.dim() != 4:
            raise ValueError('SPTransform expects 4D tensor (B,C,H,W)')
        f = F.normalize(feat.flatten(2), p=2, dim=2)  # B x C x HW
        G = torch.matmul(f, f.transpose(1, 2))  # B x C x C
        flat = flatten_last_dims(G)
        if self.projector is None:
            self.projector = Projector1D(flat.size(1), self.out_dim).to(G.device)
        vec = self.projector(flat)
        return vec, {'G': G}


class MaskedTransform(nn.Module):
    """Masked reconstruction transform.

    Randomly masks spatial positions and tries to reconstruct features with a
    shallow conv net, then pools and projects.
    """

    def __init__(self, in_channels: int, out_dim: int = 256, mask_ratio: float = 0.5) -> None:
        super().__init__()
        self.mask_ratio = float(mask_ratio)
        self.align = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.recon = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
        )
        self.proj = Projector1D(in_channels, out_dim)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        if feat.dim() != 4:
            raise ValueError('MaskedTransform expects 4D tensor (B,C,H,W)')
        B, C, H, W = feat.shape
        mask = (torch.rand(B, 1, H, W, device=feat.device) > self.mask_ratio).float()
        aligned = self.align(feat)
        recon = self.recon(aligned * mask)
        vec = self.proj(global_pool_vec(recon))
        return vec, {'mask': mask}


class AttentionTransform(nn.Module):
    """Attention-based transform using Lp-activation maps.

    Sums channel-wise |A|^p to form attention maps, flattens, and projects.
    """

    def __init__(self, p: float = 2.0, out_dim: int = 256) -> None:
        super().__init__()
        self.p = float(p)
        self.projector: Optional[Projector1D] = None
        self.out_dim = out_dim

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        if feat.dim() != 4:
            raise ValueError('AttentionTransform expects 4D tensor (B,C,H,W)')
        att = feat.abs().pow(self.p).sum(dim=1)  # B x H x W
        flat = flatten_last_dims(att)
        if self.projector is None:
            self.projector = Projector1D(flat.size(1), self.out_dim).to(flat.device)
        vec = self.projector(flat)
        return vec, {}
