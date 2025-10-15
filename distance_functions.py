import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any


def flatten_last_dims(x: torch.Tensor) -> torch.Tensor:
    """Flatten all non-batch dims safely for non-contiguous tensors."""
    return x.flatten(1)


class FrobeniusDistance(nn.Module):
    """Frobenius/MSE for matrices or vectors.

    If structured matrices are present in metadata under keys 'intra', 'G', or 'ICC',
    compute MSE over those; otherwise fall back to MSE over flattened vectors.
    """

    def __init__(self) -> None:
        super().__init__()
        self._keys = ('intra', 'G', 'ICC')

    def forward(self, student: Any, teacher: Any) -> torch.Tensor:
        s_vec, s_meta = (student if isinstance(student, tuple) else (student, {}))
        t_vec, t_meta = (teacher if isinstance(teacher, tuple) else (teacher, {}))

        for key in self._keys:
            if key in s_meta and key in t_meta:
                s_val, t_val = s_meta[key], t_meta[key]
                if s_val.shape == t_val.shape:
                    return F.mse_loss(s_val, t_val)
                # shape mismatch -> fallback to vector MSE
                break
        return F.mse_loss(flatten_last_dims(s_vec), flatten_last_dims(t_vec))


class L2Distance(nn.Module):
    """L2/MSE between canonical vectors (or flattened fallback)."""

    def forward(self, student: Any, teacher: Any) -> torch.Tensor:
        s_vec = student[0] if isinstance(student, tuple) else student
        t_vec = teacher[0] if isinstance(teacher, tuple) else teacher
        if s_vec.ndim > 2 or t_vec.ndim > 2:
            s_vec = flatten_last_dims(s_vec)
            t_vec = flatten_last_dims(t_vec)
        return F.mse_loss(s_vec, t_vec)


class KLDivergenceDistance(nn.Module):
    """KL Divergence over probability maps (used in CWD).

    Expects 'probs' in metadata; falls back to L2 over vectors if absent.
    Computes KL(T || S) averaged over all dimensions for stability.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = float(eps)

    def forward(self, student: Any, teacher: Any) -> torch.Tensor:
        s_val, s_meta = (student if isinstance(student, tuple) else (student, {}))
        t_val, t_meta = (teacher if isinstance(teacher, tuple) else (teacher, {}))

        if 'probs' in s_meta and 'probs' in t_meta:
            S = s_meta['probs']
            T = t_meta['probs']
            # Ensure strictly positive for log
            S = S.clamp_min(self.eps)
            T = T.clamp_min(self.eps)
            kl = T * (T.log() - S.log())
            return kl.mean()

        # fallback: L2 if no probs
        s_vec = s_val if not isinstance(student, tuple) else student[0]
        t_vec = t_val if not isinstance(teacher, tuple) else teacher[0]
        if s_vec.ndim > 2 or t_vec.ndim > 2:
            s_vec = flatten_last_dims(s_vec)
            t_vec = flatten_last_dims(t_vec)
        return F.mse_loss(s_vec, t_vec)
