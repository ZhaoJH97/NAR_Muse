"""Masking schedule and forward (corruption) process for discrete masked diffusion.

Implements the LaDA-Band training-time forward process: each accompaniment token
is independently replaced by ``[MASK]`` with probability ``t`` (Eq. (3) region),
where ``t`` is drawn from a cosine schedule over a quasi-random (Sobol) progress
variable ``r`` (see Sec. 5.2 of the paper).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch


def cosine_mask_prob(r: torch.Tensor) -> torch.Tensor:
    """Map a progress variable ``r in [0, 1]`` to a mask probability via a cosine
    schedule. ``r = 0`` -> ``t = 1`` (fully masked), ``r = 1`` -> ``t = 0``."""
    return torch.cos(0.5 * math.pi * r.clamp(0.0, 1.0))


class SobolMaskSampler:
    """Draws per-sample mask probabilities ``t`` using a scrambled Sobol sequence.

    Low-discrepancy sampling gives smoother coverage of the mask-ratio range
    within and across batches than i.i.d. uniform sampling.
    """

    def __init__(self, min_mask_prob: float = 1e-3, seed: int = 0):
        self.min_mask_prob = float(min_mask_prob)
        self._engine = torch.quasirandom.SobolEngine(dimension=1, scramble=True, seed=seed)

    def sample(self, batch_size: int, device: torch.device) -> torch.Tensor:
        r = self._engine.draw(batch_size).reshape(-1)            # (B,) in [0, 1)
        t = cosine_mask_prob(r).clamp(min=self.min_mask_prob, max=1.0)
        return t.to(device)


def forward_mask(
    acc_ids: torch.Tensor,
    pad_mask: torch.Tensor,
    t: torch.Tensor,
    mask_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Independently mask accompaniment tokens with per-sample probability ``t``.

    Args:
        acc_ids:  (B, T) ground-truth accompaniment token ids.
        pad_mask: (B, T) bool, True for real (non-padding) frames.
        t:        (B,)   per-sample mask probability.
        mask_id:  the ``[MASK]`` token id to write at masked positions.

    Returns:
        masked_acc:     (B, T) with ``mask_id`` at masked positions.
        mask_positions: (B, T) bool, True exactly where a token was masked.
    """
    B, T = acc_ids.shape
    rand = torch.rand(B, T, device=acc_ids.device)
    mask = (rand < t[:, None]) & pad_mask

    # Guarantee at least one masked token per (non-empty) sample so the CML loss
    # is well-defined and the 1/t weighting never divides an all-zero numerator.
    lengths = pad_mask.sum(dim=1)
    none_masked = (mask.sum(dim=1) == 0) & (lengths > 0)
    for b in torch.nonzero(none_masked, as_tuple=False).flatten().tolist():
        valid = torch.nonzero(pad_mask[b], as_tuple=False).flatten()
        j = valid[torch.randint(len(valid), (1,), device=valid.device)]
        mask[b, j] = True

    masked_acc = torch.where(mask, torch.full_like(acc_ids, mask_id), acc_ids)
    return masked_acc, mask
