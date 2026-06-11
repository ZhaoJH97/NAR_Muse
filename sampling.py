"""Token sampling utilities for iterative masked-diffusion decoding.

Provides temperature / top-k / top-p (nucleus) filtering and the confidence
score used by the low-confidence remasking strategy (Sec. 4.3 of the paper).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def _top_k_top_p_filter(
    logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0, filter_value: float = float("-inf")
) -> torch.Tensor:
    """Filter a logits tensor ``(..., V)`` in-place style, returning filtered logits."""
    logits = logits.clone()
    if top_k and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
        logits = torch.where(logits < kth, torch.full_like(logits, filter_value), logits)
    if top_p and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cum > top_p
        # keep at least the top-1 token
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        remove = remove.scatter(-1, sorted_idx, remove)
        logits = torch.where(remove, torch.full_like(logits, filter_value), logits)
    return logits


def sample_tokens(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample one token per position and return its probability (confidence).

    Args:
        logits: (B, T, V) raw accompaniment logits.
    Returns:
        tokens:     (B, T) sampled token ids.
        confidence: (B, T) probability assigned to the sampled token under the
                    *unfiltered* softmax (used for low-confidence remasking).
    """
    B, T, V = logits.shape
    probs_full = F.softmax(logits.float(), dim=-1)

    if temperature <= 0:
        tokens = logits.argmax(dim=-1)
    else:
        filtered = _top_k_top_p_filter(logits.float() / temperature, top_k=top_k, top_p=top_p)
        probs = F.softmax(filtered, dim=-1)
        tokens = torch.multinomial(probs.reshape(-1, V), num_samples=1).reshape(B, T)

    confidence = torch.gather(probs_full, -1, tokens.unsqueeze(-1)).squeeze(-1)
    return tokens, confidence


def gumbel_noise_like(x: torch.Tensor) -> torch.Tensor:
    """Sample standard Gumbel(0,1) noise with the shape of ``x``."""
    u = torch.rand_like(x).clamp_(1e-9, 1.0 - 1e-9)
    return -torch.log(-torch.log(u))
