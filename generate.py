"""Iterative masked-diffusion decoding (Sec. 4.3) — MaskGIT-style confidence-based
parallel decoding.

Starting from a fully-masked accompaniment, we run ``num_steps`` denoising
iterations under a cosine schedule. At each step the model predicts **all** masked
positions in parallel (one forward); the highest-confidence predictions are kept
and the lowest-confidence fraction is re-masked for the next iteration. The number
of tokens kept masked follows ``gamma = cos(pi/2 * step/num_steps)`` so the mask
count monotonically goes 1 -> 0 and the sequence converges.

Two remasking strategies:

* ``"progressive"`` (default, canonical MaskGIT / LaDA-Band): once a token is
  revealed it is *locked in*; only currently-masked positions compete to be
  remasked. The mask set shrinks monotonically.
* ``"reconsider"``: every valid position (including already-revealed ones) is
  re-ranked by its current confidence each step, so an early low-confidence commit
  can be remasked and re-sampled later. Slightly more robust to early mistakes.
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn.functional as F

from .tokens import ACC_MASK_ID, ACC_PAD_ID
from .sampling import sample_tokens, gumbel_noise_like


@torch.no_grad()
def generate_accompaniment(
    model,
    vocal_ids: torch.Tensor,
    pad_mask: torch.Tensor,
    num_steps: int = 20,
    temperature: float = 1.0,
    top_k: int = 100,
    top_p: float = 0.9,
    mask_temp: float = 10.5,
    cond: Optional[torch.Tensor] = None,
    guidance: float = 1.0,
    remask_strategy: str = "progressive",
) -> torch.Tensor:
    """Return generated accompaniment ids ``(B, T)`` (padding kept at pad frames)."""
    assert remask_strategy in ("progressive", "reconsider"), remask_strategy
    model.eval()
    device = vocal_ids.device
    B, T = vocal_ids.shape
    valid = pad_mask.to(device)
    lengths = valid.sum(dim=1)  # (B,)

    acc = torch.full((B, T), ACC_PAD_ID, dtype=torch.long, device=device)
    acc[valid] = ACC_MASK_ID

    for step in range(num_steps):
        is_mask = acc == ACC_MASK_ID
        if not is_mask.any():
            break

        # 1) one parallel forward -> sample every position, with its confidence
        if guidance != 1.0 and cond is not None and model.use_condition_prefix:
            logits = model.predict_logits_cfg(vocal_ids, acc, pad_mask, cond, guidance)
        else:
            logits = model.predict_logits(vocal_ids, acc, pad_mask, cond=cond)
        tokens, conf = sample_tokens(logits, temperature=temperature, top_k=top_k, top_p=top_p)

        # 2) cosine schedule: how many tokens stay masked after this step
        ratio = (step + 1) / num_steps
        gamma = math.cos(0.5 * math.pi * ratio)
        n_mask = torch.floor(gamma * lengths.float()).long()  # (B,)

        # 3) per-position confidence score (+ annealed Gumbel noise = mask temperature)
        if remask_strategy == "reconsider":
            # confidence over ALL valid positions: newly-sampled tokens use their
            # own prob; already-revealed tokens use the prob of their committed id.
            probs = F.softmax(logits.float(), dim=-1)
            committed = acc.clamp(0, logits.shape[-1] - 1)
            committed_conf = probs.gather(-1, committed.unsqueeze(-1)).squeeze(-1)
            conf_all = torch.where(is_mask, conf, committed_conf)
            acc = torch.where(is_mask, tokens, acc)               # commit at masked
            score = conf_all.float().clamp_min(1e-9).log()
            score = score + mask_temp * (1.0 - ratio) * gumbel_noise_like(score)
            score = score.masked_fill(~valid, float("inf"))       # every valid is a candidate
        else:  # progressive: revealed tokens are locked in
            acc = torch.where(is_mask, tokens, acc)
            score = conf.float().clamp_min(1e-9).log()
            score = score + mask_temp * (1.0 - ratio) * gumbel_noise_like(score)
            score = score.masked_fill((~is_mask) | (~valid), float("inf"))

        # 4) re-mask the n_mask lowest-confidence positions for the next iteration
        new_mask = torch.zeros_like(is_mask)
        for b in range(B):
            k = int(n_mask[b].item())
            if k > 0:
                idx = torch.topk(score[b], k, largest=False).indices
                new_mask[b, idx] = True
        acc = torch.where(new_mask, torch.full_like(acc, ACC_MASK_ID), acc)

    # Fill any residual masks deterministically.
    leftover = acc == ACC_MASK_ID
    if leftover.any():
        logits = model.predict_logits(vocal_ids, acc, pad_mask, cond=cond)
        acc = torch.where(leftover, logits.argmax(dim=-1), acc)
    return acc


def split_to_list(acc: torch.Tensor, pad_mask: torch.Tensor) -> List[torch.Tensor]:
    """Trim a padded ``(B, T)`` batch back to a list of per-sample 1-D sequences."""
    out = []
    for b in range(acc.shape[0]):
        n = int(pad_mask[b].sum().item())
        out.append(acc[b, :n].cpu())
    return out
