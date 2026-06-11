"""LaDA-Band Vocal-to-Accompaniment masked-diffusion model.

Architecture (Sec. 4 of arXiv:2604.11052), adapted to the Muse
``qwen3-0.6B-music`` backbone:

    vocal_ids  --[vocal_embed]--\\
                                 concat(feature) --[input_proj]--> X (B,T,D)
    masked_acc --[acc_embed]----/

    X  --[ (optional) prepend condition prefix ]-->  Qwen3 backbone (BIDIRECTIONAL)
       --> hidden (B,T,D) --[acc_head]--> logits (B,T,16384)

Losses: conditional masked modelling (Eq. 3) + optional replaced-token
detection (Eq. 4), combined as ``L = L_CML + lambda * L_RTD`` (Eq. 5).
"""

from __future__ import annotations

from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .tokens import (
    NUM_AUDIO_TOKENS,
    ACC_MASK_ID,
    ACC_PAD_ID,
    VOCAL_PAD_ID,
    ACC_INPUT_VOCAB,
    VOCAL_INPUT_VOCAB,
    ACC_OUTPUT_VOCAB,
)
from .masking import SobolMaskSampler, forward_mask


def _build_backbone(cfg: ModelConfig):
    """Return ``(backbone, audio_rows)`` where ``backbone`` is a Qwen3Model with
    its token embedding stripped, and ``audio_rows`` are the pretrained
    ``<AUDIO_*>`` embedding rows for warm-starting (or ``None``)."""
    from transformers import Qwen3Config

    audio_rows: Optional[torch.Tensor] = None

    if cfg.pretrained_backbone:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        full = AutoModelForCausalLM.from_pretrained(
            cfg.pretrained_backbone,
            attn_implementation=cfg.attn_implementation,
            dtype=torch.float32,
        )
        backbone = full.model  # Qwen3Model

        if cfg.warm_start_audio_embeddings and cfg.track_embed_dim == backbone.config.hidden_size:
            def _resolve_rows():
                tok = AutoTokenizer.from_pretrained(cfg.pretrained_backbone)
                ids = [tok.convert_tokens_to_ids(cfg.audio_token_template.format(i))
                       for i in range(NUM_AUDIO_TOKENS)]
                unk = tok.unk_token_id
                missing = [i for i, tid in enumerate(ids)
                           if tid is None or tid < 0 or tid == unk]
                if missing:
                    raise KeyError(
                        f"{len(missing)} audio tokens not found via template "
                        f"'{cfg.audio_token_template}' (e.g. id {missing[0]}). "
                        f"Set model.audio_token_template to match your vocab.")
                emb = full.get_input_embeddings().weight.data  # (V, D)
                rows = emb[torch.tensor(ids, dtype=torch.long)].clone()
                print(f"[lada_band] warm-start: resolved {NUM_AUDIO_TOKENS} audio rows, "
                      f"token-id range [{min(ids)}, {max(ids)}] from '{cfg.pretrained_backbone}'.")
                return rows
            try:
                audio_rows = _resolve_rows()
            except Exception as e:
                msg = f"[lada_band] warm-start FAILED: {e}"
                if cfg.warm_start_strict:
                    raise RuntimeError(msg)
                print(msg + "  -> falling back to random audio embeddings.")
        elif cfg.warm_start_audio_embeddings:
            print(f"[lada_band] warm-start skipped: track_embed_dim "
                  f"({cfg.track_embed_dim}) != hidden_size ({backbone.config.hidden_size}).")
        # Free the large (vocab x D) embedding / lm_head we do not use.
        backbone.embed_tokens = None
        if hasattr(full, "lm_head"):
            full.lm_head = None
    else:
        qcfg = Qwen3Config(
            hidden_size=cfg.hidden_size,
            num_hidden_layers=cfg.num_hidden_layers,
            num_attention_heads=cfg.num_attention_heads,
            num_key_value_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            intermediate_size=cfg.intermediate_size,
            rms_norm_eps=cfg.rms_norm_eps,
            rope_theta=cfg.rope_theta,
            max_position_embeddings=cfg.max_position_embeddings,
            vocab_size=8,  # tiny dummy; embed_tokens is discarded anyway
            attn_implementation=cfg.attn_implementation,
        )
        from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

        backbone = Qwen3Model(qcfg)
        backbone.embed_tokens = None

    backbone.config._attn_implementation = cfg.attn_implementation
    return backbone, audio_rows


class LaDABandV2A(nn.Module):
    def __init__(self, cfg: ModelConfig, mask_seed: int = 0):
        super().__init__()
        self.cfg = cfg
        self.backbone, audio_rows = _build_backbone(cfg)
        self.hidden_size = self.backbone.config.hidden_size
        d_embed = cfg.track_embed_dim

        self.vocal_embed = nn.Embedding(VOCAL_INPUT_VOCAB, d_embed, padding_idx=VOCAL_PAD_ID)
        self.acc_embed = nn.Embedding(ACC_INPUT_VOCAB, d_embed, padding_idx=ACC_PAD_ID)
        self.input_proj = nn.Linear(2 * d_embed, self.hidden_size)
        self.acc_head = nn.Linear(self.hidden_size, ACC_OUTPUT_VOCAB, bias=False)

        self.use_rtd = cfg.use_rtd
        self.rtd_weight = cfg.rtd_weight
        if self.use_rtd:
            self.rtd_head = nn.Linear(self.hidden_size, 1)

        self.use_condition_prefix = cfg.use_condition_prefix
        if self.use_condition_prefix:
            self.cond_proj = nn.Linear(cfg.cond_dim, self.hidden_size)
            # learned embedding used when the condition is dropped (CFG / unconditional)
            self.null_cond = nn.Parameter(torch.zeros(self.hidden_size))

        self._init_new_params(audio_rows)
        self.mask_sampler = SobolMaskSampler(seed=mask_seed)

    # ------------------------------------------------------------------ init
    def _init_new_params(self, audio_rows: Optional[torch.Tensor]):
        nn.init.normal_(self.vocal_embed.weight, std=0.02)
        nn.init.normal_(self.acc_embed.weight, std=0.02)
        nn.init.normal_(self.acc_head.weight, std=0.02)
        with torch.no_grad():
            self.vocal_embed.weight[VOCAL_PAD_ID].zero_()
            self.acc_embed.weight[ACC_PAD_ID].zero_()
            if audio_rows is not None:
                # Warm-start the per-track embeddings and the prediction head from
                # the backbone's pretrained <AUDIO_*> rows (requires d_embed == D).
                self.vocal_embed.weight[:NUM_AUDIO_TOKENS].copy_(audio_rows)
                self.acc_embed.weight[:NUM_AUDIO_TOKENS].copy_(audio_rows)
                self.acc_head.weight.copy_(audio_rows)
                print("[lada_band] warm-started vocal/acc embeddings + acc head from <AUDIO_*>.")
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

    # -------------------------------------------------------------- embedding
    def embed_input(self, vocal_ids: torch.Tensor, acc_ids: torch.Tensor) -> torch.Tensor:
        ev = self.vocal_embed(vocal_ids)
        ea = self.acc_embed(acc_ids)
        return self.input_proj(torch.cat([ev, ea], dim=-1))

    def _prepend_condition(self, x, pad_mask, cond, training: bool):
        """Optionally prepend a single condition prefix token along time.

        Returns ``(x, pad_mask, prefix_len)``."""
        if not self.use_condition_prefix:
            return x, pad_mask, 0
        B = x.shape[0]
        if cond is None:
            prefix = self.null_cond.to(x.dtype).expand(B, 1, -1)
        else:
            prefix = self.cond_proj(cond).unsqueeze(1)
            if training and self.cfg.cond_dropout_prob > 0:
                drop = (torch.rand(B, 1, 1, device=x.device) < self.cfg.cond_dropout_prob)
                prefix = torch.where(drop, self.null_cond.to(x.dtype).expand(B, 1, -1), prefix)
        x = torch.cat([prefix, x], dim=1)
        pad_mask = torch.cat([pad_mask.new_ones(B, 1), pad_mask], dim=1)
        return x, pad_mask, 1

    # --------------------------------------------------------------- backbone
    def _run_backbone(self, inputs_embeds: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """Bidirectional pass over the Qwen3 decoder layers (no causal mask)."""
        B, L, _ = inputs_embeds.shape
        device, dtype = inputs_embeds.device, inputs_embeds.dtype
        min_val = torch.finfo(dtype).min

        key_invalid = ~pad_mask  # (B, L) True where padding
        attn = torch.zeros(B, 1, L, L, dtype=dtype, device=device)
        attn = attn.masked_fill(key_invalid[:, None, None, :], min_val)

        position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        pos_emb = self.backbone.rotary_emb(inputs_embeds, position_ids)

        h = inputs_embeds
        for layer in self.backbone.layers:
            out = layer(
                h,
                attention_mask=attn,
                position_ids=position_ids,
                position_embeddings=pos_emb,
                use_cache=False,
            )
            h = out[0] if isinstance(out, tuple) else out
        return self.backbone.norm(h)

    def _hidden_to_logits(self, vocal_ids, acc_in_ids, pad_mask, cond, training):
        x = self.embed_input(vocal_ids, acc_in_ids)
        x, full_pad, plen = self._prepend_condition(x, pad_mask, cond, training)
        h = self._run_backbone(x, full_pad)
        if plen:
            h = h[:, plen:, :]
        return h

    # ----------------------------------------------------------------- losses
    def forward(
        self,
        vocal_ids: torch.Tensor,
        acc_ids: torch.Tensor,
        pad_mask: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Training step. ``acc_ids`` is the ground-truth accompaniment (pad =
        ACC_PAD_ID); ``vocal_ids`` pad = VOCAL_PAD_ID; ``pad_mask`` True on real
        frames."""
        B, T = acc_ids.shape
        t = self.mask_sampler.sample(B, acc_ids.device)
        masked_acc, mask_positions = forward_mask(acc_ids, pad_mask, t, ACC_MASK_ID)

        h = self._hidden_to_logits(vocal_ids, masked_acc, pad_mask, cond, training=self.training)
        logits = self.acc_head(h)  # (B, T, V)

        cml = self._cml_loss(logits, acc_ids, mask_positions, t, pad_mask)
        out = {"loss": cml, "cml_loss": cml.detach(),
               "mask_prob": t.mean().detach()}

        if self.use_rtd:
            rtd = self._rtd_loss(logits, vocal_ids, acc_ids, mask_positions, pad_mask, cond)
            out["rtd_loss"] = rtd.detach()
            out["loss"] = cml + self.rtd_weight * rtd
        return out

    @staticmethod
    def _cml_loss(logits, acc_ids, mask_positions, t, pad_mask):
        """Conditional masked modelling loss with 1/t reweighting (Eq. 3)."""
        B, T, V = logits.shape
        ce = F.cross_entropy(
            logits.reshape(-1, V), acc_ids.reshape(-1),
            reduction="none", ignore_index=ACC_PAD_ID,
        ).reshape(B, T)
        ce = ce * mask_positions.float()
        per_sample = ce.sum(dim=1)                                  # (B,)
        lengths = pad_mask.float().sum(dim=1).clamp(min=1.0)        # (B,)
        denom = (t * lengths).clamp(min=1e-4)
        return (per_sample / denom).mean()

    def _rtd_loss(self, logits, vocal_ids, acc_ids, mask_positions, pad_mask, cond):
        """Replaced-token detection (Eq. 4). The mask predictor doubles as the
        discriminator. Masked positions are filled with the model's own argmax
        prediction (detached); the discriminator must flag replaced tokens."""
        pred = logits.argmax(dim=-1).detach()
        corrupt = torch.where(mask_positions, pred, acc_ids)
        replaced = mask_positions & (pred != acc_ids)
        is_original = (~replaced).float()  # d_i = 1 if original

        h = self._hidden_to_logits(vocal_ids, corrupt, pad_mask, cond, training=self.training)
        d_logits = self.rtd_head(h).squeeze(-1)  # (B, T)
        bce = F.binary_cross_entropy_with_logits(d_logits, is_original, reduction="none")
        bce = bce * pad_mask.float()
        return bce.sum() / pad_mask.float().sum().clamp(min=1.0)

    # ------------------------------------------------------------- generation
    @torch.no_grad()
    def predict_logits(self, vocal_ids, acc_in_ids, pad_mask, cond=None) -> torch.Tensor:
        h = self._hidden_to_logits(vocal_ids, acc_in_ids, pad_mask, cond, training=False)
        return self.acc_head(h)

    @torch.no_grad()
    def predict_logits_cfg(self, vocal_ids, acc_in_ids, pad_mask, cond, guidance: float):
        """Classifier-free-guided logits (only meaningful with a condition)."""
        cond_logits = self.predict_logits(vocal_ids, acc_in_ids, pad_mask, cond=cond)
        if not self.use_condition_prefix or guidance == 1.0 or cond is None:
            return cond_logits
        uncond_logits = self.predict_logits(vocal_ids, acc_in_ids, pad_mask, cond=None)
        return uncond_logits + guidance * (cond_logits - uncond_logits)
