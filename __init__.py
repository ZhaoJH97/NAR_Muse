"""LaDA-Band-style Vocal-to-Accompaniment (V2A) Non-Autoregressive model.

A discrete masked-diffusion model that generates accompaniment (BGM) MuCodec
tokens conditioned on vocal MuCodec tokens, built on top of the Muse
``qwen3-0.6B-music`` backbone.

Reference: "LaDA-Band: Language Diffusion Models for Vocal-to-Accompaniment
Generation" (arXiv:2604.11052).
"""

from .tokens import (
    NUM_AUDIO_TOKENS,
    ACC_MASK_ID,
    ACC_PAD_ID,
    VOCAL_PAD_ID,
    ACC_INPUT_VOCAB,
    VOCAL_INPUT_VOCAB,
    ACC_OUTPUT_VOCAB,
    load_codec_pt,
)

__all__ = [
    "NUM_AUDIO_TOKENS",
    "ACC_MASK_ID",
    "ACC_PAD_ID",
    "VOCAL_PAD_ID",
    "ACC_INPUT_VOCAB",
    "VOCAL_INPUT_VOCAB",
    "ACC_OUTPUT_VOCAB",
    "load_codec_pt",
]
