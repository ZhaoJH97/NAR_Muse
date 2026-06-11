"""Configuration dataclasses for the LaDA-Band V2A model.

A single YAML file holds three sections: ``model``, ``data`` and ``train``.
See ``configs/base_0.6b.yaml`` for a documented example.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, asdict
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    # Path to a HF Qwen3 checkpoint to initialise the transformer backbone from
    # (the Muse ``qwen3-0.6B-music`` model). If null, a fresh backbone is built
    # from ``hf_config_name`` / the backbone dims below (random init).
    pretrained_backbone: Optional[str] = None
    # Fallback config when ``pretrained_backbone`` is null (used for tests too).
    hidden_size: int = 1024
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 3072
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 40_960

    # Per-track embedding dim. When equal to ``hidden_size`` and a pretrained
    # backbone is given, the per-track embeddings + accompaniment head are
    # warm-started from the backbone's pretrained ``<AUDIO_*>`` rows.
    track_embed_dim: int = 1024
    warm_start_audio_embeddings: bool = True
    # If True, abort when the audio tokens cannot be resolved for warm-starting
    # (instead of silently falling back to random embeddings).
    warm_start_strict: bool = False
    # Token string of the i-th audio token in the backbone tokenizer, used to
    # locate the pretrained audio embedding rows for warm-starting.
    audio_token_template: str = "<AUDIO_{}>"

    attn_implementation: str = "sdpa"  # "sdpa" | "eager" | "flash_attention_2"

    # --- Auxiliary Replaced-Token-Detection objective (ELECTRA-style) ---
    use_rtd: bool = True
    rtd_weight: float = 0.2            # lambda in Eq. (5)

    # --- Optional global condition prefix (CLaMP3 style). Off by default: the
    # core task is vocal-only zero-shot V2A. When enabled, a pre-extracted
    # condition vector of ``cond_dim`` is linearly projected to one prefix token.
    use_condition_prefix: bool = False
    cond_dim: int = 768
    cond_dropout_prob: float = 0.5     # classifier-free guidance dropout


@dataclass
class DataConfig:
    # "messages": Muse-style chat JSONL with [VOCAL_SOA]..[VOCAL_EOA] / [SOA]..[EOA]
    #             spans of <AUDIO_x> tokens (the text prompt is ignored).
    # "pt":       manifest of {"vocal": "...pt", "acc": "...pt"} pairs.
    format: str = "messages"
    train_manifest: str = ""
    val_manifest: Optional[str] = None
    # Random-crop length (in codec frames). 25 Hz -> 1024 frames ~= 40.96 s.
    max_frames: int = 1024
    min_frames: int = 16
    # Drop pairs whose vocal/acc lengths differ by more than this many frames.
    max_len_mismatch: int = 4
    num_workers: int = 8


@dataclass
class TrainConfig:
    output_dir: str = "runs/lada_band_0.6b"
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_steps: int = 10_000
    max_steps: int = 150_000
    lr_scheduler: str = "cosine"       # "cosine" | "constant"
    min_lr_ratio: float = 0.1          # cosine decays to ratio * peak lr
    bf16: bool = True
    seed: int = 42
    log_every: int = 20
    save_every: int = 2_000
    eval_every: int = 2_000
    save_total_limit: int = 10
    resume_from: Optional[str] = None


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @staticmethod
    def from_yaml(path: str) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return Config(
            model=ModelConfig(**(raw.get("model") or {})),
            data=DataConfig(**(raw.get("data") or {})),
            train=TrainConfig(**(raw.get("train") or {})),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def save_yaml(self, path: str) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)
