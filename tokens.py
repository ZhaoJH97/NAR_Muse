"""Codec token constants and ``.pt`` loading helpers.

MuCodec (as used by Muse) encodes 48 kHz stereo audio into a *single* stream of
discrete tokens at **25 Hz** with a codebook of size **16384**.  We follow the
LaDA-Band convention for the special ids:

    acc_mask  = 16384   (the [MASK] token used by the masked-diffusion process)
    acc_pad   = 16385   (padding for variable-length batching)

The vocal track is always fully observed (never masked), so it only needs a
padding id:

    vocal_pad = 16384

The accompaniment *prediction head* only ever predicts real acoustic tokens, so
its output vocabulary is exactly ``NUM_AUDIO_TOKENS`` (mask / pad are never
targets).
"""

from __future__ import annotations

import torch

# ---- MuCodec codebook ------------------------------------------------------
NUM_AUDIO_TOKENS = 16384          # codebook size (1 x 16384)
CODEC_FRAME_RATE = 25             # Hz (frames per second)

# ---- Accompaniment (target) track special ids ------------------------------
ACC_MASK_ID = NUM_AUDIO_TOKENS        # 16384
ACC_PAD_ID = NUM_AUDIO_TOKENS + 1     # 16385
ACC_INPUT_VOCAB = NUM_AUDIO_TOKENS + 2  # acoustic + mask + pad  -> 16386
ACC_OUTPUT_VOCAB = NUM_AUDIO_TOKENS     # prediction classes      -> 16384

# ---- Vocal (condition) track special ids -----------------------------------
VOCAL_PAD_ID = NUM_AUDIO_TOKENS       # 16384
VOCAL_INPUT_VOCAB = NUM_AUDIO_TOKENS + 1  # acoustic + pad -> 16385


def load_codec_pt(path: str) -> torch.Tensor:
    """Load a MuCodec ``.pt`` file and return a 1-D ``LongTensor`` of token ids.

    ``train/encode_audio.py`` in Muse saves the output of ``MuCodec.sound2code``
    which has shape ``[1, 1, T]``.  To be robust we accept any of:
    ``[T]``, ``[1, T]``, ``[1, 1, T]``, ``[B, 1, T]`` (first item taken), or a
    python ``list``.  Any extra leading singleton/batch dims are squeezed and we
    keep the last (time) axis.
    """
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, (list, tuple)):
        obj = torch.as_tensor(obj)
    if not isinstance(obj, torch.Tensor):
        raise TypeError(f"{path}: expected a tensor/list, got {type(obj)}")

    t = obj
    # Collapse everything but the final time axis. The codec stream is 1 codebook
    # so all non-time dims are singletons (or a batch we index into).
    while t.dim() > 1:
        if t.shape[0] != 1:
            # batched dump -> take the first item
            t = t[0]
        else:
            t = t.squeeze(0)
    t = t.reshape(-1).to(torch.long)

    if t.numel() == 0:
        raise ValueError(f"{path}: empty codec sequence")
    _min, _max = int(t.min()), int(t.max())
    if _min < 0 or _max >= NUM_AUDIO_TOKENS:
        raise ValueError(
            f"{path}: token id out of range [0, {NUM_AUDIO_TOKENS - 1}], "
            f"got [{_min}, {_max}]. Is this really a MuCodec dump?"
        )
    return t
