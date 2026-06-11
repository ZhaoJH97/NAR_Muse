"""Datasets and collator for the V2A model.

Two input formats are supported (``data.format`` in the config):

* ``"messages"`` — a Muse-style chat JSONL. Each line is
  ``{"messages": [{"role": "user", "content": "<text>[VOCAL_SOA]<AUDIO_..>..[VOCAL_EOA]"},
                  {"role": "assistant", "content": "[SOA]<AUDIO_..>..[EOA]"}]}``.
  Vocal tokens are read from ``[VOCAL_SOA]..[VOCAL_EOA]`` spans in *user* turns;
  accompaniment (target) tokens from ``[SOA]..[EOA]`` spans in *assistant* turns.
  Multiple turns/spans are concatenated in order (section-by-section songs). The
  leading text prompt is ignored.

* ``"pt"`` — a manifest whose lines are
  ``{"vocal": "/path/vocal.pt", "acc": "/path/acc.pt"}`` (see ``build_manifest.py``).

Large ``messages`` JSONLs are not loaded into RAM: we index line byte-offsets
once and ``seek``/parse each line on demand.
"""

from __future__ import annotations

import re
import json
import random
from typing import List, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .tokens import load_codec_pt, NUM_AUDIO_TOKENS, VOCAL_PAD_ID, ACC_PAD_ID


# --------------------------------------------------------------------- parsing
_AUDIO_RE = re.compile(r"<AUDIO_(\d+)>")
_VOCAL_SPAN_RE = re.compile(r"\[VOCAL_SOA\](.*?)\[VOCAL_EOA\]", re.DOTALL)
_ACC_SPAN_RE = re.compile(r"\[SOA\](.*?)\[EOA\]", re.DOTALL)


def _ids_from_spans(text: str, span_re: re.Pattern) -> List[int]:
    out: List[int] = []
    for m in span_re.finditer(text):
        out.extend(int(x) for x in _AUDIO_RE.findall(m.group(1)))
    return out


def messages_to_vocal(messages: List[dict]) -> torch.Tensor:
    """Concatenate all vocal tokens from ``[VOCAL_SOA]..[VOCAL_EOA]`` user spans."""
    ids: List[int] = []
    for msg in messages:
        if msg.get("role") == "user":
            ids.extend(_ids_from_spans(msg.get("content") or "", _VOCAL_SPAN_RE))
    return torch.tensor(ids, dtype=torch.long)


def messages_to_pair(messages: List[dict]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(vocal_ids, acc_ids)`` parsed from a messages list. Text ignored."""
    vocal: List[int] = []
    acc: List[int] = []
    for msg in messages:
        role, content = msg.get("role"), (msg.get("content") or "")
        if role == "user":
            vocal.extend(_ids_from_spans(content, _VOCAL_SPAN_RE))
        elif role == "assistant":
            acc.extend(_ids_from_spans(content, _ACC_SPAN_RE))
    return torch.tensor(vocal, dtype=torch.long), torch.tensor(acc, dtype=torch.long)


def ids_to_audio_str(ids, soa: str = "[SOA]", eoa: str = "[EOA]") -> str:
    """Inverse of the parser: wrap a token id sequence as ``[SOA]<AUDIO_..>..[EOA]``."""
    body = "".join(f"<AUDIO_{int(x)}>" for x in ids)
    return f"{soa}{body}{eoa}"


def _validate_range(t: torch.Tensor, what: str):
    if t.numel() == 0:
        raise ValueError(f"empty {what} sequence")
    if int(t.min()) < 0 or int(t.max()) >= NUM_AUDIO_TOKENS:
        raise ValueError(f"{what} token id out of range [0,{NUM_AUDIO_TOKENS - 1}]")


# ------------------------------------------------------------------ jsonl index
class JsonlLineIndex:
    """Byte-offset index over a JSONL file for memory-light random access."""

    def __init__(self, path: str):
        self.path = path
        self.offsets: List[int] = []
        with open(path, "rb") as f:
            off = f.tell()
            line = f.readline()
            while line:
                if line.strip():
                    self.offsets.append(off)
                off = f.tell()
                line = f.readline()
        self._fh = None

    def __len__(self):
        return len(self.offsets)

    def get(self, idx: int) -> dict:
        if self._fh is None:
            self._fh = open(self.path, "rb")
        self._fh.seek(self.offsets[idx])
        return json.loads(self._fh.readline().decode("utf-8"))

    def __getstate__(self):
        s = self.__dict__.copy()
        s["_fh"] = None  # never pickle an open file handle (DataLoader workers)
        return s


def read_manifest(path: str) -> List[dict]:
    """Eagerly read a (small) JSONL into a list. Used by infer/build utilities."""
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ---------------------------------------------------------------------- dataset
class V2ADataset(Dataset):
    def __init__(
        self,
        manifest: str,
        data_format: str = "messages",
        max_frames: int = 1024,
        min_frames: int = 16,
        max_len_mismatch: int = 4,
        random_crop: bool = True,
    ):
        assert data_format in ("messages", "pt"), data_format
        self.data_format = data_format
        self.index = JsonlLineIndex(manifest)
        self.max_frames = max_frames
        self.min_frames = min_frames
        self.max_len_mismatch = max_len_mismatch
        self.random_crop = random_crop

    def __len__(self):
        return len(self.index)

    def _load_pair(self, item: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.data_format == "messages":
            vocal, acc = messages_to_pair(item["messages"])
        else:
            vocal, acc = load_codec_pt(item["vocal"]), load_codec_pt(item["acc"])
        _validate_range(vocal, "vocal")
        _validate_range(acc, "acc")
        n = min(vocal.numel(), acc.numel())
        return vocal[:n], acc[:n]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            vocal, acc = self._load_pair(self.index.get(idx))
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

        n = vocal.numel()
        if n < self.min_frames:
            return self.__getitem__((idx + 1) % len(self))
        if n > self.max_frames:
            start = random.randint(0, n - self.max_frames) if self.random_crop else 0
            vocal = vocal[start:start + self.max_frames]
            acc = acc[start:start + self.max_frames]
        return {"vocal": vocal, "acc": acc}


class V2ACollator:
    """Right-pads a batch to the longest sequence and builds the pad mask."""

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch = [b for b in batch if b is not None]
        B = len(batch)
        T = max(b["vocal"].numel() for b in batch)

        vocal_ids = torch.full((B, T), VOCAL_PAD_ID, dtype=torch.long)
        acc_ids = torch.full((B, T), ACC_PAD_ID, dtype=torch.long)
        pad_mask = torch.zeros(B, T, dtype=torch.bool)
        for i, b in enumerate(batch):
            n = b["vocal"].numel()
            vocal_ids[i, :n] = b["vocal"]
            acc_ids[i, :n] = b["acc"]
            pad_mask[i, :n] = True
        return {"vocal_ids": vocal_ids, "acc_ids": acc_ids, "pad_mask": pad_mask}
