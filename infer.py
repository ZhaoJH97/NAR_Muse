#!/usr/bin/env python3
"""Generate accompaniment MuCodec tokens from vocal MuCodec tokens.

    python scripts/infer.py \
        --ckpt runs/lada_band_0.6b/checkpoint-150000 \
        --manifest infer_list.jsonl \
        --output_dir outputs \
        --num_steps 20

``manifest`` lines: ``{"vocal": "/path/vocal.pt", "name": "song1", "cond": "..."?}``.
Each output ``<name>.pt`` is saved with shape ``[1, 1, T]`` so it can be decoded
straight away by Muse's ``infer/decode_audio.py``.
"""

import os
import sys
import json
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lada_band.config import Config, ModelConfig
from lada_band.model import LaDABandV2A
from lada_band.data import read_manifest, messages_to_vocal, ids_to_audio_str
from lada_band.tokens import load_codec_pt
from lada_band.generate import generate_accompaniment


def load_model(ckpt_dir, device, attn_impl=None):
    cfg = Config.from_yaml(os.path.join(ckpt_dir, "config.yaml"))
    mcfg: ModelConfig = cfg.model
    # Backbone weights live in model.pt, so do not re-download the pretrained one.
    mcfg.pretrained_backbone = None
    mcfg.warm_start_audio_embeddings = False
    if attn_impl:
        mcfg.attn_implementation = attn_impl
    model = LaDABandV2A(mcfg).to(device)
    sd = torch.load(os.path.join(ckpt_dir, "model.pt"), map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[warn] missing keys: {missing[:4]}{' ...' if len(missing) > 4 else ''}")
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected[:4]}{' ...' if len(unexpected) > 4 else ''}")
    model.eval()
    return model, cfg


def chunk_indices(n, chunk, overlap=0):
    if chunk <= 0 or n <= chunk:
        return [(0, n)]
    step = max(1, chunk - overlap)
    out = []
    s = 0
    while s < n:
        out.append((s, min(s + chunk, n)))
        if s + chunk >= n:
            break
        s += step
    return out


@torch.no_grad()
def generate_one(model, vocal, device, args, cond=None):
    """Generate accompaniment for a single 1-D vocal sequence (whole song or
    stitched from chunks)."""
    pieces = []
    for (s, e) in chunk_indices(vocal.numel(), args.chunk_frames, args.chunk_overlap):
        v = vocal[s:e].unsqueeze(0).to(device)
        pad = torch.ones_like(v, dtype=torch.bool)
        gen = generate_accompaniment(
            model, v, pad, num_steps=args.num_steps, temperature=args.temperature,
            top_k=args.top_k, top_p=args.top_p, mask_temp=args.mask_temp,
            cond=cond, guidance=args.guidance, remask_strategy=args.remask_strategy,
        )[0].cpu()
        # On overlapping chunks keep only the newly-advanced part to avoid double counting.
        if pieces and args.chunk_overlap > 0:
            gen = gen[args.chunk_overlap:]
        pieces.append(gen)
    return torch.cat(pieces)[: vocal.numel()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_steps", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=100)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--mask_temp", type=float, default=10.5)
    ap.add_argument("--remask_strategy", choices=["progressive", "reconsider"], default="progressive",
                    help="MaskGIT remasking: 'progressive' locks revealed tokens; "
                         "'reconsider' lets revealed low-confidence tokens be remasked")
    ap.add_argument("--guidance", type=float, default=1.0, help="CFG scale (needs condition)")
    ap.add_argument("--chunk_frames", type=int, default=0, help="0 = whole song; else window size")
    ap.add_argument("--chunk_overlap", type=int, default=0)
    ap.add_argument("--data_format", choices=["messages", "pt"], default="messages")
    ap.add_argument("--out_jsonl", type=str, default=None,
                    help="also write a messages-format JSONL with generated [SOA]..[EOA] tokens")
    ap.add_argument("--attn_impl", type=str, default="sdpa")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    model, cfg = load_model(args.ckpt, device, attn_impl=args.attn_impl)

    items = read_manifest(args.manifest)
    jsonl_out = open(args.out_jsonl, "w", encoding="utf-8") if args.out_jsonl else None
    for i, item in enumerate(items):
        if args.data_format == "messages":
            name = item.get("name") or item.get("id") or f"sample_{i}"
            vocal = messages_to_vocal(item["messages"])
        else:
            name = item.get("name") or os.path.splitext(os.path.basename(item["vocal"]))[0]
            vocal = load_codec_pt(item["vocal"])
        if vocal.numel() == 0:
            print(f"[skip] {name}: no vocal tokens")
            continue

        out_path = os.path.join(args.output_dir, f"{name}.pt")
        if os.path.exists(out_path):
            print(f"[skip] {out_path} exists")
            continue

        acc = generate_one(model, vocal, device, args, cond=None)
        # save in MuCodec [1, 1, T] layout for decode_audio.py
        torch.save(acc.reshape(1, 1, -1), out_path)
        print(f"[{i + 1}/{len(items)}] {name}: {acc.numel()} frames -> {out_path}")

        if jsonl_out is not None:
            rec = {"name": name,
                   "messages": [{"role": "assistant", "content": ids_to_audio_str(acc.tolist())}]}
            jsonl_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            jsonl_out.flush()
    if jsonl_out is not None:
        jsonl_out.close()


if __name__ == "__main__":
    main()
