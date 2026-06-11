#!/usr/bin/env python3
"""Build a train/val manifest by pairing vocal and accompaniment ``.pt`` files.

Pairs are matched by file *stem* (filename without extension). For example
``vocal_dir/song123.pt`` pairs with ``acc_dir/song123.pt``.

    python scripts/build_manifest.py \
        --vocal_dir /data/vocal_mucodec \
        --acc_dir   /data/acc_mucodec \
        --out manifest.jsonl --val_out val.jsonl --val_size 500

Use ``--check_lengths`` to load each pair and drop those whose vocal/acc frame
counts differ by more than ``--max_mismatch`` (slower, but catches misaligned
data up front).
"""

import os
import sys
import json
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def stem_map(d):
    out = {}
    for f in os.listdir(d):
        if f.endswith(".pt"):
            out[os.path.splitext(f)[0]] = os.path.join(d, f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocal_dir", required=True)
    ap.add_argument("--acc_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val_out", default=None)
    ap.add_argument("--val_size", type=int, default=0)
    ap.add_argument("--cond_dir", default=None, help="optional dir of condition vectors (.pt) by stem")
    ap.add_argument("--check_lengths", action="store_true")
    ap.add_argument("--max_mismatch", type=int, default=4)
    ap.add_argument("--min_frames", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    vmap, amap = stem_map(args.vocal_dir), stem_map(args.acc_dir)
    cmap = stem_map(args.cond_dir) if args.cond_dir else {}
    common = sorted(set(vmap) & set(amap))
    print(f"vocal={len(vmap)} acc={len(amap)} paired={len(common)}")

    rows, dropped = [], 0
    if args.check_lengths:
        from lada_band.tokens import load_codec_pt
    for stem in common:
        row = {"vocal": vmap[stem], "acc": amap[stem], "name": stem}
        if stem in cmap:
            row["cond"] = cmap[stem]
        if args.check_lengths:
            try:
                nv = load_codec_pt(vmap[stem]).numel()
                na = load_codec_pt(amap[stem]).numel()
            except Exception as e:
                dropped += 1
                continue
            if min(nv, na) < args.min_frames or abs(nv - na) > args.max_mismatch:
                dropped += 1
                continue
        rows.append(row)

    random.Random(args.seed).shuffle(rows)
    val = []
    if args.val_out and args.val_size > 0:
        val, rows = rows[:args.val_size], rows[args.val_size:]

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} -> {args.out}  (dropped {dropped})")
    if val:
        with open(args.val_out, "w") as f:
            for r in val:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(val)} -> {args.val_out}")


if __name__ == "__main__":
    main()
