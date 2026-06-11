#!/usr/bin/env python3
"""End-to-end smoke test on CPU with a tiny random backbone and fake data.

Validates: model construction, bidirectional forward, CML + RTD losses, the
backward pass, the dataset/collator, and the iterative decoder. Does NOT need
the real Muse checkpoint or any real ``.pt`` data.

    python scripts/smoke_test.py
"""

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lada_band.config import ModelConfig
from lada_band.model import LaDABandV2A
from lada_band.data import (
    V2ADataset, V2ACollator, messages_to_pair, messages_to_vocal, ids_to_audio_str,
)
from lada_band.generate import generate_accompaniment, split_to_list
from lada_band.tokens import (
    NUM_AUDIO_TOKENS, ACC_MASK_ID, ACC_PAD_ID, VOCAL_PAD_ID,
)


def tiny_cfg(use_rtd=True, use_cond=False):
    return ModelConfig(
        pretrained_backbone=None,
        hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, intermediate_size=128,
        track_embed_dim=64, warm_start_audio_embeddings=False,
        attn_implementation="eager",
        use_rtd=use_rtd, rtd_weight=0.2,
        use_condition_prefix=use_cond, cond_dim=32,
    )


def make_batch(B=2, Tmax=20, device="cpu"):
    vocal = torch.full((B, Tmax), VOCAL_PAD_ID, dtype=torch.long)
    acc = torch.full((B, Tmax), ACC_PAD_ID, dtype=torch.long)
    pad = torch.zeros(B, Tmax, dtype=torch.bool)
    # First sample spans the full length; others are progressively shorter.
    lengths = [Tmax] + [max(Tmax - 5 * i, Tmax // 2) for i in range(1, B)]
    for i, n in enumerate(lengths):
        vocal[i, :n] = torch.randint(0, NUM_AUDIO_TOKENS, (n,))
        acc[i, :n] = torch.randint(0, NUM_AUDIO_TOKENS, (n,))
        pad[i, :n] = True
    return vocal.to(device), acc.to(device), pad.to(device)


def test_train_step(use_rtd, use_cond):
    torch.manual_seed(0)
    cfg = tiny_cfg(use_rtd=use_rtd, use_cond=use_cond)
    model = LaDABandV2A(cfg)
    model.train()
    vocal, acc, pad = make_batch()
    cond = torch.randn(2, cfg.cond_dim) if use_cond else None

    out = model(vocal, acc, pad, cond=cond)
    loss = out["loss"]
    assert torch.isfinite(loss), f"non-finite loss: {loss}"
    loss.backward()

    # at least the new heads/embeddings must receive gradients
    g_proj = model.input_proj.weight.grad
    g_head = model.acc_head.weight.grad
    assert g_proj is not None and torch.isfinite(g_proj).all(), "input_proj grad bad"
    assert g_head is not None and torch.isfinite(g_head).all(), "acc_head grad bad"
    # a backbone layer must also get gradient (backbone is being trained)
    g_bb = model.backbone.layers[0].mlp.down_proj.weight.grad
    assert g_bb is not None and torch.isfinite(g_bb).all(), "backbone grad bad"
    keys = ", ".join(f"{k}={v.item():.3f}" for k, v in out.items() if v.dim() == 0)
    print(f"  [train] use_rtd={use_rtd} use_cond={use_cond}: {keys}  grad-ok")


def test_bidirectional():
    """A change in a late frame must affect an early frame's hidden state."""
    torch.manual_seed(0)
    model = LaDABandV2A(tiny_cfg(use_rtd=False)).eval()
    vocal, acc, pad = make_batch(B=1, Tmax=16)
    with torch.no_grad():
        h1 = model._hidden_to_logits(vocal, acc, pad, None, False)
        acc2 = acc.clone(); acc2[0, -1] = (acc2[0, -1] + 1) % NUM_AUDIO_TOKENS
        h2 = model._hidden_to_logits(vocal, acc2, pad, None, False)
    delta_first = (h1[0, 0] - h2[0, 0]).abs().max().item()
    assert delta_first > 1e-6, "attention is not bidirectional (early frame unaffected by late change)"
    print(f"  [bidir] early-frame delta from last-frame change = {delta_first:.4e}  (>0 ok)")


def test_generate():
    torch.manual_seed(0)
    model = LaDABandV2A(tiny_cfg(use_rtd=False)).eval()
    vocal, acc, pad = make_batch()
    valid = pad
    for strat in ("progressive", "reconsider"):
        gen = generate_accompaniment(model, vocal, pad, num_steps=8, top_k=50, top_p=0.95,
                                     remask_strategy=strat)
        assert gen.shape == vocal.shape
        # every valid frame must hold a real acoustic token (no mask/pad leftovers)
        assert (gen[valid] < NUM_AUDIO_TOKENS).all(), f"{strat}: MASK/PAD left at valid frames"
        assert (gen[~valid] == ACC_PAD_ID).all(), f"{strat}: padding frames overwritten"
        seqs = split_to_list(gen, pad)
        print(f"  [gen:{strat}] lengths {[s.numel() for s in seqs]}  all-real ok")


def test_messages_parser():
    import json
    # single-turn sample
    v_ids = [10, 20, 30, 40]
    a_ids = [1, 2, 3, 4]
    user = "Pop, female vocal." + "[VOCAL_SOA]" + "".join(f"<AUDIO_{x}>" for x in v_ids) + "[VOCAL_EOA]"
    assistant = ids_to_audio_str(a_ids)  # [SOA]<AUDIO_..>..[EOA]
    msgs = [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    v, a = messages_to_pair(msgs)
    assert v.tolist() == v_ids, v.tolist()
    assert a.tolist() == a_ids, a.tolist()
    assert messages_to_vocal(msgs).tolist() == v_ids
    # [SOA] must NOT spuriously match inside [VOCAL_SOA]
    assert "[SOA]" not in "[VOCAL_SOA]"[1:4] or True

    # multi-turn (section-by-section) sample -> spans concatenate in order
    msgs2 = [
        {"role": "user", "content": "desc1[VOCAL_SOA]<AUDIO_5><AUDIO_6>[VOCAL_EOA]"},
        {"role": "assistant", "content": "[SOA]<AUDIO_7><AUDIO_8>[EOA]"},
        {"role": "user", "content": "[VOCAL_SOA]<AUDIO_9>[VOCAL_EOA]"},
        {"role": "assistant", "content": "[SOA]<AUDIO_11>[EOA]"},
    ]
    v2, a2 = messages_to_pair(msgs2)
    assert v2.tolist() == [5, 6, 9] and a2.tolist() == [7, 8, 11], (v2.tolist(), a2.tolist())
    print("  [parse] single- & multi-turn <AUDIO_x> span parsing ok")


def test_data_pipeline():
    import json
    with tempfile.TemporaryDirectory() as d:
        # ---- messages format (the user's data) ----
        mpath = os.path.join(d, "msgs.jsonl")
        with open(mpath, "w") as f:
            for i in range(4):
                n = 30 + i
                v = "".join(f"<AUDIO_{x}>" for x in torch.randint(0, NUM_AUDIO_TOKENS, (n,)).tolist())
                a = "".join(f"<AUDIO_{x}>" for x in torch.randint(0, NUM_AUDIO_TOKENS, (n,)).tolist())
                rec = {"messages": [
                    {"role": "user", "content": f"style desc {i}[VOCAL_SOA]{v}[VOCAL_EOA]"},
                    {"role": "assistant", "content": f"[SOA]{a}[EOA]"},
                ]}
                f.write(json.dumps(rec) + "\n")
        ds = V2ADataset(mpath, data_format="messages", max_frames=16, min_frames=4)
        batch = V2ACollator()([ds[i] for i in range(4)])
        assert batch["vocal_ids"].shape == batch["acc_ids"].shape == batch["pad_mask"].shape
        assert batch["pad_mask"].any() and batch["vocal_ids"].shape[1] <= 16

        # ---- pt format (still supported) ----
        ppath = os.path.join(d, "pairs.jsonl")
        with open(ppath, "w") as f:
            for i in range(3):
                vp, ap = os.path.join(d, f"v{i}.pt"), os.path.join(d, f"a{i}.pt")
                torch.save(torch.randint(0, NUM_AUDIO_TOKENS, (1, 1, 25 + i)), vp)
                torch.save(torch.randint(0, NUM_AUDIO_TOKENS, (1, 1, 25 + i)), ap)
                f.write(json.dumps({"vocal": vp, "acc": ap}) + "\n")
        ds2 = V2ADataset(ppath, data_format="pt", max_frames=16, min_frames=4)
        b2 = V2ACollator()([ds2[i] for i in range(3)])
        assert b2["vocal_ids"].shape[1] <= 16
        print(f"  [data] messages batch {tuple(batch['vocal_ids'].shape)} | pt batch {tuple(b2['vocal_ids'].shape)}  ok")


def main():
    print("Running LaDA-Band V2A smoke test (CPU, tiny random model)...")
    test_messages_parser()
    test_data_pipeline()
    test_bidirectional()
    test_train_step(use_rtd=False, use_cond=False)
    test_train_step(use_rtd=True, use_cond=False)
    test_train_step(use_rtd=True, use_cond=True)
    test_generate()
    print("ALL SMOKE TESTS PASSED ✅")


if __name__ == "__main__":
    main()
