#!/usr/bin/env python3
"""Distributed training for the LaDA-Band V2A masked-diffusion model.

Single GPU:
    python scripts/train.py --config configs/base_0.6b.yaml

Multi-GPU (8x):
    torchrun --nproc_per_node=8 scripts/train.py --config configs/base_0.6b.yaml

Any ``train.*`` / ``data.*`` / ``model.*`` field can be overridden from the CLI,
e.g. ``--train.learning_rate 2e-5 --model.use_rtd false``.
"""

import os
import sys
import math
import time
import json
import argparse
import shutil

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lada_band.config import Config
from lada_band.model import LaDABandV2A
from lada_band.data import V2ADataset, V2ACollator


# --------------------------------------------------------------------------- ddp
def ddp_setup():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return True, int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), local_rank
    return False, 0, 1, 0


def is_main(rank):
    return rank == 0


# ----------------------------------------------------------------------- config
def parse_overrides(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args, extra = ap.parse_known_args(argv)
    cfg = Config.from_yaml(args.config)

    # apply --section.key value overrides
    def cast(old, val):
        if isinstance(old, bool):
            return str(val).lower() in ("1", "true", "yes")
        if old is None:
            return val
        return type(old)(val)

    i = 0
    while i < len(extra):
        key = extra[i].lstrip("-")
        val = extra[i + 1]
        sec, field = key.split(".", 1)
        obj = getattr(cfg, sec)
        setattr(obj, field, cast(getattr(obj, field), val))
        i += 2
    return cfg, args.config


# -------------------------------------------------------------------- scheduler
def build_lr_lambda(tcfg):
    warmup, total = tcfg.warmup_steps, tcfg.max_steps
    min_ratio = tcfg.min_lr_ratio

    def fn(step):
        if step < warmup:
            return step / max(1, warmup)
        if tcfg.lr_scheduler == "constant":
            return 1.0
        progress = (step - warmup) / max(1, total - warmup)
        cos = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_ratio + (1.0 - min_ratio) * cos

    return fn


# ----------------------------------------------------------------- checkpoint io
def save_checkpoint(path, model, optimizer, scheduler, step, cfg):
    os.makedirs(path, exist_ok=True)
    raw = model.module if isinstance(model, DDP) else model
    torch.save(raw.state_dict(), os.path.join(path, "model.pt"))
    torch.save(
        {"optimizer": optimizer.state_dict(),
         "scheduler": scheduler.state_dict(),
         "step": step},
        os.path.join(path, "trainer.pt"),
    )
    cfg.save_yaml(os.path.join(path, "config.yaml"))


def prune_checkpoints(output_dir, keep):
    cks = sorted(
        [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
        key=lambda d: int(d.split("-")[1]),
    )
    for d in cks[:-keep] if keep > 0 else []:
        shutil.rmtree(os.path.join(output_dir, d), ignore_errors=True)


# ------------------------------------------------------------------------- main
def main():
    cfg, cfg_path = parse_overrides(sys.argv[1:])
    use_ddp, rank, world_size, local_rank = ddp_setup()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.train.seed + rank)

    if is_main(rank):
        os.makedirs(cfg.train.output_dir, exist_ok=True)
        cfg.save_yaml(os.path.join(cfg.train.output_dir, "config.yaml"))
        print(f"[config] {json.dumps(cfg.to_dict(), indent=2)}")

    # ---- model
    model = LaDABandV2A(cfg.model, mask_seed=cfg.train.seed + rank).to(device)
    if cfg.train.resume_from:
        sd = torch.load(os.path.join(cfg.train.resume_from, "model.pt"), map_location="cpu")
        model.load_state_dict(sd)
        if is_main(rank):
            print(f"[resume] loaded weights from {cfg.train.resume_from}")
    if use_ddp:
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None,
                    find_unused_parameters=cfg.model.use_condition_prefix)

    # ---- data
    train_ds = V2ADataset(cfg.data.train_manifest, data_format=cfg.data.format,
                          max_frames=cfg.data.max_frames, min_frames=cfg.data.min_frames,
                          max_len_mismatch=cfg.data.max_len_mismatch, random_crop=True)
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
    loader = DataLoader(
        train_ds, batch_size=cfg.train.per_device_batch_size, sampler=sampler,
        shuffle=(sampler is None), num_workers=cfg.data.num_workers, drop_last=True,
        collate_fn=V2ACollator(), pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.data.num_workers > 0,
    )

    # ---- optim / sched
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate,
                                  weight_decay=cfg.train.weight_decay, eps=cfg.train.adam_epsilon)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, build_lr_lambda(cfg.train))
    start_step = 0
    if cfg.train.resume_from and os.path.exists(os.path.join(cfg.train.resume_from, "trainer.pt")):
        tr = torch.load(os.path.join(cfg.train.resume_from, "trainer.pt"), map_location="cpu")
        optimizer.load_state_dict(tr["optimizer"]); scheduler.load_state_dict(tr["scheduler"])
        start_step = tr["step"]

    amp_dtype = torch.bfloat16 if cfg.train.bf16 and torch.cuda.is_available() else torch.float32
    use_amp = amp_dtype == torch.bfloat16

    # ---- loop
    model.train()
    step = start_step
    accum = cfg.train.gradient_accumulation_steps
    micro = 0
    optimizer.zero_grad(set_to_none=True)
    running = {}
    t0 = time.time()
    epoch = 0

    def batches():
        nonlocal epoch
        while True:
            if sampler is not None:
                sampler.set_epoch(epoch)
            for b in loader:
                yield b
            epoch += 1

    for batch in batches():
        if step >= cfg.train.max_steps:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        cond = batch.get("cond")

        sync = (micro + 1) % accum == 0
        ctx = model.no_sync() if (use_ddp and not sync) else _nullcontext()
        with ctx:
            with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu",
                                dtype=amp_dtype, enabled=use_amp):
                out = model(batch["vocal_ids"], batch["acc_ids"], batch["pad_mask"], cond=cond)
            loss = out["loss"] / accum
            loss.backward()

        for k, v in out.items():
            if v.dim() == 0:
                running[k] = running.get(k, 0.0) + float(v) / accum
        micro += 1

        if sync:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            step += 1

            if is_main(rank) and step % cfg.train.log_every == 0:
                dt = time.time() - t0
                lr = scheduler.get_last_lr()[0]
                msg = " ".join(f"{k}={v / cfg.train.log_every:.4f}" for k, v in running.items())
                print(f"step {step}/{cfg.train.max_steps} | {msg} | lr={lr:.2e} "
                      f"| gnorm={float(grad_norm):.2f} | {cfg.train.log_every / dt:.2f} it/s")
                running = {}; t0 = time.time()

            if is_main(rank) and step % cfg.train.save_every == 0:
                ckpt = os.path.join(cfg.train.output_dir, f"checkpoint-{step}")
                save_checkpoint(ckpt, model, optimizer, scheduler, step, cfg)
                prune_checkpoints(cfg.train.output_dir, cfg.train.save_total_limit)
                print(f"[ckpt] saved {ckpt}")
            micro = 0

    if is_main(rank):
        ckpt = os.path.join(cfg.train.output_dir, f"checkpoint-{step}")
        save_checkpoint(ckpt, model, optimizer, scheduler, step, cfg)
        print(f"[done] final checkpoint {ckpt}")
    if use_ddp:
        dist.destroy_process_group()


class _nullcontext:
    def __enter__(self):
        return None
    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    main()
