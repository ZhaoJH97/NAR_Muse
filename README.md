# Muse-NAR: Vocal-to-Accompaniment via Discrete Masked Diffusion

A **non-autoregressive (NAR)** vocal-to-accompaniment (V2A) model built on the
[Muse](https://github.com/yuhui1038/Muse) `qwen3-0.6B-music` backbone. It takes
**vocal MuCodec tokens** + **masked accompaniment (BGM) MuCodec tokens** and
predicts the **full BGM token sequence** by iterative masked-diffusion denoising.

The design follows **LaDA-Band: Language Diffusion Models for Vocal-to-Accompaniment
Generation** ([arXiv:2604.11052](https://arxiv.org/pdf/2604.11052)), re-targeted from
its LLaMA-3.2-1B backbone to the Muse Qwen3-0.6B music backbone.

---

## How it works

```
 vocal tokens  v_1..v_T  ──[vocal_embed]──┐
                                          ├─ concat(feature dim) ─[Linear]→ X ∈ R^{T×D}
 masked acc    ã_1..ã_T  ──[acc_embed]────┘
                                          │
        (optional global condition prefix)│  prepend along time
                                          ▼
                  ┌─────────────────────────────────────────┐
                  │  Qwen3-0.6B backbone, BIDIRECTIONAL attn │   (no causal mask)
                  └─────────────────────────────────────────┘
                                          │
                       hidden ∈ R^{T×D} ──[acc_head]→ logits ∈ R^{T×16384}
```

* **Dual-track input (Eq. 1–2).** Each track's discrete tokens are embedded and
  concatenated **along the feature dimension per frame**, then linearly projected
  to the backbone hidden size `D=1024`. This is exactly the requested
  *"逐帧 embedding concat 后 linear 到 LLM 的 embedding size"*.
* **Bidirectional backbone.** The pretrained Qwen3 layers are reused, but the
  causal mask is removed so every accompaniment frame attends to the whole vocal
  track and the whole (partially observed) accompaniment — full-sequence context.
* **Training = conditional masked modelling (Eq. 3).** Each acc token is masked
  independently with probability `t` drawn from a cosine schedule over a Sobol
  progress variable; the model reconstructs the originals, weighted by `1/t`.
* **Auxiliary RTD (Eq. 4–5).** An ELECTRA-style replaced-token-detection head adds
  dense supervision (helps weakly-anchored intros/interludes). `λ=0.2`.
* **Inference = iterative denoising (Sec. 4.3).** Start fully masked; over
  `num_steps` cosine steps predict all masked frames, keep the highest-confidence
  ones and re-mask the lowest-confidence fraction.

### Token conventions (MuCodec, 25 Hz, codebook `1×16384`)

| id range            | meaning                                   |
|---------------------|-------------------------------------------|
| `0 .. 16383`        | acoustic tokens (both tracks)             |
| `acc_mask  = 16384` | `[MASK]` for the accompaniment track      |
| `acc_pad   = 16385` | accompaniment padding                     |
| `vocal_pad = 16384` | vocal padding (vocal is never masked)     |

The accompaniment **prediction head** has exactly `16384` classes (mask/pad are
never targets). These match LaDA-Band Sec. 5.2 (`acc_mask=16384`, `acc_pad=16385`).

---

## Installation

```bash
pip install -r requirements.txt   # torch, transformers==4.57.3, pyyaml, numpy, tqdm
```

Download the Muse base model (used to initialise the backbone):

```bash
huggingface-cli download bolshyC/qwen3-0.6B-music --local-dir ./qwen3-0.6B-music
```

Audio ⇄ token conversion (MuCodec) uses the **separate** Muse environment — see
`Muse/requirements_mucodec.txt` and `Muse/train/encode_audio.py` /
`Muse/infer/decode_audio.py`.

---

## 1. Prepare data

**Primary format — `messages` JSONL (your existing train/test data).** Point
`data.train_manifest` straight at your Muse-style chat JSONL. Each line:

```json
{"messages": [
  {"role": "user",      "content": "Pop, female vocal, ...[VOCAL_SOA]<AUDIO_12><AUDIO_77>...[VOCAL_EOA]"},
  {"role": "assistant", "content": "[SOA]<AUDIO_5><AUDIO_91>...[EOA]"}
]}
```

* **Vocal** tokens are read from every `[VOCAL_SOA]..[VOCAL_EOA]` span in *user*
  turns; **accompaniment (target)** tokens from every `[SOA]..[EOA]` span in
  *assistant* turns. `<AUDIO_x>` → integer id `x` (0–16383).
* Multiple turns/spans are concatenated **in order**, so section-by-section
  songs reconstruct the full vocal/accompaniment streams.
* The leading **text prompt is ignored** (vocal-only V2A, per your spec).
* Large files are **not loaded into RAM** — line byte-offsets are indexed and
  each line is parsed on demand.

No conversion step is needed; just set `data.format: messages` (the default).

**Alternative — paired `.pt` files.** If instead you have MuCodec `.pt` dumps
(`[1,1,T]`, single codebook, 25 Hz), pair them by filename stem and set
`data.format: pt`:

```bash
python scripts/build_manifest.py \
    --vocal_dir /data/vocal_mucodec --acc_dir /data/acc_mucodec \
    --out data/train.jsonl --val_out data/val.jsonl --val_size 1000 --check_lengths
```

> V2A assumes the vocal and accompaniment of a song are **frame-aligned**. The
> loader truncates to the shorter of the two streams.

## 2. Train

```bash
# 8 GPUs (Stage 1, short-form 40.96 s = 1024 frames)
NGPU=8 bash train.sh

# single GPU
python scripts/train.py --config configs/base_0.6b.yaml

# any field can be overridden:
python scripts/train.py --config configs/base_0.6b.yaml \
    --train.learning_rate 2e-5 --model.use_rtd false --data.max_frames 1024
```

**Two-stage curriculum (paper Sec. 4.4).** Stage 1 on short clips, then Stage 2
fine-tunes on full-length audio:

```bash
NGPU=8 bash train.sh \
    --data.max_frames 4096 \
    --train.max_steps 40000 \
    --train.resume_from runs/lada_band_0.6b/checkpoint-150000
```

## 3. Generate & decode to audio

```bash
# reads vocal tokens from your messages JSONL -> accompaniment .pt (+ optional jsonl)
python scripts/infer.py \
    --ckpt runs/lada_band_0.6b/checkpoint-150000 \
    --manifest data/test.jsonl --data_format messages \
    --output_dir outputs \
    --out_jsonl outputs/generated.jsonl \
    --num_steps 20 --top_k 100 --top_p 0.9 --temperature 1.0 --mask_temp 10.5 \
    --remask_strategy progressive   # MaskGIT remasking: progressive (default) | reconsider
```

Decoding is **MaskGIT-style confidence-based parallel decoding**: each step runs one
parallel forward over all masked positions, keeps the highest-confidence predictions,
and re-masks the lowest-confidence fraction under a cosine schedule until convergence
(`lada_band/generate.py`). `progressive` locks revealed tokens (canonical MaskGIT);
`reconsider` lets already-revealed low-confidence tokens be re-masked and re-sampled.

The vocal is taken from the `[VOCAL_SOA]..[VOCAL_EOA]` spans of each test line
(the assistant/ground-truth content, if present, is ignored). `--out_jsonl`
additionally writes the generated `[SOA]<AUDIO_x>..[EOA]` per sample for your
eval pipeline. Outputs are also saved as `[1,1,T]` tensors — feed straight into MuCodec:

```bash
# in the Muse / MuCodec environment, point decode_audio.py at outputs/<name>.pt
python Muse/infer/decode_audio.py
```

---

## Configuration reference

See [`configs/base_0.6b.yaml`](configs/base_0.6b.yaml). Key knobs:

| field | default | notes |
|-------|---------|-------|
| `model.pretrained_backbone` | `./qwen3-0.6B-music` | `null` ⇒ train backbone from scratch |
| `model.warm_start_audio_embeddings` | `true` | copy pretrained `<AUDIO_*>` rows into both track embeddings + acc head |
| `model.use_rtd` / `rtd_weight` | `true` / `0.2` | auxiliary replaced-token detection (doubles fwd cost) |
| `model.use_condition_prefix` | `false` | enable a global style/ref-audio prefix token (CLaMP3-style) + CFG |
| `data.max_frames` | `1024` | 25 Hz × 40.96 s; use `4096` for Stage 2 |
| `train.*` | — | AdamW, lr 1e-5, cosine + 10k warmup, bf16, grad-clip 1.0 (paper Sec. 5.2) |

---

## Design decisions & assumptions

1. **Vocal-only.** The text prompt in the user message is parsed out and **ignored**
   — the model conditions purely on the vocal track (your spec). An optional global
   condition prefix (CLaMP3-style vector + classifier-free guidance) remains wired
   behind `model.use_condition_prefix: true` if you ever want style control.
2. **Backbone warm-start.** With a pretrained backbone and `track_embed_dim == D`,
   the per-track embeddings and the accompaniment head are initialised from the
   Muse model's pretrained `<AUDIO_*>` rows, then fine-tuned bidirectionally.
3. **Token format.** Codec tokens are `<AUDIO_x>` (MuCodec single codebook
   `1×16384`, 25 Hz), the same vocab for both tracks. The `pt` loader also accepts
   `[T]`/`[1,T]`/`[1,1,T]` dumps. All ids are validated to lie in `[0, 16383]`.
4. **No ms-swift.** The NAR objective (bidirectional attention, masked diffusion,
   RTD) does not fit ms-swift's causal-SFT paradigm, so training is a standalone
   `torchrun` DDP loop.

## Repository layout

```
lada_band/
  tokens.py     constants + .pt loader
  config.py     YAML-backed dataclasses
  masking.py    cosine/Sobol mask schedule + forward masking
  sampling.py   top-k/top-p, gumbel, confidence
  model.py      LaDABandV2A: dual-track embed, bidirectional Qwen3, CML + RTD
  data.py       messages-jsonl + .pt datasets, <AUDIO_x> parser, padding collator
  generate.py   iterative masked-diffusion decoder
scripts/
  build_manifest.py   pair vocal/acc .pt by stem (only for data.format=pt)
  train.py            DDP training loop
  infer.py            messages/.pt vocal -> accompaniment .pt (+ optional jsonl)
  smoke_test.py       CPU end-to-end self-test (no real model/data needed)
configs/base_0.6b.yaml
train.sh
```

Run the self-test any time (no GPU / checkpoint / data needed):

```bash
python scripts/smoke_test.py
```

## Reference

```bibtex
@article{wang2026ladaband,
  title={LaDA-Band: Language Diffusion Models for Vocal-to-Accompaniment Generation},
  author={Wang, Qi and Shen, Zhexu and Chen, Meng and Yu, Guoxin and Pang, Chaoxu and Zhao, Weifeng and Zhou, Wenjiang},
  journal={arXiv preprint arXiv:2604.11052},
  year={2026}
}
```
import json, random, torch
from lada_band.data import messages_to_pair
lines = [json.loads(l) for l in open("你的train.jsonl")]
random.shuffle(lines)
for rec in lines[:20]:
    v, a = messages_to_pair(rec["messages"])
    n = min(len(v), len(a))
    eq = (v[:n] == a[:n]).float().mean().item()          # 逐帧相同比例
    print(f"len v={len(v)} a={len(a)} | frame-equal={eq:.3f} | v[:8]={v[:8].tolist()} a[:8]={a[:8].tolist()}")
