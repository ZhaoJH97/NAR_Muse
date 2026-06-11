#!/usr/bin/env bash
# Launch LaDA-Band V2A training. Mirrors the paper's 8-GPU setup.
set -e

export NCCL_DEBUG=WARN
NGPU=${NGPU:-8}
CONFIG=${CONFIG:-configs/base_0.6b.yaml}

# Stage 1 (short-form, max_frames=1024). For Stage 2 long-form, override:
#   CONFIG=configs/base_0.6b.yaml NGPU=8 bash train.sh \
#       --data.max_frames 4096 --train.learning_rate 1e-5 \
#       --train.max_steps 40000 --train.resume_from runs/lada_band_0.6b/checkpoint-150000

torchrun --nproc_per_node="${NGPU}" --master_port="${MASTER_PORT:-29500}" \
    scripts/train.py --config "${CONFIG}" "$@"
