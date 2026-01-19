#!/bin/bash
export TORCHINDUCTOR_AUTOTUNE_ATEN=0
export TORCHINDUCTOR_AUTOTUNE_TRITON=0

torchrun \
    --standalone \
    --nproc_per_node 8 \
    ./scripts/train_eagle3_offline.py \
    --target-model-path /models/gpt-oss-120b \
    --draft-model-config /models/gpt-oss-120b-Eagle3/config.json \
    --train-hidden-states-path /cache/vllm_tide_dump_0119/pending \
    --output-dir ./outputs_vllm_tide_dump_0119_outputs \
    --draft-global-batch-size 16 \
    --draft-micro-batch-size 1 \
    --num-epochs 3 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template gpt-oss \
    --baseline-dir /models/gpt-oss-120b-Eagle3