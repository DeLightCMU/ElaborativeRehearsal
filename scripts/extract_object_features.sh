#!/bin/bash

dataset=Kinetics

CUDA_VISIBLE_DEVICES=$1 python bit_pytorch/extract_bit_feature.py \
    --model_dir datasets/premodels/BiT \
    --model BiT-M-R50x1 --video_dir datasets/${dataset}/videos \
    --video_meta_file datasets/${dataset}/zsl220/{}_video_metas.jsonl \
    --output_dir datasets/${dataset}/features/bit_m_r50x1 \
    --num_workers 6
