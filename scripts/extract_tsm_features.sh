#! /bin/bash

dataset=Kinetics

CUDA_VISIBLE_DEVICES=$1 python tools/extract_video_features.py \
    --config experiments/tsm/kinetics_tsm_resnet50_ft.yaml \
    --setnames train val test \
    --output_dir datasets/${dataset}/features/tsm_resnet50
