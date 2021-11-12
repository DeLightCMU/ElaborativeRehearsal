#! /bin/bash

mkdir -p datasets/premodels/BiT
wget -P datasets/premodels/BiT https://storage.googleapis.com/bit_models/BiT-M-R50x1.npz
wget -P datasets/premodels/BiT https://storage.googleapis.com/bit_models/imagenet21k_wordnet_ids.txt
wget -P datasets/premodels/BiT https://storage.googleapis.com/bit_models/imagenet21k_wordnet_lemmas.txt

mkdir -p datasets/premodels/TSM
# wget -P datasets/premodels/TSM https://file.lzhu.me/projects/tsm/models/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense.pth
wget -P datasets/premodels/TSM https://hanlab.mit.edu/projects/tsm/models/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense.pth
