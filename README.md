# Elaborative Rehearsal for Zero-shot Action Recognition
  
This is an official implementation of:

**Shizhe Chen and Dong Huang**, ***Elaborative Rehearsal for Zero-shot Action Recognition***, ICCV, 2021. [Arxiv Version](https://arxiv.org/abs/2108.02833)

Elaborating a new concept and relating it to known concepts, we reach the dawn of zero-shot action recognition models being comparable to supervised models trained on few samples.

New SOTA results are also achieved on the standard ZSAR benchmarks (Olympics, HMDB51, UCF101) as well as the first large scale ZSAR benchmak (we proposed) on the Kinetics database.  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/elaborative-rehearsal-for-zero-shot-action/zero-shot-action-recognition-on-hmdb51)](https://paperswithcode.com/sota/zero-shot-action-recognition-on-hmdb51?p=elaborative-rehearsal-for-zero-shot-action)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/elaborative-rehearsal-for-zero-shot-action/zero-shot-action-recognition-on-olympics)](https://paperswithcode.com/sota/zero-shot-action-recognition-on-olympics?p=elaborative-rehearsal-for-zero-shot-action)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/elaborative-rehearsal-for-zero-shot-action/zero-shot-action-recognition-on-ucf101)](https://paperswithcode.com/sota/zero-shot-action-recognition-on-ucf101?p=elaborative-rehearsal-for-zero-shot-action)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/elaborative-rehearsal-for-zero-shot-action/zero-shot-action-recognition-on-kinetics)](https://paperswithcode.com/sota/zero-shot-action-recognition-on-kinetics?p=elaborative-rehearsal-for-zero-shot-action)


<img src = "figures/teaser.png" width ="280" /> <img src = "figures/ZSARvsFew.png" width ="520" />
<img src = "figures/framework.png" width ="800" />


## Installation
```bash
git clone https://github.com/DeLightCMU/ElaborativeRehearsal.git
cd ElaborativeRehearsal
export PYTHONPATH=$(pwd):${PYTHONPATH}

pip install -r requirements.txt

# download pretrained models
bash scripts/download_premodels.sh
```


## Zero-shot Action Recognition (ZSAR)

### Extract Features in Video
1. spatial-temporal features
```bash
bash scripts/extract_tsm_features.sh '0,1,2'
```

2. object features
```bash
bash scripts/extract_object_features.sh '0,1,2'
```

### ZSAR Training and Inference

1. Baselines: [DEVISE](https://papers.nips.cc/paper/2013/hash/7cce53cf90577442771720a370c3c723-Abstract.html), [ALE](https://arxiv.org/abs/1503.08677), [SJE](https://arxiv.org/abs/1409.8403), [DEM](https://arxiv.org/abs/1611.05088), [ESZSL](http://proceedings.mlr.press/v37/romera-paredes15.html) and [GCN](https://arxiv.org/abs/2008.12432).
```bash
# mtype: devise, ale, sje, dem, eszsl
mtype=devise
CUDA_VISIBLE_DEVICES=0 python zeroshot/driver/zsl_baselines.py zeroshot/configs/zsl_baseline_${mtype}_config.yaml ${mtype} --is_train
CUDA_VISIBLE_DEVICES=0 python zeroshot/driver/zsl_baselines.py zeroshot/configs/zsl_baseline_${mtype}_config.yaml ${mtype} --eval_set tst
# evaluate other splits
ksplit=1
CUDA_VISIBLE_DEVICES=0 python zeroshot/driver/zsl_baselines_eval_splits.py zeroshot/configs/zsl_baseline_${mtype}_config.yaml ${mtype} ${ksplit}

# gcn
CUDA_VISIBLE_DEVICES=0 python zeroshot/driver/zsl_kgraphs.py zeroshot/configs/zsl_baseline_kgraph_config.yaml --is_train
CUDA_VISIBLE_DEVICES=0 python zeroshot/driver/zsl_kgraphs.py zeroshot/configs/zsl_baseline_kgraph_config.yaml --eval_set tst
```

2. ER-ZSAR and ablations:
```bash
# TSM + ED class representation + AttnPool (2nd row in Table 4(b))
CUDA_VISIBLE_DEVICES=0 python zeroshot/driver/zsl_vse.py zeroshot/configs/zsl_vse_wordembed_config.yaml --is_train --resume_file datasets/Kinetics/zsl220/word.glove42b.th

# TSM + ED class representation + BERT (last row in Table 4(a) and Table 4(b))
CUDA_VISIBLE_DEVICES=0 python zeroshot/driver/zsl_vse.py zeroshot/configs/zsl_vse_config.yaml --is_train

# Obj + ED class representation + BERT + ER Loss (last row in Table 4(c))
CUDA_VISIBLE_DEVICES=0 python zeroshot/driver/zsl_cptembed.py zeroshot/configs/zsl_cpt_config.yaml --is_train

# ER-ZSAR Full Model
CUDA_VISIBLE_DEVICES=0 python zeroshot/driver/zsl_ervse.py zeroshot/configs/zsl_ervse_config.yaml --is_train
```

## Citation
If you find this repository useful, please cite our paper:
```
@proceeding{ChenHuang2021ER,
  title={Elaborative Rehearsal for Zero-shot Action Recognition},
  author={Shizhe Chen and Dong Huang},
  booktitle = {ICCV},
  year={2021}
}
```

## Acknowledgement
- [BiT](https://github.com/google-research/big_transfer)
- [X-Temporal](https://github.com/Sense-X/X-Temporal)