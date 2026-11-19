# HSLNet: A Hard-sample Focusing and Similarity-driven Label-calibration Network for Unsupervised Visible-Infrared Person Re-Identification
## Framework Overview
![Uploading image.png…]()

HSLNet is a hard‑sample focusing and similarity‑driven label‑calibration network for unsupervised visible‑infrared person re‑identification (US‑VI‑ReID). The MPA module is designed for modality‑adaptive cross‑modal feature association and hard‑sample mining, while the CLS module performs clustering‑aware pseudo‑label refinement. The two modules work collaboratively to mitigate pseudo‑label noise and boost cross‑modal feature alignment performance.

The task of this paper is unsupervised visible‑infrared person re‑identification (US‑VI‑ReID). Given visible and infrared pedestrian images, the model achieves cross‑modal pedestrian identity matching and retrieval without any manual annotations. The proposed HSLNet framework adopts TransReID ViT as the feature extraction backbone. It learns modality‑invariant discriminative features via modality‑adaptive proxy association and clustering‑aware label smoothing, alleviates pseudo‑label noise introduced by clustering, and improves the stability of cross‑modal feature alignment as well as retrieval accuracy.

1. **MPA: Modality‑adaptive Proxy Association**

   MPA conducts modality‑aware parameter calibration, feature‑diversity‑driven proxy quality assessment, proxy‑guided hard‑sample mining, and camera‑slot distribution regularization. It computes adaptive temperature parameters for visible and infrared modalities separately, evaluates the representativeness of each proxy according to intra‑proxy feature variance, assigns quality‑aware focal weights to hard samples, and introduces camera‑level distribution constraints. It mitigates modality discrepancy and the misguidance from low‑quality proxies, alleviates cross‑camera feature bias, and suppresses global feature‑level mismatches.

2. **CLS: Clustering‑aware Label Smoothing**

   CLS refines pseudo‑labels generated from initial clustering via multi‑factor adaptive smoothing coefficient modulation and similarity‑driven soft‑label assignment. It dynamically adjusts smoothing strength according to training epoch, sample‑to‑cluster distance and cluster size frequency. Then it assigns smoothing weights only to top‑K semantically similar clusters to construct semantic‑aware soft labels. This module further eliminates residual pseudo‑label noise, reduces error propagation caused by noisy clustering results, and stabilizes model training gradients.

# Prepare Datasets
Put SYSU-MM01 and RegDB dataset into data/sysu and data/regdb , run prepare\_sysu.py and prepare\_regdb.py to prepare the training data (convert to market1501 format).

# Prepare Pre-trained model
We adopt the self-supervised pre-trained models (ViT-B/16+ICS) from [Self-Supervised Pre-Training for Transformer-Based Person Re-Identification](https://github.com/damo-cv/TransReID-SSL?tab=readme-ov-file).
Download link:https://drive.google.com/file/d/1ZFMCBZ-lNFMeBD5K8PtJYJfYEk5D9isd/view

# Training

We utilize 2 4090 GPUs for training.

**examples:**

SYSU-MM01:

1. Train:
```shell
sh train_sysu.sh
```


2. Test:
```shell
sh test_sysu.sh
```

RegDB:

1. Train:
:
```shell
sh train_regdb.sh
```

2. Test:
```shell
sh test_regdb.sh
```


LLCM:

1. Train:
:
```shell
sh train_llcm.sh
```

2. Test:
```shell
sh test_llcm.sh
