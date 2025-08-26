# <img width="30" height="30" alt="通知" src="https://github.com/user-attachments/assets/d88b47b7-6a9b-40e5-95ea-25c802aefa9e" /> Solution of Team fwzfwzfwz for FLARE2025 Challenge

**This repository is the official implementation of A Dual-Branch Fusion Network with Asymmetric Depthwise Convolutions for Whole-Body Tumor Segmentation of Team fwzfwzfwz on FLARE25 challenge.**

## Introduction

### Our network.


<img src="https://github.com/Wenze-Fan/FLARE25-Task1/blob/main/img/model1.png" alt="image" width="65%"/><img src="https://github.com/Wenze-Fan/FLARE25-Task1/blob/main/img/model2.png" alt="image" width="70%"/>

## 📂 Key Files in this Repository
Only one file you need to focus on:
[nnUNetTrainernew.py](https://github.com/Wenze-Fan/FLARE25-Task1/blob/main/nnunetv2/training/nnUNetTrainer/nnUNetTrainernew.py)
    （A custom nnUNet trainer）
## Environments and Requirements

The basic language for our work is [python](https://www.python.org/), and the baseline
is [nnU-Net](https://github.com/MIC-DKFZ/nnUNet).In addition, you also need to configure a virtual environment that matches nn-UNet.First,ensure you have **PyTorch > 2.0** installed with **CUDA > 11.6** support.
Set up your environment by running:

```
conda create -n FLARE24_nnUNet
conda activate FLARE24_nnUNet
pip install -e .
```

## Prepocessing
We used over 5000 partially annotated images provided by FLARE25 challenge, and did not use unlabeled data or pseudo labels.Preprocessing still follows the default nn-UNet preprocessing process.First，you need to put labeled data into ``nnUNet_raw`` in the following structure:
```
Dataset555_BodyTumor/
├── imagesTr/
│   ├── Adrenal_Ki67_Seg_001_0000.nii.gz
│   ├── ...
│   └── (all 5,000 labeled images ending with .nii.gz)
├── labelsTr/
│   ├── Adrenal_Ki67_Seg_001.nii.gz
│   ├── ...
└── dataset.json
```

The preprocessing command is as follows：
```
nnUNetv2_plan_and_preprocess -d 555 --verify_dataset_integrity
nnUNetv2_preprocess -d 555 -c 3d_fullres
```

## Training
```
nnUNetv2_train 555 3d_fullres all -tr nnUNetTrainernew
```

## Inference
```
nnUNetv2_predict -d 555 -i Testimages path -o Path for saving test image results -f all -tr nnUNetTrainernew -c 3d_fullres -chk checkpoint_best.pth -p nnUNetPlans
```

## Results
<img src="https://github.com/Wenze-Fan/FLARE25-Task1/blob/main/img/results.png" alt="image" width="70%"/>







