# RRC-Gen: Recalibrated Visual Feature Extraction and Relational Contrastive Learning for Automated Radiology Report Generation

RRC-Gen is a lightweight framework for **Automated Radiology Report Generation (ARRG)** from chest X-ray images. The proposed method improves visual representation learning and cross-modal semantic alignment while maintaining the efficient **BaseCMN** architecture.

## Overview

Despite recent advances in vision-language models, existing radiology report generation approaches still face several challenges:

1. **Insufficient visual representation learning**, making subtle pathological findings difficult to recognize.
2. **Weak cross-modal semantic alignment**, limiting accurate image-report correspondence.
3. **Limited recognition of rare abnormalities**, reducing clinical reliability.

To address these challenges, RRC-Gen introduces two key components:

* **Recalibrated Visual Feature Extraction (RVFE)**
  Enhances disease-related visual representations through spatial and channel-aware feature recalibration.

* **Relational Contrastive Clustering Learning (RCCL)**
  Improves semantic consistency by learning relational structures among visual representations through contrastive learning.

Together, these modules improve visual feature quality, multimodal alignment, and radiology report generation performance.

---
# Framework Overview

<p align="center">

<img width="1081" height="677" alt="RRC-Gen" src="https://github.com/user-attachments/assets/596a7189-53b6-454e-a7ec-307755e68cbf" />
</p>
# Important Notice

This repository provides the official PyTorch implementation of:

**RRC-Gen: Recalibrated Visual Feature Extraction and Relational Contrastive Learning for Automated Radiology Report Generation**

If you use this repository in your research, please cite our paper.

This repository includes:

* PyTorch implementation
* Training pipeline
* Evaluation scripts
* IU X-Ray and MIMIC-CXR support
* Preprocessing and inference pipeline

---

# Key Features

| Module                 | Purpose                                                                          |
| ---------------------- | -------------------------------------------------------------------------------- |
| **RVFE**               | Enhances disease-related visual features using spatial and channel recalibration |
| **RCCL**               | Learns relational contrastive representations for stronger semantic consistency  |
| **BaseCMN Decoder**    | Generates radiology reports from enhanced visual representations                 |
| **Joint Optimization** | Optimizes report generation and contrastive learning simultaneously              |

The proposed framework provides:

* Improved visual feature representation
* Stronger multimodal semantic alignment
* Better abnormality recognition
* More accurate report generation
* Lightweight computational design

---

# Framework Overview

## Simplified Algorithm

```python
# Input:
# Chest X-ray image

# Output:
# Radiology Report

# 1. Visual Feature Extraction
F = ResNet101(Image)

# 2. Recalibrated Visual Feature Extraction
F_refined = RVFE(F)

# 3. Relational Contrastive Learning
Z = ProjectionHead(F_refined)
loss_rccl = RCCL(Z)

# 4. Cross Memory Encoding
Memory = BaseCMNEncoder(F_refined)

# 5. Report Generation
Report = BaseCMNDecoder(Memory)

# Final Objective
Loss = CaptionLoss + λ × RCCL
```

---

# Requirements

The code has been tested with:

* Python >= 3.8
* PyTorch >= 1.10
* CUDA >= 11.3

Required packages:

```text
torch
torchvision
numpy
scipy
transformers
opencv-python
matplotlib
scikit-learn
pandas
tqdm
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Dataset

Supported datasets:

* IU X-Ray
* MIMIC-CXR

Dataset structure:

```
RRC-Gen/
│
├── config/
├── data/
│   ├── iu_xray/
│   └── mimic_cxr/
│
├── models/
├── modules/
│   ├── dataloader/
│   ├── losses/
│   ├── metrics/
│   ├── tokenizer/
│   └── visual_extractor/
│
├── pycocoevalcap/
│
├── train_iu_xray.sh
├── train_mimic_cxr.sh
├── test_iu_xray.sh
├── test_mimic_cxr.sh
│
├── main_train.py
├── main_test.py
└── README.md
```

---

# Training

## IU X-Ray

```bash
bash train_iu_xray.sh
```

## MIMIC-CXR

```bash
bash train_mimic_cxr.sh
```

---

# Testing

## IU X-Ray

```bash
bash test_iu_xray.sh
```

## MIMIC-CXR

```bash
bash test_mimic_cxr.sh
```



# Evaluation Metrics

The following metrics are used:

* BLEU-1
* BLEU-2
* BLEU-3
* BLEU-4
* METEOR
* ROUGE-L
* CIDEr

---

# Citation

If you find this repository useful, please cite:

```bibtex
@article{usman2026rrcgen,
  title={RRC-Gen: Recalibrated Visual Feature Extraction and Relational Contrastive Learning for Automated Radiology Report Generation},
  author={Usman, M. and Co-authors},
  journal={Multimedia Systems},
  year={2026},
  note={Under Review}
}
```

---

# Acknowledgments

We thank the authors of the following open-source projects:

* R2Gen
* BaseCMN
* PyTorch
* IU X-Ray Dataset
* MIMIC-CXR Dataset

---

# Contact

For questions, suggestions, or collaborations, please open an Issue or Pull Request.

Thank you for your interest in **RRC-Gen**.
