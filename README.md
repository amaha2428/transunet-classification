# TransUNet for Medical Image Classification

> **Repurposing a segmentation transformer for general-purpose medical image classification — trained from scratch, no task-specific pretraining.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Overview

[TransUNet](https://arxiv.org/abs/2102.04306) was originally proposed for **medical image segmentation**. This project demonstrates that its hybrid ResNet + Vision Transformer encoder can be adapted for **binary and multi-class image classification** with competitive performance — even when trained *from scratch* with no pretrained weights.

### Key Contribution

The adaptation is straightforward but effective:

- The UNet decoder is discarded entirely
- The transformer patch token outputs are globally average-pooled to form a fixed-size feature vector
- A lightweight classification head (LayerNorm + Dropout + Linear) replaces the segmentation output

This makes TransUNet a general-purpose classifier usable on any medical imaging task, across modalities and disease types.

---

## Results

### Cross-Dataset Comparison

| Dataset | Modality | TransUNet Acc | TransUNet F1 | TransUNet AUC | DenseNet-121 Acc | DenseNet-121 F1 | DenseNet-121 AUC |
|---|---|---|---|---|---|---|---|
| Tuberculosis | Chest X-ray | 95.71% | 95.72% | 99.77% | 100.00% | 100.00% | 100.00% |
| Pneumonia | Chest X-ray | 97.79% | 97.80% | 99.77% | 97.79% | 97.80% | 99.90% |
| Brain Tumor | MRI | 95.56% | 95.56% | 99.46% | 98.89% | 98.89% | 99.98% |
| Skin Cancer | Dermoscopy | 85.20% | 85.17% | 94.57% | 90.94% | 90.89% | 97.03% |

**Key observations:**

- TransUNet matches DenseNet-121 exactly on Pneumonia (97.79%) with no pretraining
- AUC-ROC remains above 94% across all four datasets and three imaging modalities
- TransUNet is trained entirely from scratch; DenseNet-121 uses ImageNet pretrained weights
- Brain Tumor is the only multi-class experiment (4 classes), all others are binary

---

## Pretrained Weights

Model weights for all experiments are available on Google Drive.
They will be migrated to HuggingFace Hub upon paper publication.

| Dataset | Model | Link |
|---|---|---|
| Tuberculosis | TransUNet | [Download](https://drive.google.com/file/d/1EtIic7ffVNrpN2Oj7TUeXesqzTs9qR2w/view?usp=sharing) |
| Tuberculosis | DenseNet-121 | [Download](https://drive.google.com/file/d/1rBbTEXH4TBkP6quyjTyZTdy36hJRsrQH/view?usp=sharing) |
| Pneumonia | TransUNet | [Download](https://drive.google.com/file/d/1ZT-kKw5usHFhxRk8nwvP-MZl0Vax25cX/view?usp=sharing) |
| Pneumonia | DenseNet-121 | [Download](https://drive.google.com/file/d/1zcmz1faxz-bdrivOhESKnckgy4ZosaIX/view?usp=sharing) |
| Brain Tumor | TransUNet | [Download](https://drive.google.com/file/d/1L5qrizUJYvA0z4r_F601_jIT45daTqAk/view?usp=sharing) |
| Brain Tumor | DenseNet-121 | [Download](https://drive.google.com/file/d/1YUOZHIF_gMh_SAVkqx6sE7kljM2FscQ0/view?usp=sharing) |
| Skin Cancer | TransUNet | [Download](https://drive.google.com/file/d/1hDvMDLJwyq92An7gM9IA6gca9pg5xj0w/view?usp=sharing) |
| Skin Cancer | DenseNet-121 | [Download](https://drive.google.com/file/d/1b9_jvz5hC3jCOIUks3NEWX6UhjGzuloO/view?usp=sharing) |

---

## Datasets

| Dataset | Modality | Task | Classes | Images | Source |
|---|---|---|---|---|---|
| Tuberculosis | Chest X-ray | Binary | Normal / TB | 1,400 | [Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) |
| Pneumonia | Chest X-ray | Binary | Normal / Pneumonia | 5,856 | [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) |
| Brain Tumor | MRI | Multi-class | Glioma / Meningioma / No Tumor / Pituitary | 7,200 | [Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) |
| Skin Cancer | Dermoscopy | Binary | Benign / Malignant | 3,297 | [Kaggle](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign) |

### Dataset Notes

**Pneumonia:** The original Kaggle download comes pre-split into train/val/test folders where the val set contains only 16 images total. All splits were merged and re-split 80/10/10 with a fixed random seed for consistency. The dataset is naturally imbalanced (1,583 Normal vs 4,273 Pneumonia), reflecting real-world clinical distribution and used as-is.

**Brain Tumor:** An earlier smaller dataset of 253 images was initially identified but deemed too small for reliable evaluation. This larger dataset of 7,200 images (1,800 per class, perfectly balanced) was used instead.

**Skin Cancer:** The originally planned ISIC 2020 dataset (~116GB) was not feasible due to storage constraints on the free Google Colab and Drive tier. This curated dermoscopy dataset covering the same binary classification task was used instead and is fully reproducible without institutional compute resources.

---

## Notebooks

Each notebook is fully self-contained and follows an identical structure. Only the config block in Section 2 changes between datasets — all training, evaluation, and visualization code is identical across notebooks.

| Notebook | Dataset | Modality | Task |
|---|---|---|---|
| TransUNet_TB_Classification.ipynb | Tuberculosis | Chest X-ray | Binary |
| TransUNet_Pneumonia_Classification.ipynb | Pneumonia | Chest X-ray | Binary |
| TransUNet_BrainTumor_Classification.ipynb | Brain Tumor | MRI | Multi-class (4) |
| TransUNet_Melanoma_Classification.ipynb | Skin Cancer | Dermoscopy | Binary |
| TransUNet_CrossDataset_Comparison.ipynb | All datasets | Aggregated | Results + Paper Figures |

---

## Quickstart (Google Colab)

1. Open any notebook in Google Colab
2. Set runtime to T4 GPU (Runtime > Change runtime type)
3. Mount your Google Drive
4. Edit the Dataset Config block in Section 2 — set DATASET_PATH and SAVE_DIR
5. Run all cells

The notebook will install dependencies, clone the TransUNet repo automatically, train both TransUNet and DenseNet-121, and save weights, training history, and all plots to your Drive.

---

## Architecture

```
Input Image [B, 3, 224, 224]
        |
        v
ResNet-50 Encoder (CNN feature extractor)
        |
        v
ViT Transformer (12 layers, hidden_size=768)
  patch tokens: [B, 196, 768]
        |
        v
Global Average Pooling -> [B, 768]
        |
        v
LayerNorm -> Dropout(0.5) -> Linear(768 -> num_classes)
        |
        v
Class Logits [B, num_classes]
```

The original segmentation decoder (skip connections + upsampling) is removed entirely.
Mean pooling over all 196 patch tokens (14x14 grid for 224x224 input) forms the
classification feature vector. This design choice is discussed in detail in the paper.

---

## Adapting to Your Own Dataset

### Folder-based datasets (TB, Pneumonia, Brain Tumor, Skin Cancer)

```
your_dataset/
    class_a/   image1.jpg  image2.png ...
    class_b/   image1.jpg  image2.png ...
```

Edit the config block in Section 2:

```python
DATASET_NAME  = 'YourDataset'
DATASET_PATH  = '/path/to/your_dataset'
CLASS_NAMES   = ['class_a', 'class_b']
NUM_CLASSES   = 2
DATASET_TYPE  = 'folder'
```

### CSV-based datasets (ISIC-style)

```python
DATASET_TYPE  = 'csv'
CSV_PATH      = '/path/to/labels.csv'
IMG_DIR       = '/path/to/images'
IMG_COL       = 'image_name'   # column with filename (no extension)
LABEL_COL     = 'target'       # column with integer label
```

Everything else — training loop, evaluation, visualizations — runs unchanged for both dataset types.

---

## Model Size

| Model | Parameters | Saved Weight Size | Pretrained |
|---|---|---|---|
| TransUNet (Ours) | ~107M | ~401MB | None |
| DenseNet-121 | ~8M | ~27MB | ImageNet |

The larger parameter count in TransUNet reflects its hybrid ResNet50 + 12-layer ViT architecture.
Model compression and distillation are left as future work.

---

## Requirements

```
torch>=1.10
torchvision
timm
ml_collections
numpy
pandas
matplotlib
seaborn
scikit-learn
Pillow
tqdm
```

Also requires cloning [TransUNet](https://github.com/Beckschen/TransUNet) — done automatically in each notebook.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{amaha2026transunet,
  title  = {TransUNet for Medical Image Classification: Adapting Vision Transformers from Segmentation to Classification},
  author = {Amaha, Godspower O.},
  year   = {2026},
  url    = {https://github.com/amaha2428/transunet-classification}
}
```

---

## Acknowledgements

- Original TransUNet: Chen et al., 2021 (https://arxiv.org/abs/2102.04306)
- TransUNet implementation: Beckschen/TransUNet (https://github.com/Beckschen/TransUNet)
- DenseNet-121: Huang et al., 2017 (https://arxiv.org/abs/1608.06993)
