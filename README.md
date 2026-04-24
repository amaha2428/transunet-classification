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
- The **UNet decoder is discarded entirely**
- The **transformer patch token outputs are globally average-pooled** to form a fixed-size feature vector
- A lightweight **classification head** (LayerNorm → Dropout → Linear) replaces the segmentation output

This makes TransUNet a general-purpose classifier usable on any medical imaging task.

---

## Results

### Tuberculosis Detection (Chest X-Ray)

| Model | Accuracy | F1 Score | AUC-ROC | Pretrained | Epochs to Converge | Parameters |
|---|---|---|---|---|---|---|
| **TransUNet (Ours)** | **98.57%** | **98.67%** | **99.88%** | None | 3 | ~107M |
| DenseNet-121 | 100.00% | 100.00% | 100.00% | ImageNet | 31 | ~8M |

> TransUNet achieves near-identical performance to a purpose-built, pretrained classifier with no pretraining and 10x faster convergence.

Results on additional datasets (Pneumonia, Brain Tumor MRI, Melanoma) reported in the paper.

---

## Datasets

| Dataset | Modality | Classes | Images | Source |
|---|---|---|---|---|
| Tuberculosis | Chest X-ray | Normal / TB | 1,400 | Kaggle |
| Pneumonia | Chest X-ray | Normal / Pneumonia | ~5,800 | Kaggle |
| Brain Tumor | MRI | No Tumor / Tumor | ~3,000 | Kaggle |
| Melanoma | Dermoscopy | Benign / Malignant | 1,400 (balanced) | ISIC 2020 |

---

## Notebooks

Each notebook is self-contained. Only the config block at the top changes between datasets.

| Notebook | Dataset | Modality |
|---|---|---|
| TransUNet_TB_Classification.ipynb | Tuberculosis | Chest X-ray |
| TransUNet_Pneumonia_Classification.ipynb | Pneumonia | Chest X-ray |
| TransUNet_BrainTumor_Classification.ipynb | Brain Tumor | MRI |
| TransUNet_Melanoma_Classification.ipynb | Melanoma | Dermoscopy |
| TransUNet_CrossDataset_Comparison.ipynb | All datasets | Aggregated results |

---

## Quickstart (Google Colab)

1. Open any notebook in Google Colab
2. Mount your Google Drive
3. Edit the Dataset Config block (Section 2) — set DATASET_PATH and SAVE_DIR
4. Run all cells

The notebook will install dependencies, clone TransUNet automatically, train both models, and save all results to your Drive.

---

## Architecture

```
Input Image [B, 3, 224, 224]
        |
        v
ResNet-50 Encoder
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

---

## Adapting to Your Own Dataset

### Folder-based (most datasets)
```python
DATASET_NAME  = 'YourDataset'
DATASET_PATH  = '/path/to/your_dataset'
CLASS_NAMES   = ['class_a', 'class_b']
NUM_CLASSES   = 2
DATASET_TYPE  = 'folder'
```

### CSV-based (ISIC-style)
```python
DATASET_TYPE  = 'csv'
CSV_PATH      = '/path/to/labels.csv'
IMG_DIR       = '/path/to/images'
IMG_COL       = 'image_name'
LABEL_COL     = 'target'
```

Everything else runs unchanged.

---

## Citation

```bibtex
@misc{transunet-classification-2025,
  title  = {TransUNet for Medical Image Classification},
  author = {[Your Name]},
  year   = {2025},
  url    = {https://github.com/[your-username]/transunet-classification}
}
```

---

## Acknowledgements

- Original TransUNet: Chen et al., 2021 (https://arxiv.org/abs/2102.04306)
- Implementation: Beckschen/TransUNet (https://github.com/Beckschen/TransUNet)
