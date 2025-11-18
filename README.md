# Pneumonia Detection with DenseNet121 + Grad-CAM

Simple project for binary classification (PNEUMONIA vs NORMAL) on chest X-ray images using DenseNet121 and Grad-CAM visualization.

## 1. Dataset

Download dataset from Kaggle: "Chest X-Ray Images (Pneumonia)" and unpack it to:

`data/chest_xray/`

with subfolders:

- `train/NORMAL`, `train/PNEUMONIA`
- `val/NORMAL`, `val/PNEUMONIA`
- `test/NORMAL`, `test/PNEUMONIA` (optional for evaluation)

## 2. Install dependencies

```bash
pip install -r requirements.txt
