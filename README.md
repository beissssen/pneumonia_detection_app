# Pneumonia Detection using DenseNet121 and Grad-CAM

This project provides a graphical application for detecting pneumonia from chest X-ray images using a fine-tuned DenseNet121 model.  
The interface allows loading an X-ray image, running inference, and visualizing the model’s attention using Grad-CAM.

---

## 1. Dataset

Download dataset from Kaggle: **"Chest X-Ray Images (Pneumonia)"** and unpack it to:


with subfolders:

- `train/NORMAL`, `train/PNEUMONIA`  
- `val/NORMAL`, `val/PNEUMONIA`  
- `test/NORMAL`, `test/PNEUMONIA` (optional for evaluation)

Dataset link:  
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

---

## 2. Install dependencies

Install required Python packages:

```bash
pip install -r requirements.txt

