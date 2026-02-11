# Pneumonia Detection using DenseNet121 and Grad-CAM

This project provides a graphical application for detecting pneumonia from chest X-ray images using a fine-tuned DenseNet121 model.  
The interface allows loading an X-ray image, running inference, and visualizing the model’s attention using Grad-CAM.
Publicly available Chest X-Ray Pneumonia dataset (Kaggle) was used for training. 
Images were standardized to 224×224 px, normalized using ImageNet statistics, and augmented with clinically safe transforms (random horizontal flip, small rotations). 
To mitigate class imbalance we applied class-weighted cross-entropy during fine-tuning. 
Model was initialized with ImageNet weights, early convolutional blocks were frozen, and final classifier layers were retrained. 
Grad-CAM was used for qualitative interpretability.

## Install dependencies

Install required Python packages:

```bash
pip install -r requirements.txt

