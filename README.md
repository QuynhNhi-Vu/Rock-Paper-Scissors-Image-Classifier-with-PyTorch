# Rock-Paper-Scissors-Image-Classifier-with-PyTorch

A complete deep learning pipeline for classifying hand gestures (Rock, Paper, Scissors) using CNNs and transfer learning. Built as part of a university AI course project.

---

## Project Overview

This project walks through the full workflow of building an image classification system:

* Training a baseline CNN model using `nn.Sequential`
* Fine-tuning a pretrained **MobileNetV2** model
* Capturing a custom dataset via **webcam** using OpenCV
* Preprocessing, labeling, and organizing data using CSV and directory structure
* Evaluating performance with accuracy metrics and confusion matrices

---

## Tech Stack

* Python
* PyTorch & Torchvision
* OpenCV
* Pandas, NumPy, Matplotlib
* Pretrained Model: **MobileNetV2**

---

## Key Features

* **Baseline Model**: CNN built with `nn.Sequential`, trained on a clean dataset
* **Transfer Learning**: MobilenetV2 fine-tuned on the same dataset for improved accuracy
* **Webcam Deployment**: Captured and classified real-time images using a webcam and OpenCV
* **Data Handling**: Auto-save images to train/devtest/test folders with CSV labels
* **Evaluation**: Test accuracy (93.6%) and confusion matrix on webcam dataset

---

## Workflow Summary

1. **Task 1.1**: Build a CNN from scratch and train on clean RPS dataset
2. **Task 2.1 & 2.2**: Improve with transfer learning using MobileNetV2
3. **Task 2.3**: Analyze performance with confusion matrices and error analysis
4. **Task 3**: Capture a new dataset with webcam, retrain and evaluate performance

---

## Final Results

| Model        | Dataset Type    | Accuracy  |
| ------------ | --------------- | --------- |
| Baseline CNN | Clean test set  | \~94%     |
| MobileNetV2  | Clean test set  | 98.7%     |
| MobileNetV2  | Webcam test set | **93.6%** |

* Most common misclassification: Scissors → Rock
* Strong generalization despite lighting/background variation

---

## Demo Screenshots

> Include test accuracy output, confusion matrix, and webcam image previews here.

---

## What I Learned

* Practical end-to-end workflow for image classification
* The importance of consistent data preprocessing
* How domain shift impacts generalization
* Real-world limitations of pretrained models without fine-tuning

---

## To Run This Notebook

```bash
pip install torch torchvision opencv-python pandas matplotlib
```

Then open `main.ipynb` and run each task section (1.1, 2.1, 3.1, etc.).

