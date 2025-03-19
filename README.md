# Alzheimer's Disease Early Prediction: A Synergistic Fusion of Deep Learning Models

## Overview
This project focuses on the **early prediction of Alzheimer's Disease (AD)** using a **Fusion of deep learning model** that combines **ResNet50** and **Vision Transformer (ViT)** architectures. By leveraging ResNet50's local feature extraction capabilities and ViT's global dependency modeling, the model aims to achieve **high accuracy** in classifying AD stages (**Non_Demented, Mild_Demented, Moderate_Demented, Very_Mild_Demented**) from **MRI scans**.

## Table of Contents
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [Team](#team)

## Installation

### Prerequisites
- Python 3.8+
- TensorFlow 2.10+
- Scikit-learn, NumPy, Matplotlib, Seaborn, Pandas
- torch, torchvision, super-gradients, split-folders

### Setup
1. **Clone the Repository:**
```bash
git clone https://github.com/SahilChaudhary17/G28-AlzheimersResViT
```

2. **Create a Virtual Environment (Optional):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

## Dataset

### Source
The project uses MRI scans with four classes:
- **Non_Demented**
- **Mild_Demented**
- **Moderate_Demented**
- **Very_Mild_Demented**

**Note:** Dataset imbalance is addressed using **class weights** and **data augmentation.**

### Directory Structure
```
dataset/
├── Mild_Demented/
├── Moderate_Demented/
├── Non_Demented/
└── Very_Mild_Demented/
```

### Preparation
The code will split the dataset into:
- **Training (70%)**
- **Validation (10%)**
- **Test (20%)**

## Usage

### Run the following Code file
```bash
python Fusion95.ipynb
python ViT90.ipynb
```
- Trains both the **Fusion of ResNet50 & ViT model** and the **solo ViT model**.
- Saves the models and results in the `models/` directory.


## Model Architecture

  ### **Fusion of ResNet50 & ViT Model**
- **Input Processing:** MRI images of size **(128, 128, 3)**.
- **Feature Extraction:** Pretrained **ResNet50** for local feature extraction.
- **Patch Embedding:** Creates **16 patches** using Conv2D.
- **Transformer Processing:** Six transformer encoder layers with **8 heads**.
- **Output:** Dense layer with **Softmax** activation for classification.

### **Solo ViT Model**
- Directly processes MRI images of size **(224, 224, 3)**.
- Similar transformer and classification pipeline.

## Training and Evaluation

### Training
**Hyperparameters:**
- **Optimizer:** AdamW
- **Learning Rate Adjustment:** ReduceLROnPlateau
- **Early Stopping:** Prevents overfitting

### Evaluation Metrics
- **Accuracy**
- **Precision**
- **Recall**
- **Specificity**
- **F1 Score**

## Results

### Performance
| Model                  | CNN      | ViT     | ResNet50 + ViT |
|------------------------|----------|---------|----------------|
| Training Accuracy	     | 83.82%	  | 92.17%	| 95.86%         |
| Validation Accuracy	   | 81.98	  | 90.79%	| 91.90%         |
| Precision	             | -	      | 90.04%	| 93.33%         |
| Recall	               | -	      | 92.13%	| 90.51%         |
| Specificity	           | -	      | 96.33%	| 96.66%         |
| F1 Score	             | -	      | 90.83%	| 91.80%         |

**Key Insights:**
- Hybrid model outperforms solo ViT.
- Effective handling of imbalanced data using class weights and augmentation.

## Contributing
We welcome contributions! Follow these steps:
1. **Fork** the repository.
2. **Create a Branch:** `git checkout -b feature-branch`
3. **Commit Changes:** `git commit -m "Add feature"`
4. **Push Changes:** `git push origin feature-branch`
5. **Open a Pull Request**

## Team
- **Sahil Chaudhary**: ViT implementation and model integration
- **Meet Prakash Joysar**: ResNet50 implementation and Training the model
- **Aaditya Vitthal More**: Data preparation and augmentation
- **Arpit Baheti**: Literature Review
- **Dakshita Motwani**: Evaluation and Reporting
