# Cytology Cancer Detection using Deep Learning

## Overview
This project explores deep learning approaches for cervical cytology image classification under limited data conditions. The goal is to classify cell images into three diagnostic categories:

- NILM (normal)
- LSIL (low-grade abnormality)
- HSIL (high-grade abnormality)

The dataset consists of approximately 600 images from the BMT ThinPrep Pap smear collection. Due to the small dataset size and high visual similarity between classes, this task presents a challenging classification problem.

---

## Objectives
- Evaluate the effectiveness of transfer learning on small medical datasets
- Compare multiple model architectures
- Analyze model behavior on fine-grained morphological differences

---

## Models Implemented

### 1. Control CNN
A baseline convolutional neural network trained from scratch to evaluate performance without transfer learning.

### 2. VGG19 (Pretrained)
- Used as a feature extractor
- Custom classification head added
- Fine-tuned on cytology dataset

### 3. ResNet50 (Pretrained)
- Uses residual connections for deeper feature extraction
- Final layer replaced with custom classifier
- Best performing model

### 4. CLIP-Based Model
- Pretrained vision encoder
- Custom classification head
- Partial fine-tuning applied
- Required additional tuning compared to CNN models

## Training Setup

- Optimizer: Adam  
- Batch Size: 16  
- Epochs: 30  

### Learning Rates
- Standard models: `1e-4`
- CLIP:
  - Encoder: `3e-5`
  - Classifier: `3e-4`

### Loss Function
- Sparse Categorical Crossentropy
### Results
| Model     | Performance Summary |
|----------|-------------------|
| CNN      | Lowest performance |
| VGG19    | Strong improvement |
| ResNet50 | Best overall performance |
| CLIP     | Moderate performance, required tuning |

