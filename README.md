# Comparative Analysis of MobileNet and DeiT-T on the CIFAR-10 Dataset

## Overview

This project examines and compares the performance of **MobileNetV3-L** and **DeiT-T (Data-efficient Image Transformer)**, a transformer-based vision model.
The aim is to compare the performance of a relatively lightweight vision transformer model to CNN models for resource constrained environments.
Tests were done with 3 different **MobileNet** models including **MobileNet V2, MobileNetV3-S, MobileNetV3-L**, but taking **MobileNetV3-L** as the baseline since most metrics had similar results and in terms of model size it is more comparable to **DeiT-T**.

## Usage

The `main.py` script serves as CLI for running the scripts available in this project.

```bash
python main.py
```

Select an option from the menu:
1. **Initialize Dataset (CIFAR-10)**: Initialize the dataset.
2. **Test Dataset (CIFAR-10)**: Evaluate the dataset and preprocessing pipeline.
3. **Train DeiT-T**: Train DeiT-T with user-specified hyperparameters.
4. **Train Train MobileNetV3S**: Train MobileNetV3 with user-specified hyperparameters.
5. **Train Train MobileNetV3L**: Train MobileNetV3L with user-specified hyperparameters.
6. **Train Train MobileNetV2**: Train MobileNetV2 with user-specified hyperparameters.
7. **Benchmark Models**: Benchmark best performing checkpoints of the models from the training.
8. **Exit**: Exits the interface.

When prompted, specify the following hyperparameters:
- **Enter number of epochs**: Example: `10`
- **Enter learning rate**: Example: `0.001`
- **Enter batch size**: Example: `32`

## Model Aspects

### **MobileNetV3-L**
- **Architecture:** Convolutional Neural Network (CNN).
- **Key Building Blocks:**
  - Depthwise separable convolutions.
  - Lightweight fully connected layers.
- **Activation Function:** Hard-swish (efficient for mobile devices).
- **Normalization:** Batch normalization.

### **DeiT-T (Vision Transformer)**
- **Architecture:** Transformer-based model that processes image patches.
- **Key Building Blocks:**
  - Self-attention mechanism.
  - Transformer encoder.
  - Class and distillation tokens.
- **Activation Function:** GELU (Gaussian Error Linear Unit).
- **Normalization:** Layer normalization.

## Training Methodology

### Dataset
- **CIFAR-10:** Contains 60,000 32x32 color images across 10 classes.
- Data normalized with a mean and standard deviation of `(0.5, 0.5, 0.5)`.

### Parameters
- **Optimizer:** Adam.
- **Learning Rate Scheduler:** StepLR with `step_size=10` and `gamma=0.1`.
- **Epochs:** 20.
- **Batch Size:** 32.

### Validation
- Dataset split into 80% training and 20% validation.
- Metrics such as precision, recall, and F1-score were logged for analysis.

## Training Results

### DeiT-T
- Training loss decreased consistently, but validation loss increased after 11 epochs, indicating overfitting.
- Achieved high precision, recall, and F1-score on training data but struggled with generalization post epoch 11.

### MobileNetV3-L
- Training loss reduced steadily, but validation loss showed signs of overfitting early.
- Suffered from low precision and recall, reflecting difficulty in learning complex patterns.

## Performance Benchmarks

| Metric         | MobileNetV3-L | DeiT-T  | Insights                                                                 |
|----------------|-------------|---------|--------------------------------------------------------------------------|
| **Accuracy**   | 12.9%       | 75.2%   | DeiT-T outperforms in accuracy.                                          |
| **Top-5 Accuracy** | 52.5%       | 98.2%   | DeiT-T reliably predicts within the top-5 outputs.                      |
| **Precision**  | 10.0%       | 75.1%   | MobileNet has many false positives, while DeiT-T is precise.            |
| **Recall**     | 12.8%       | 75.2%   | MobileNet fails to capture true positives effectively.                  |
| **F1 Score**   | 5.4%        | 75.1%   | Reflects DeiT-T's superior balance between precision and recall.        |
| **mAP**        | 17.0%       | 82.9%   | DeiT-T ranks predictions well across all classes.                       |
| **Latency**    | 13.4ms      | 22.2ms  | MobileNet is faster, suitable for real-time applications.               |
| **Throughput** | 74.52 img/s | 44.95 img/s | MobileNet processes more images due to its lightweight architecture.   |

### Key Observations
- **DeiT-T**: Dominates in accuracy metrics, making it ideal for high-quality applications.
- **MobileNetV3-L**: Excels in latency and throughput, better suited for low-power or real-time systems.

## Confusion Matrix Insights

### DeiT-T
- Higher precision and recall across all classes.
- Misclassifications are minimal and dispersed.

### MobileNetV3-L
- Strong bias towards predicting specific classes, resulting in high misclassification rates.
- Poor performance in predicting several classes.
