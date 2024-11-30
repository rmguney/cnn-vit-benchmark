# Comparative Analysis of DeiT and MobileNet

This project explores a comparative analysis between two deep learning models, **DeiT (Data-efficient Image Transformer)** and **MobileNetV3**, focusing on their architectures, training methodologies, and performance benchmarks. The models are trained on the CIFAR-10 dataset for image classification tasks.

## Usage

The `main.py` script provides a menu-driven interface for running different experiments.

```bash
python main.py
```

Select an option from the menu:
1. **Initialize Dataset (CIFAR-10)**: Initialize the dataset.
2. **Test Dataset (CIFAR-10)**: Evaluate the dataset and preprocessing pipeline.
3. **Train Train MobileNetV3S**: Train MobileNetV3 with user-specified hyperparameters.
4. **Train DeiT-T**: Train DeiT with user-specified hyperparameters.
5. **Exit**: Exits the interface.

When prompted, specify the following hyperparameters:
- **Enter number of epochs**: Example: `10`
- **Enter learning rate**: Example: `0.001`
- **Enter batch size**: Example: `32`

## Model Aspects

### Neural Network Architecture
- **DeiT**: Vision Transformer that processes image data by dividing it into patches and modeling their relationships using self-attention mechanisms.
- **MobileNetV3**: Convolutional Neural Network optimized for efficiency using localized convolution operations.

### Key Building Blocks
- **DeiT**: Transformer Encoder leveraging self-attention for understanding relationships between image regions.
- **MobileNetV3**: Depthwise Separable Convolutions and MLP layers for efficient feature extraction.

### Activation Functions
- **DeiT**: GELU (Gaussian Error Linear Unit) for smoother and more complex learning.
- **MobileNetV3**: Hard-Swish for computational efficiency, making it suitable for mobile devices.

### Normalization Techniques
- **DeiT**: Layer Norm for stabilizing training in sequential transformer architectures.
- **MobileNetV3**: Batch Norm for accelerating convergence in CNNs.

## Training Methodology

### Optimizer and Loss Function
- **Optimizer**: Adam optimizer.
- **Loss Function**: Cross-Entropy Loss for multi-class classification.

### Learning Rate Scheduler
A StepLR scheduler adjusts the learning rate to ensure stable convergence:
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
```

### Default Training Parameters
- **Epochs**: Configurable via `main.py` menu.
- **Learning Rate**: Configurable via `main.py` menu.
- **Batch Size**: 32 (fixed).

### Dataset and Preprocessing
- **Dataset**: CIFAR-10, containing 60,000 32x32 color images across 10 classes.
- **Split**: 80% training, 20% validation.
- **Normalization**: Mean and standard deviation of (0.5, 0.5, 0.5):
  ```python
  transform = transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])
  ```

### Metrics and Checkpointing
- Metrics like loss, precision, recall, F1 score, and mAP (mean Average Precision) are logged for model evaluation.
- Best-performing models are saved using a callback mechanism:
  ```python
  checkpoint_callback = ModelCheckpoint(
      monitor="val_map",
      mode="max",
      dirpath="saved_models/",
      filename="best_model"
  )
  ```

## Results and Performance Benchmarks

Performance metrics, including loss, precision, recall, and F1 scores, are evaluated for both models. The results help determine which model is better suited for tasks involving image classification, especially considering the trade-offs between efficiency and accuracy.

## License
This project is open-source and available under the MIT License.
