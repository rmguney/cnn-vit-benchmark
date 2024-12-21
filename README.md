## Overview

This project examines and compares the performance of lightweight vision transformers to convolutional neural networks with similar sizes. 
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
