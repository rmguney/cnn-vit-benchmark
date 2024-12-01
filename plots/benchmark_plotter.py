import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Updated Data for MobileNetV2
mobilenet_metrics = {
    "accuracy": 0.1000,
    "top_5_accuracy": 0.5000,
    "precision": 0.0100,
    "recall": 0.1000,
    "f1_score": 0.0182,
    "mAP": 0.1006,
    "latency": 0.0298,
    "throughput": 33.57,
}
mobilenet_conf_matrix = np.array([
    [0, 0, 0, 0, 0, 0, 0, 1000, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1000, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1000, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1000, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1000, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1000, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1000, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1000, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1000, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1000, 0, 0],
])
mobilenet_class_metrics = {
    "precision": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00, 0.00],
    "recall": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00],
    "f1_score": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.18, 0.00, 0.00],
}

# Updated Data for DeiT-T
deit_metrics = {
    "accuracy": 0.7519,
    "top_5_accuracy": 0.9818,
    "precision": 0.7514,
    "recall": 0.7519,
    "f1_score": 0.7510,
    "mAP": 0.8292,
    "latency": 0.0229,
    "throughput": 43.59,
}
deit_conf_matrix = np.array([
    [802, 21, 36, 12, 14, 5, 12, 8, 61, 29],
    [30, 849, 3, 6, 0, 0, 5, 4, 23, 80],
    [73, 5, 633, 47, 92, 42, 60, 25, 14, 9],
    [22, 10, 49, 581, 66, 168, 56, 27, 13, 8],
    [17, 3, 64, 43, 743, 29, 42, 43, 13, 3],
    [14, 6, 32, 174, 37, 656, 19, 53, 5, 4],
    [7, 3, 28, 58, 39, 10, 840, 5, 6, 4],
    [28, 3, 33, 32, 59, 52, 8, 767, 4, 14],
    [70, 26, 8, 13, 3, 4, 3, 3, 856, 14],
    [38, 103, 3, 9, 8, 3, 11, 8, 25, 792],
])
deit_class_metrics = {
    "precision": [0.73, 0.83, 0.71, 0.60, 0.70, 0.68, 0.80, 0.81, 0.84, 0.83],
    "recall": [0.80, 0.85, 0.63, 0.58, 0.74, 0.66, 0.84, 0.77, 0.86, 0.79],
    "f1_score": [0.76, 0.84, 0.67, 0.59, 0.72, 0.67, 0.82, 0.79, 0.85, 0.81],
}

# Create a directory for saving plots
save_dir = "./plots"
os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Define the save paths
confusion_matrix_path = os.path.join(save_dir, "confusion_matrices.png")
class_metrics_path = os.path.join(save_dir, "class_metrics.png")
overall_metrics_path = os.path.join(save_dir, "overall_metrics.png")

# Plot 1: Confusion Matrices
fig1, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
sns.heatmap(mobilenet_conf_matrix, annot=True, fmt='g', cmap='coolwarm', ax=axes[0], cbar=False)
axes[0].set_title("MobileNet Confusion Matrix", color="white")
sns.heatmap(deit_conf_matrix, annot=True, fmt='g', cmap='coolwarm', ax=axes[1], cbar=False)
axes[1].set_title("DeiT-T Confusion Matrix", color="white")
fig1.patch.set_facecolor("#212121")
for ax in axes:
    ax.set_facecolor("#212121")
fig1.savefig(confusion_matrix_path, facecolor="#212121")
plt.close(fig1)

# Plot 2: Class-Wise Metrics
fig2, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=100)
x_labels = [f"Class {i}" for i in range(10)]
metrics = ["precision", "recall", "f1_score"]
titles = ["Precision by Class", "Recall by Class", "F1-Score by Class"]
mobilenet_data = [mobilenet_class_metrics[m] for m in metrics]
deit_data = [deit_class_metrics[m] for m in metrics]

for i, ax in enumerate(axes):
    ax.bar(x_labels, mobilenet_data[i], alpha=0.7, label="MobileNet")
    ax.bar(x_labels, deit_data[i], alpha=0.7, label="DeiT-T")
    ax.set_title(titles[i], color="white")
    ax.legend()
    ax.set_facecolor("#212121")

fig2.patch.set_facecolor("#212121")
fig2.savefig(class_metrics_path, facecolor="#212121")
plt.close(fig2)

# Plot 3: Overall Model Metrics
fig3, ax = plt.subplots(figsize=(10, 6), dpi=100)
overall_metrics = ["accuracy", "top_5_accuracy", "precision", "recall", "f1_score", "mAP", "latency", "throughput"]
mobilenet_overall = [mobilenet_metrics[m] for m in overall_metrics]
deit_overall = [deit_metrics[m] for m in overall_metrics]
x_labels = ["Accuracy", "Top-5 Accuracy", "Precision", "Recall", "F1-Score", "mAP", "Latency", "Throughput"]

x = np.arange(len(x_labels))
width = 0.35
ax.bar(x - width / 2, mobilenet_overall, width, label="MobileNet")
ax.bar(x + width / 2, deit_overall, width, label="DeiT-T")
ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=45, ha="right")
ax.legend()
ax.set_facecolor("#212121")
fig3.patch.set_facecolor("#212121")
fig3.savefig(overall_metrics_path, facecolor="#212121")
plt.close(fig3)

print("Plots saved as: 'confusion_matrices.png', 'class_metrics.png', 'overall_metrics.png'")
