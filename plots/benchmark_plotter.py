import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Function to plot class metrics
def save_class_metrics_plot(metrics, title, save_path):
    classes = [f"Class {i}" for i in range(10)]
    x = np.arange(len(classes))
    width = 0.25

    plt.figure(figsize=(10, 6))
    plt.bar(x - width, metrics["Precision"], width, label="Precision", color='lightblue')
    plt.bar(x, metrics["Recall"], width, label="Recall", color='orange')
    plt.bar(x + width, metrics["F1-Score"], width, label="F1-Score", color='green')

    plt.xlabel("Classes", fontsize=12, color='white')
    plt.ylabel("Scores", fontsize=12, color='white')
    plt.title(title, fontsize=14, color='white')
    plt.xticks(x, classes, rotation=45, fontsize=10, color='white')
    plt.yticks(fontsize=10, color='white')
    plt.legend(fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.gca().set_facecolor("#212121")
    plt.gcf().set_facecolor("#212121")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# Function to plot confusion matrix
def save_confusion_matrix_plot(conf_matrix, title, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=[f"Class {i}" for i in range(10)], yticklabels=[f"Class {i}" for i in range(10)])
    plt.xlabel("Predicted Labels", fontsize=12, color='white')
    plt.ylabel("True Labels", fontsize=12, color='white')
    plt.title(title, fontsize=14, color='white')
    plt.xticks(fontsize=10, color='white')
    plt.yticks(fontsize=10, color='white', rotation=0)
    plt.gca().set_facecolor("#212121")
    plt.gcf().set_facecolor("#212121")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# MobileNetV3-Large Metrics
mobilenet_large_metrics = {
    "Precision": [0.09, 0.12, 0.05, 0.11, 0.18, 0.00, 0.21, 0.13, 1.00, 0.00],
    "Recall": [0.35, 0.00, 0.00, 0.62, 0.01, 0.00, 0.01, 0.02, 0.00, 0.00],
    "F1-Score": [0.15, 0.01, 0.01, 0.18, 0.01, 0.00, 0.01, 0.04, 0.00, 0.00],
}
mobilenet_large_conf_matrix = np.array([
    [355, 4, 10, 626, 2, 0, 0, 3, 0, 0],
    [279, 3, 23, 657, 3, 1, 1, 33, 0, 0],
    [452, 0, 4, 518, 8, 2, 3, 13, 0, 0],
    [322, 4, 5, 623, 7, 3, 10, 26, 0, 0],
    [477, 2, 3, 493, 6, 1, 2, 16, 0, 0],
    [310, 1, 2, 663, 4, 0, 2, 18, 0, 0],
    [697, 6, 2, 268, 2, 0, 6, 19, 0, 0],
    [361, 2, 1, 609, 1, 0, 4, 22, 0, 0],
    [200, 1, 11, 775, 1, 0, 1, 10, 1, 0],
    [290, 3, 13, 685, 0, 0, 0, 9, 0, 0],
])

# DeiT-T Metrics
deit_metrics = {
    "Precision": [0.73, 0.83, 0.71, 0.60, 0.70, 0.68, 0.80, 0.81, 0.84, 0.83],
    "Recall": [0.80, 0.85, 0.63, 0.58, 0.74, 0.66, 0.84, 0.77, 0.86, 0.79],
    "F1-Score": [0.76, 0.84, 0.67, 0.59, 0.72, 0.67, 0.82, 0.79, 0.85, 0.81],
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

# Save MobileNetV3-Large plots
save_class_metrics_plot(mobilenet_large_metrics, "MobileNetV3-Large Metrics", "mobilenet_large_metrics.png")
save_confusion_matrix_plot(mobilenet_large_conf_matrix, "MobileNetV3-Large Confusion Matrix", "mobilenet_large_conf_matrix.png")

# Save DeiT-T plots
save_class_metrics_plot(deit_metrics, "DeiT-T Metrics", "deit_metrics.png")
save_confusion_matrix_plot(deit_conf_matrix, "DeiT-T Confusion Matrix", "deit_conf_matrix.png")
