import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Updated Data for DeiT-T
deit_metrics = {
    "accuracy": 0.7519,
    "top_5_accuracy": 0.9818,
    "precision": 0.7514,
    "recall": 0.7519,
    "f1_score": 0.7510,
    "mAP": 0.8292,
    "latency": 0.0258,
    "throughput": 38.74,
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

# Updated Data for EfficientNet-B0
effnet_metrics = {
    "accuracy": 0.9317,
    "top_5_accuracy": 0.9981,
    "precision": 0.9314,
    "recall": 0.9317,
    "f1_score": 0.9315,
    "mAP": 0.9788,
    "latency": 0.0394,
    "throughput": 25.37,
}
effnet_conf_matrix = np.array([
    [946, 4, 11, 4, 0, 0, 1, 2, 23, 9],
    [0, 975, 1, 1, 0, 0, 2, 0, 3, 18],
    [17, 1, 911, 16, 22, 8, 12, 10, 2, 1],
    [8, 2, 22, 850, 18, 68, 14, 13, 4, 1],
    [5, 0, 8, 13, 942, 4, 6, 19, 2, 1],
    [4, 0, 15, 70, 18, 871, 4, 14, 2, 2],
    [5, 0, 13, 18, 8, 3, 950, 0, 3, 0],
    [4, 0, 4, 8, 6, 2, 2, 971, 0, 3],
    [15, 6, 2, 1, 0, 0, 0, 0, 966, 10],
    [5, 41, 2, 3, 0, 0, 3, 2, 9, 935],
])
effnet_class_metrics = {
    "precision": [0.94, 0.95, 0.92, 0.86, 0.93, 0.91, 0.96, 0.94, 0.95, 0.95],
    "recall": [0.95, 0.97, 0.91, 0.85, 0.94, 0.87, 0.95, 0.97, 0.97, 0.94],
    "f1_score": [0.94, 0.96, 0.92, 0.86, 0.94, 0.89, 0.95, 0.96, 0.96, 0.94],
}

# Create a directory for saving plots
save_dir = "./plots"
os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Define the save paths
confusion_matrix_path = os.path.join(save_dir, "confusion_matrices.png")
class_metrics_path = os.path.join(save_dir, "class_metrics.png")
overall_metrics_path = os.path.join(save_dir, "overall_metrics.png")

# Apply custom plot style
def custom_plot_style_with_larger_figsize():
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': '#212121',
        'axes.facecolor': '#212121',
        'axes.edgecolor': 'white',
        'text.color': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'grid.color': 'white',
        'axes.labelcolor': 'white',
        'font.size': 14,
        'legend.fontsize': 12,
        'lines.linewidth': 2,
        'lines.markersize': 8
    })
custom_plot_style_with_larger_figsize()

# Plot 1: Confusion Matrices
fig1, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
sns.heatmap(deit_conf_matrix, annot=True, fmt='g', cmap='coolwarm', ax=axes[0], cbar=False)
axes[0].set_title("DeiT-T Confusion Matrix", color="white")
sns.heatmap(effnet_conf_matrix, annot=True, fmt='g', cmap='coolwarm', ax=axes[1], cbar=False)
axes[1].set_title("EfficientNet-B0 Confusion Matrix", color="white")
fig1.savefig(confusion_matrix_path, facecolor="#212121")
plt.close(fig1)

# Plot 2: Class-Wise Metrics
fig2, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=100)
x_labels = [f"{i}" for i in range(10)]
metrics = ["precision", "recall", "f1_score"]
titles = ["Precision by Class", "Recall by Class", "F1-Score by Class"]
deit_data = [deit_class_metrics[m] for m in metrics]
effnet_data = [effnet_class_metrics[m] for m in metrics]

for i, ax in enumerate(axes):
    ax.bar(x_labels, deit_data[i], alpha=0.7, label="DeiT-T")
    ax.bar(x_labels, effnet_data[i], alpha=0.7, label="EfficientNet-B0")
    ax.set_title(titles[i], color="white")
    ax.legend()

fig2.savefig(class_metrics_path, facecolor="#212121")
plt.close(fig2)

# Plot 3: Overall Model Metrics
fig3, ax = plt.subplots(figsize=(10, 6), dpi=100)
overall_metrics = ["accuracy", "top_5_accuracy", "precision", "recall", "f1_score", "mAP", "latency", "throughput"]
deit_overall = [deit_metrics[m] for m in overall_metrics]
effnet_overall = [effnet_metrics[m] for m in overall_metrics]
x_labels = ["Accuracy", "Top-5 Accuracy", "Precision", "Recall", "F1-Score", "mAP", "Latency", "Throughput"]

x = np.arange(len(x_labels))
width = 0.35
ax.bar(x - width / 2, deit_overall, width, label="DeiT-T")
ax.bar(x + width / 2, effnet_overall, width, label="EfficientNet-B0")
ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=45, ha="right")
ax.legend()
fig3.savefig(overall_metrics_path, facecolor="#212121")
plt.close(fig3)

print("Plots saved as: 'confusion_matrices.png', 'class_metrics.png', 'overall_metrics.png'")
