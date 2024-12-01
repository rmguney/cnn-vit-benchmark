import pandas as pd
import matplotlib.pyplot as plt

# Load the uploaded CSV file
metrics_file_path = '../logs/MobileNetV2CIFAR10_2024-12-01_00-11-05/version_0/metrics.csv'
metrics_data = pd.read_csv(metrics_file_path)

# Filter rows for training and validation based on the presence of metrics
epoch_data = metrics_data.dropna(subset=['epoch']).reset_index(drop=True)
train_rows = epoch_data.dropna(subset=['train_precision'])
val_rows = epoch_data.dropna(subset=['val_precision'])

# Extract relevant metrics
train_loss = train_rows['train_loss']
val_loss = val_rows['val_loss']
train_precision = train_rows['train_precision']
val_precision = val_rows['val_precision']
train_recall = train_rows['train_recall']
val_recall = val_rows['val_recall']
train_f1 = train_rows['train_f1']
val_f1 = val_rows['val_f1']
train_map = train_rows['train_map']
val_map = val_rows['val_map']
grad_norm = train_rows['grad_norm']
epochs = train_rows['epoch'].reset_index(drop=True) + 1

# Define custom plotting style
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
        'font.size': 14,  # Larger font size
        'legend.fontsize': 12,  # Larger legend font size
        'lines.linewidth': 2,  # Thicker lines
        'lines.markersize': 8  # Larger markers
    })

# Apply the updated style
custom_plot_style_with_larger_figsize()

# Output directory
output_dir = '/mnt/data/'

# Generate and save plots with larger styling
# Training and Validation Loss
plt.figure(figsize=(12, 8))  # Larger figure size
plt.plot(epochs, train_loss, marker='o', linestyle='-', label="Training Loss")
plt.plot(epochs, val_loss, marker='x', linestyle='--', label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}metrics_training_validation_loss_larger.png", facecolor='#212121')
plt.close()

# Precision
plt.figure(figsize=(12, 8))  # Larger figure size
plt.plot(epochs, train_precision, marker='o', linestyle='-', label="Training Precision")
plt.plot(epochs, val_precision, marker='x', linestyle='--', label="Validation Precision")
plt.xlabel("Epochs")
plt.ylabel("Precision")
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}metrics_precision_larger.png", facecolor='#212121')
plt.close()

# Recall
plt.figure(figsize=(12, 8))  # Larger figure size
plt.plot(epochs, train_recall, marker='o', linestyle='-', label="Training Recall")
plt.plot(epochs, val_recall, marker='x', linestyle='--', label="Validation Recall")
plt.xlabel("Epochs")
plt.ylabel("Recall")
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}metrics_recall_larger.png", facecolor='#212121')
plt.close()

# F1 Score
plt.figure(figsize=(12, 8))  # Larger figure size
plt.plot(epochs, train_f1, marker='o', linestyle='-', label="Training F1 Score")
plt.plot(epochs, val_f1, marker='x', linestyle='--', label="Validation F1 Score")
plt.xlabel("Epochs")
plt.ylabel("F1 Score")
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}metrics_f1_score_larger.png", facecolor='#212121')
plt.close()

# mAP
plt.figure(figsize=(12, 8))  # Larger figure size
plt.plot(epochs, train_map, marker='o', linestyle='-', label="Training mAP")
plt.plot(epochs, val_map, marker='x', linestyle='--', label="Validation mAP")
plt.xlabel("Epochs")
plt.ylabel("mAP")
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}metrics_map_larger.png", facecolor='#212121')
plt.close()

# Gradient Norm
plt.figure(figsize=(12, 8))  # Larger figure size
plt.plot(epochs, grad_norm, marker='o', linestyle='-', label="Gradient Norm")
plt.xlabel("Epochs")
plt.ylabel("Gradient Norm")
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}metrics_gradient_norm_larger.png", facecolor='#212121')
plt.close()

# Provide the updated files for download
larger_plot_files = [
    "metrics_training_validation_loss_larger.png",
    "metrics_precision_larger.png",
    "metrics_recall_larger.png",
    "metrics_f1_score_larger.png",
    "metrics_map_larger.png",
    "metrics_gradient_norm_larger.png",
]

larger_plot_file_paths = [f"{output_dir}{file}" for file in larger_plot_files]
larger_plot_file_paths