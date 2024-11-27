import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Define class names for CIFAR-10
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def load_cifar10_batch(file_path):
    """Load a single batch of CIFAR-10 dataset."""
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
        data = data_dict[b'data']
        labels = data_dict[b'labels']
        data = data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # Reshape to (N, 32, 32, 3)
        labels = np.array(labels)
        return data, labels

def test_cifar10(data_dir="data/cifar10"):
    """Test the CIFAR-10 dataset."""
    extracted_dir = os.path.join(data_dir, "cifar-10-batches-py")
    if not os.path.exists(extracted_dir):
        print("Dataset not found. Please initialize the dataset first.")
        return

    # Load batch 1 as a sample
    batch_file = os.path.join(extracted_dir, "data_batch_1")
    if not os.path.exists(batch_file):
        print("Data batch file not found.")
        return
    
    data, labels = load_cifar10_batch(batch_file)
    print(f"Loaded {data.shape[0]} samples from batch 1.")
    print(f"Image shape: {data[0].shape}")
    print(f"Number of classes: {len(CLASS_NAMES)}")
    
    # Display some random images with proper labels
    print("Displaying a few random images...")
    for i in range(5):
        index = np.random.randint(0, data.shape[0])
        image = data[index]
        label = CLASS_NAMES[labels[index]]
        plt.imshow(image)
        plt.title(f"Label: {label}")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    test_cifar10()
