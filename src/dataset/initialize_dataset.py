import os
import tarfile
import urllib.request

def download_and_extract_cifar10(data_dir="data/cifar10"):
    """Download and extract CIFAR-10 dataset."""
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tar_path = os.path.join(data_dir, "cifar-10-python.tar.gz")
    extracted_dir = os.path.join(data_dir, "cifar-10-batches-py")
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    # Download CIFAR-10 dataset
    if not os.path.exists(tar_path):
        print("Downloading CIFAR-10 dataset...")
        urllib.request.urlretrieve(url, tar_path)
        print("Download complete.")
    else:
        print("CIFAR-10 dataset already downloaded.")

    # Extract the dataset
    if not os.path.exists(extracted_dir):
        print("Extracting CIFAR-10 dataset...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=data_dir)
        print("Extraction complete.")
    else:
        print("CIFAR-10 dataset already extracted.")

if __name__ == "__main__":
    download_and_extract_cifar10()
