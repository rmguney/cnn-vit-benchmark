import torch
from torchvision import datasets, transforms
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import CSVLogger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import os
import time
import numpy as np
from transformers import AutoModelForImageClassification
from torchvision.models import efficientnet_b0
import json

class BenchmarkModel(LightningModule):
    def __init__(self, model, test_loader, model_name):
        super().__init__()
        self.model = model
        self.test_loader = test_loader
        self.model_name = model_name
        self.results = {}
        self.all_outputs = []  # Use this to collect outputs

    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self.model(images)
        preds = torch.argmax(outputs, dim=1)
        probabilities = torch.softmax(outputs, dim=1)  # Get predicted probabilities for mAP

        # Collect outputs
        self.all_outputs.append({"preds": preds.cpu(), "labels": labels.cpu(), "probs": probabilities.cpu()})

        return {"preds": preds.cpu(), "labels": labels.cpu(), "probs": probabilities.cpu()}

    def on_test_epoch_end(self):
        # Combine all outputs
        all_preds = torch.cat([x["preds"] for x in self.all_outputs], dim=0)
        all_labels = torch.cat([x["labels"] for x in self.all_outputs], dim=0)
        all_probs = torch.cat([x["probs"] for x in self.all_outputs], dim=0)

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        top5_accuracy = None
        if all_probs.size(1) >= 5:
            top5_accuracy = (all_labels.unsqueeze(1) == torch.topk(all_probs, k=5, dim=1)[1]).sum().item() / len(all_labels)

        precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        conf_matrix = confusion_matrix(all_labels, all_preds)

        # Class-wise metrics
        class_report = classification_report(all_labels, all_preds, target_names=[f"Class {i}" for i in range(10)])

        # Calculate mAP
        map_score = self.calculate_map(all_probs.numpy(), all_labels.numpy(), num_classes=10)

        # Store results
        self.results = {
            "model": self.model_name,
            "accuracy": accuracy,
            "top5_accuracy": top5_accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": conf_matrix.tolist(),
            "class_report": class_report,
            "mAP": map_score,
        }

        # Print results
        print(f"\nResults for {self.model_name}:")
        print(f"Accuracy: {accuracy:.4f}")
        if top5_accuracy is not None:
            print(f"Top-5 Accuracy: {top5_accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"mAP: {map_score:.4f}")
        print(f"Confusion Matrix:\n{conf_matrix}")
        print(f"Class-Wise Metrics:\n{class_report}")

    def calculate_map(self, probs, labels, num_classes):
        """
        Calculate Mean Average Precision (mAP) for single-label classification.
        :param probs: Predicted probabilities (N x C array).
        :param labels: True labels (N array).
        :param num_classes: Number of classes.
        :return: Mean Average Precision (mAP).
        """
        aps = []
        for c in range(num_classes):
            # Get true binary labels and predicted probabilities for class c
            true_labels = (labels == c).astype(int)
            predicted_probs = probs[:, c]

            if np.sum(true_labels) == 0:  # No positive samples for this class
                aps.append(0)
                continue

            # Sort by predicted probability
            sorted_indices = np.argsort(-predicted_probs)
            true_labels = true_labels[sorted_indices]
            predicted_probs = predicted_probs[sorted_indices]

            # Calculate precision-recall curve
            tp = np.cumsum(true_labels)
            fp = np.cumsum(1 - true_labels)
            precision = tp / (tp + fp + 1e-7)
            recall = tp / (np.sum(true_labels) + 1e-7)

            # Calculate AP for this class
            ap = np.sum(precision[1:] * np.diff(recall))
            aps.append(ap)

        return np.mean(aps)

    def benchmark_speed(self, device, input_size=None, num_runs=100):
        """
        Benchmark the latency and throughput of the model.
        :param device: Device to run the benchmark (CPU or GPU).
        :param input_size: Tuple specifying the input size (default: (1, 3, 224, 224)).
        :param num_runs: Number of runs for the benchmark.
        """
        self.model.eval()

        # Determine the input size
        if input_size is None:
            input_size = (1, 3, 224, 224)

        dummy_input = torch.randn(input_size).to(device)

        # Warm-up runs
        for _ in range(10):
            with torch.no_grad():
                self.model(dummy_input)

        # Measure latency
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                self.model(dummy_input)
        end_time = time.time()

        avg_latency = (end_time - start_time) / num_runs
        throughput = 1 / avg_latency

        self.results["latency"] = avg_latency
        self.results["throughput"] = throughput

        print(f"Latency: {avg_latency:.4f}s, Throughput: {throughput:.2f} images/s")


# Load CIFAR-10 test dataset
def load_test_dataset(batch_size=32, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)
    return test_loader


# Main benchmarking function
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_loader = load_test_dataset()

    # Define DeiT-T architecture
    class DeiTModel(torch.nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.model = AutoModelForImageClassification.from_pretrained(
                "facebook/deit-tiny-patch16-224",
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )

        def forward(self, x):
            return self.model(x).logits  # Extract logits from Hugging Face model output

    # Define EfficientNet-B0 architecture
    class EfficientNetB0Model(torch.nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.model = efficientnet_b0(pretrained=True)
            self.model.classifier[1] = torch.nn.Linear(self.model.classifier[1].in_features, num_classes)

        def forward(self, x):
            return self.model(x)

    deit_model = DeiTModel(num_classes=10)
    efficientnet_model = EfficientNetB0Model(num_classes=10)

    # Load state_dict from checkpoint
    deit_model_path = './saved_models/DeiTTinyForClassification_2024-11-27_22-38-14_best.ckpt'
    efficientnet_model_path = './saved_models/EfficientNetB0ForClassification_2024-12-21_10-47-18_best.ckpt'

    deit_checkpoint = torch.load(deit_model_path, map_location=device)
    efficientnet_checkpoint = torch.load(efficientnet_model_path, map_location=device)

    # Load weights into the models
    deit_model.load_state_dict(deit_checkpoint["state_dict"])
    efficientnet_model.load_state_dict(efficientnet_checkpoint["state_dict"])

    # Send models to device
    deit_model.to(device)
    efficientnet_model.to(device)

    # Initialize CSV Logger
    logger = CSVLogger(save_dir="./logs", name="benchmark_logs")

    # Benchmark DeiT
    print("\nBenchmarking DeiT...")
    deit_benchmark = BenchmarkModel(deit_model, test_loader, "DeiT-T")
    trainer = Trainer(logger=logger, accelerator=device, devices=1, max_epochs=1)
    trainer.test(deit_benchmark, test_loader)
    deit_benchmark.benchmark_speed(device)

    # Benchmark EfficientNet-B0
    print("\nBenchmarking EfficientNet-B0...")
    efficientnet_benchmark = BenchmarkModel(efficientnet_model, test_loader, "EfficientNet-B0")
    trainer.test(efficientnet_benchmark, test_loader)
    efficientnet_benchmark.benchmark_speed(device)

    # Save results to a human-readable file
    results_path = "./logs/extended_benchmark_results.txt"
    with open(results_path, "w") as f:
        for result in [deit_benchmark.results, efficientnet_benchmark.results]:
            f.write(f"Model: {result['model']}\n")
            f.write(f"Accuracy: {result['accuracy']:.4f}\n")
            if result['top5_accuracy'] is not None:
                f.write(f"Top-5 Accuracy: {result['top5_accuracy']:.4f}\n")
            f.write(f"Precision: {result['precision']:.4f}\n")
            f.write(f"Recall: {result['recall']:.4f}\n")
            f.write(f"F1 Score: {result['f1_score']:.4f}\n")
            f.write(f"mAP: {result['mAP']:.4f}\n")
            f.write(f"Latency: {result['latency']:.4f}s\n")
            f.write(f"Throughput: {result['throughput']:.2f} images/s\n")
            f.write(f"Confusion Matrix:\n{result['confusion_matrix']}\n")
            f.write(f"Class-Wise Metrics:\n{result['class_report']}\n")
            f.write("\n")

    print(f"\nBenchmark results saved to: {results_path}")

if __name__ == '__main__':
    main()
