import os
import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
import datetime
from torchmetrics.classification import (
    MulticlassPrecision, MulticlassRecall, MulticlassF1Score,
    MulticlassAveragePrecision, MulticlassConfusionMatrix
)
import numpy as np
from pytorch_lightning import seed_everything
import time
from network import MobileNetV2

# Dynamically resolve the path to the weights file
weights_path = os.path.join(os.path.dirname(__file__), "../../saved_models/weights.pkl")
weights_path = os.path.abspath(weights_path)

class MobileNetV2CIFAR10(pl.LightningModule):
    def __init__(self, num_classes=10, learning_rate=0.1):
        super().__init__()
        self.save_hyperparameters()

        # Load MobileNetV2 architecture
        self.model = MobileNetV2(output_size=num_classes, alpha=1)

        # Load pretrained weights from the repository
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict)

        # Loss function
        self.criterion = torch.nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

        # Metrics
        self.precision = MulticlassPrecision(num_classes=num_classes, average=None)
        self.recall = MulticlassRecall(num_classes=num_classes, average=None)
        self.f1 = MulticlassF1Score(num_classes=num_classes, average=None)
        self.map = MulticlassAveragePrecision(num_classes=num_classes)
        self.confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)

        # Gradient norms per epoch
        self.grad_norm_values = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)

        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        precision = self.precision(preds, y)
        recall = self.recall(preds, y)
        f1 = self.f1(preds, y)
        ap = self.map(logits, y)

        # Log metrics
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.log("train_precision", precision.mean(), on_step=False, on_epoch=True)
        self.log("train_recall", recall.mean(), on_step=False, on_epoch=True)
        self.log("train_f1", f1.mean(), on_step=False, on_epoch=True)
        self.log("train_map", ap.mean(), on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)

        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        precision = self.precision(preds, y)
        recall = self.recall(preds, y)
        f1 = self.f1(preds, y)
        ap = self.map(logits, y)

        # Ensure AP is iterable and handle NaN values
        if ap.dim() == 0:
            ap = ap.unsqueeze(0)
        ap = torch.nan_to_num(ap, nan=0.0)

        confusion_matrix = self.confusion_matrix(preds, y).cpu().numpy()

        # Derive FP/FN from confusion matrix
        false_positives = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
        false_negatives = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)

        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_precision", precision.mean(), on_step=False, on_epoch=True)
        self.log("val_recall", recall.mean(), on_step=False, on_epoch=True)
        self.log("val_f1", f1.mean(), on_step=False, on_epoch=True)
        self.log("val_map", ap.mean(), on_step=False, on_epoch=True)

        # Log per-class metrics
        for i, (fp, fn, class_ap) in enumerate(zip(false_positives, false_negatives, ap)):
            self.log(f"class_{i}_fp", fp, on_step=False, on_epoch=True)
            self.log(f"class_{i}_fn", fn, on_step=False, on_epoch=True)
            self.log(f"class_{i}_ap", class_ap, on_step=False, on_epoch=True)

        return loss

    def on_before_backward(self, loss):
        # Track gradient norms
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        self.grad_norm_values.append(total_norm)
        self.log("grad_norm", total_norm, on_step=True, on_epoch=False)

    def on_train_epoch_end(self):
        # Log average gradient norm for the epoch
        if self.grad_norm_values:
            avg_grad_norm = np.mean(self.grad_norm_values)
            self.log("avg_grad_norm", avg_grad_norm, on_epoch=True)
            self.grad_norm_values = []

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=4e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
        return [optimizer], [scheduler]


# Prepare CIFAR-10 dataset
def prepare_data(data_dir="data/cifar10"):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    train_len = int(len(dataset) * 0.8)
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
    return train_dataset, val_dataset


def main():
    # Setup directories
    saved_models_dir = "saved_models"
    logs_dir = "logs"
    os.makedirs(saved_models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Read training parameters from environment variables, or use default values
    epochs = int(os.getenv("EPOCHS", 300))
    batch_size = int(os.getenv("BATCH_SIZE", 128))

    # Seed everything for reproducibility
    seed_everything(42, workers=True)

    # Dataset and DataLoader
    train_dataset, val_dataset = prepare_data()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4)

    # Initialize model
    model = MobileNetV2CIFAR10()

    # Callbacks
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_map",
        mode="max",
        dirpath=saved_models_dir,
        filename=f"MobileNetV2CIFAR10_{current_time}_best",
        save_top_k=1
    )
    checkpoint_callback_epoch = ModelCheckpoint(
        every_n_epochs=1,
        dirpath=saved_models_dir,
        filename=f"MobileNetV2CIFAR10_{current_time}_epoch{{epoch}}"
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    csv_logger = CSVLogger(logs_dir, name=f"MobileNetV2CIFAR10_{current_time}")

    # Trainer
    start_time = time.time()
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback, checkpoint_callback_epoch, lr_monitor],
        logger=csv_logger,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1
    )
    trainer.fit(model, train_loader, val_loader)
    print(f"Training completed in {(time.time() - start_time) / 3600:.2f} hours")


if __name__ == "__main__":
    main()
