"""
Weather Perception Model Training Script
==========================================
Purpose:
    Trains the multi-modal weather perception model on radar and satellite
    imagery. Supports both CNN (ResNet50/VGG16) and ViT backbones with
    optional transfer learning. Outputs a trained checkpoint, training
    curves, and a performance summary.

Output:
    results/best_perception_model.pth  — best model checkpoint
    results/weather_model_accuracy.png — training/validation curves
    results/perception_training_log.csv — per-epoch metric log

Usage:
    python experiments/train_weather_model.py --model vit --epochs 30 --seed 42
"""

import os
import sys
import csv
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from typing import Tuple, Optional

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_weather_model import CNNWeatherModel, WeatherPredictionLoss
from models.transformer_weather_model import ViTWeatherModel
from models.multimodal_encoder import MultiModalWeatherEncoder


# ---------------------------------------------------------------------------
# Synthetic dataset (used when real data is unavailable)
# ---------------------------------------------------------------------------

class SyntheticWeatherDataset(Dataset):
    """
    Synthetic dataset that generates (radar_img, satellite_img, labels) tuples.

    Labels are 3-dimensional float vectors: [storm_prob, rainfall, flood_risk].
    Samples are generated deterministically from a fixed seed for reproducibility.

    Args:
        n_samples:           Number of synthetic samples.
        img_size:            Spatial resolution of generated images.
        radar_channels:      Number of radar input channels.
        satellite_channels:  Number of satellite input channels.
        seed:                Random seed.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        img_size: int = 64,
        radar_channels: int = 1,
        satellite_channels: int = 3,
        seed: int = 42,
    ) -> None:
        self.n_samples = n_samples
        self.img_size = img_size
        self.radar_channels = radar_channels
        self.satellite_channels = satellite_channels
        self.rng = np.random.default_rng(seed=seed)

        # Pre-generate all samples
        self.radar_data = self.rng.uniform(
            0.0, 1.0,
            size=(n_samples, radar_channels, img_size, img_size),
        ).astype(np.float32)
        self.sat_data = self.rng.uniform(
            0.0, 1.0,
            size=(n_samples, satellite_channels, img_size, img_size),
        ).astype(np.float32)

        # Labels derived from image statistics (simulates correlation)
        self.labels = np.column_stack([
            self.radar_data.mean(axis=(1, 2, 3)),          # storm_prob ~ radar mean
            self.sat_data[:, 0].mean(axis=(1, 2)),         # rainfall ~ sat channel 0
            0.5 * (self.radar_data.mean(axis=(1, 2, 3))   # flood_risk ~ combination
                   + self.sat_data.mean(axis=(1, 2, 3))),
        ]).astype(np.float32)

        # Normalize labels to [0, 1]
        for col in range(self.labels.shape[1]):
            col_min = self.labels[:, col].min()
            col_max = self.labels[:, col].max()
            self.labels[:, col] = (self.labels[:, col] - col_min) / (col_max - col_min + 1e-8)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        radar = torch.from_numpy(self.radar_data[idx])
        sat = torch.from_numpy(self.sat_data[idx])
        label = torch.from_numpy(self.labels[idx])
        return radar, sat, label


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seed for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(
    model_type: str,
    radar_channels: int,
    satellite_channels: int,
    pretrained: bool,
    freeze_backbone: bool,
) -> nn.Module:
    """
    Instantiate the perception model based on the CLI argument.

    Args:
        model_type:          'cnn', 'vit', or 'multimodal'.
        radar_channels:      Channels in radar input.
        satellite_channels:  Channels in satellite input.
        pretrained:          Load ImageNet weights (CNN only).
        freeze_backbone:     Freeze backbone during initial training.

    Returns:
        Configured PyTorch model.
    """
    if model_type == "cnn":
        model = CNNWeatherModel(
            backbone="resnet50",
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            in_channels=radar_channels + satellite_channels,
        )
    elif model_type == "vit":
        model = ViTWeatherModel(
            model_name="vit_custom_tiny",
            pretrained=False,
            freeze_backbone=freeze_backbone,
            in_channels=radar_channels + satellite_channels,
        )
    elif model_type == "multimodal":
        model = MultiModalWeatherEncoder(
            backbone="vit",
            radar_channels=radar_channels,
            satellite_channels=satellite_channels,
            embed_dim=128,
            pretrained=pretrained,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    print(f"[INFO] Model: {model_type}, "
          f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    model_type: str,
) -> float:
    """
    Run one training epoch and return mean loss.

    Args:
        model:      Perception model.
        loader:     Training DataLoader.
        optimizer:  Optimizer.
        criterion:  Loss function.
        device:     Compute device.
        model_type: 'cnn', 'vit', or 'multimodal'.

    Returns:
        Mean loss over all batches.
    """
    model.train()
    total_loss = 0.0

    for radar, sat, labels in loader:
        radar = radar.to(device)
        sat = sat.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if model_type == "multimodal":
            predictions = model(radar, sat)
        else:
            # Concatenate radar and satellite along the channel dimension
            inputs = torch.cat([radar, sat], dim=1)
            predictions = model(inputs)

        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    model_type: str,
) -> Tuple[float, float]:
    """
    Evaluate model on validation data.

    Returns:
        (mean_loss, accuracy) where accuracy = fraction of samples with
        MSE below 0.1 threshold (rough surrogate metric for this regression task).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for radar, sat, labels in loader:
            radar = radar.to(device)
            sat = sat.to(device)
            labels = labels.to(device)

            if model_type == "multimodal":
                predictions = model(radar, sat)
            else:
                inputs = torch.cat([radar, sat], dim=1)
                predictions = model(inputs)

            loss = criterion(predictions, labels)
            total_loss += loss.item()

            # Accuracy: prediction within 0.15 of target (per-sample all outputs)
            pred_tensor = torch.stack([
                predictions.storm_probability,
                predictions.rainfall_intensity,
                predictions.flood_risk_score,
            ], dim=1)
            within_threshold = (pred_tensor - labels).abs().max(dim=1).values < 0.15
            correct += within_threshold.sum().item()
            total += labels.size(0)

    mean_loss = total_loss / max(len(loader), 1)
    accuracy = correct / max(total, 1)
    return mean_loss, accuracy


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    val_accuracies: list,
    save_path: str,
) -> None:
    """Save training/validation loss and accuracy curves to a PNG file."""
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, label="Train Loss", color="royalblue")
    ax1.plot(epochs, val_losses, label="Val Loss", color="tomato")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_accuracies, color="seagreen", label="Val Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy (threshold=0.15)")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[INFO] Training curves saved to {save_path}")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    model_type: str = "vit",
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-4,
    seed: int = 42,
    n_samples: int = 2000,
    img_size: int = 64,
    output_dir: str = "results",
    pretrained: bool = False,
    freeze_backbone: bool = False,
) -> None:
    """
    Full training pipeline for the weather perception model.

    Args:
        model_type:      'cnn', 'vit', or 'multimodal'.
        epochs:          Number of training epochs.
        batch_size:      Batch size.
        lr:              Learning rate.
        seed:            Random seed.
        n_samples:       Number of synthetic training samples.
        img_size:        Spatial resolution for synthetic images.
        output_dir:      Directory to save outputs.
        pretrained:      Use pretrained backbone.
        freeze_backbone: Freeze backbone during training.
    """
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ----- Dataset -----
    dataset = SyntheticWeatherDataset(
        n_samples=n_samples, img_size=img_size, seed=seed
    )
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"[INFO] Train: {train_size}, Val: {val_size}")

    # ----- Model -----
    model = build_model(
        model_type, radar_channels=1, satellite_channels=3,
        pretrained=pretrained, freeze_backbone=freeze_backbone,
    ).to(device)

    # ----- Optimizer and loss -----
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = WeatherPredictionLoss(weights=(1.0, 1.0, 1.0))

    # ----- Training loop -----
    train_losses, val_losses, val_accuracies = [], [], []
    best_val_loss = float("inf")
    log_rows = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, model_type)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, model_type)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        log_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        })

        print(f"Epoch {epoch:>3}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),
                       os.path.join(output_dir, "best_perception_model.pth"))

    # ----- Save results -----
    plot_training_curves(
        train_losses, val_losses, val_accuracies,
        save_path=os.path.join(output_dir, "weather_model_accuracy.png"),
    )

    log_path = os.path.join(output_dir, "perception_training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_accuracy"])
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"[INFO] Training log saved to {log_path}")
    print(f"[INFO] Best val loss: {best_val_loss:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the weather perception model.")
    parser.add_argument("--model", type=str, default="vit",
                        choices=["cnn", "vit", "multimodal"],
                        help="Model architecture.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        n_samples=args.n_samples,
        img_size=args.img_size,
        output_dir=args.output_dir,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
    )
