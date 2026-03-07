"""
CNN-Based Weather Perception Model
=====================================
Purpose:
    Implements a convolutional neural network backbone (ResNet50 / VGG16) for
    classifying severe weather conditions from radar and satellite imagery.
    Supports transfer learning from ImageNet-pretrained weights.

Input:
    Tensor of shape (B, C, H, W) where:
      B = batch size
      C = number of input channels (default: 3 for RGB-like stacked inputs)
      H, W = spatial resolution (default: 224 x 224)

Output:
    WeatherPrediction dataclass with fields:
      storm_probability   (float, per sample)
      rainfall_intensity  (float, per sample)
      flood_risk_score    (float, per sample)

Example usage:
    model = CNNWeatherModel(backbone='resnet50', pretrained=True, freeze_backbone=True)
    imgs = torch.randn(4, 3, 224, 224)
    preds = model(imgs)
    print(preds.storm_probability)
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models
from dataclasses import dataclass
from typing import Literal


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class WeatherPrediction:
    """
    Structured output of the weather perception model.

    Attributes:
        storm_probability:  Predicted probability of a severe storm event [0, 1].
        rainfall_intensity: Predicted normalized rainfall intensity level [0, 1].
        flood_risk_score:   Predicted flood risk score [0, 1].
        raw_logits:         Raw model output tensor before sigmoid (shape: B x 3).
    """
    storm_probability: torch.Tensor
    rainfall_intensity: torch.Tensor
    flood_risk_score: torch.Tensor
    raw_logits: torch.Tensor


# ---------------------------------------------------------------------------
# CNN model
# ---------------------------------------------------------------------------

class CNNWeatherModel(nn.Module):
    """
    Transfer-learning-based CNN for multi-output weather state prediction.

    Supports ResNet50 and VGG16 backbones from torchvision. The final
    fully-connected head is replaced with a 3-output regression head that
    predicts storm probability, rainfall intensity, and flood risk.

    Args:
        backbone:        Architecture name. Options: 'resnet50', 'vgg16'.
        pretrained:      If True, load ImageNet-pretrained weights.
        freeze_backbone: If True, freeze all backbone layers (only train head).
        in_channels:     Number of input image channels (default: 3).
        dropout_rate:    Dropout probability applied in the classification head.
    """

    SUPPORTED_BACKBONES: tuple = ("resnet50", "vgg16")

    def __init__(
        self,
        backbone: Literal["resnet50", "vgg16"] = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        in_channels: int = 3,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. "
                f"Choose from: {self.SUPPORTED_BACKBONES}"
            )

        self.backbone_name = backbone
        self.in_channels = in_channels
        weights = "IMAGENET1K_V1" if pretrained else None

        # ----- Build backbone -----
        if backbone == "resnet50":
            base = tv_models.resnet50(weights=weights)
            # Adapt first conv layer if in_channels != 3
            if in_channels != 3:
                base.conv1 = nn.Conv2d(
                    in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
            feature_dim = base.fc.in_features
            base.fc = nn.Identity()  # Remove original classifier
            self.backbone = base

        elif backbone == "vgg16":
            base = tv_models.vgg16(weights=weights)
            if in_channels != 3:
                base.features[0] = nn.Conv2d(
                    in_channels, 64, kernel_size=3, padding=1
                )
            feature_dim = 4096
            base.classifier = nn.Identity()
            self._vgg_avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self._vgg_features = base.features
            self._vgg_flatten = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(p=dropout_rate),
            )
            self.backbone = base

        # ----- Freeze backbone weights if requested -----
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ----- Weather prediction head -----
        # 3 outputs: storm_prob, rainfall_intensity, flood_risk
        self.weather_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 3),  # 3 weather state outputs
        )

        # Sigmoid to constrain outputs to [0, 1]
        self.sigmoid = nn.Sigmoid()

    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the backbone feature extractor.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Feature tensor of shape (B, feature_dim).
        """
        if self.backbone_name == "resnet50":
            return self.backbone(x)
        elif self.backbone_name == "vgg16":
            feats = self._vgg_features(x)
            feats = self._vgg_avgpool(feats)
            feats = feats.view(feats.size(0), -1)
            feats = self._vgg_flatten(feats)
            return feats

    def forward(self, x: torch.Tensor) -> WeatherPrediction:
        """
        Full forward pass returning structured weather predictions.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            WeatherPrediction with per-sample scores.
        """
        features = self._extract_features(x)   # (B, feature_dim)
        logits = self.weather_head(features)    # (B, 3)
        outputs = self.sigmoid(logits)          # (B, 3) in [0, 1]

        return WeatherPrediction(
            storm_probability=outputs[:, 0],
            rainfall_intensity=outputs[:, 1],
            flood_risk_score=outputs[:, 2],
            raw_logits=logits,
        )

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

class WeatherPredictionLoss(nn.Module):
    """
    Combined MSE loss for the three weather prediction outputs.

    Supports optional per-output loss weighting to prioritize certain
    prediction targets during training.

    Args:
        weights: Tuple of (storm_weight, rainfall_weight, flood_weight).
                 Defaults to equal weighting (1.0, 1.0, 1.0).
    """

    def __init__(self, weights: tuple = (1.0, 1.0, 1.0)) -> None:
        super().__init__()
        self.weights = weights
        self.mse = nn.MSELoss()

    def forward(
        self,
        predictions: WeatherPrediction,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute weighted MSE loss.

        Args:
            predictions: WeatherPrediction output from model forward pass.
            targets:     Ground-truth tensor of shape (B, 3) in [0, 1].
                         Columns: [storm_prob, rainfall_intensity, flood_risk].

        Returns:
            Scalar loss tensor.
        """
        pred_stacked = torch.stack([
            predictions.storm_probability,
            predictions.rainfall_intensity,
            predictions.flood_risk_score,
        ], dim=1)  # (B, 3)

        loss = (
            self.weights[0] * self.mse(pred_stacked[:, 0], targets[:, 0])
            + self.weights[1] * self.mse(pred_stacked[:, 1], targets[:, 1])
            + self.weights[2] * self.mse(pred_stacked[:, 2], targets[:, 2])
        )
        return loss


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== CNN Weather Model Smoke Test ===")
    for arch in ("resnet50", "vgg16"):
        model = CNNWeatherModel(backbone=arch, pretrained=False)
        dummy = torch.randn(2, 3, 224, 224)
        preds = model(dummy)
        print(f"[{arch}] storm_prob: {preds.storm_probability.detach()}")
        print(f"[{arch}] trainable params: {model.count_parameters():,}")
    print("Smoke test passed.")
