"""
Vision Transformer Weather Perception Model
=============================================
Purpose:
    Implements a Vision Transformer (ViT) backbone for classifying severe weather
    conditions from multi-modal radar and satellite imagery. Uses fine-tuning
    via the `timm` library with pretrained ViT-B/16 or ViT-S/16 weights.

Input:
    Tensor of shape (B, C, H, W).

Output:
    WeatherPrediction dataclass with storm_probability, rainfall_intensity,
    and flood_risk_score (all in [0, 1]).

Example usage:
    model = ViTWeatherModel(model_name='vit_base_patch16_224', pretrained=True)
    imgs = torch.randn(4, 3, 224, 224)
    preds = model(imgs)
"""

import torch
import torch.nn as nn
from typing import Literal

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("[WARNING] timm not installed. Run: pip install timm")

from models.cnn_weather_model import WeatherPrediction, WeatherPredictionLoss  # noqa


# ---------------------------------------------------------------------------
# Supported ViT configurations
# ---------------------------------------------------------------------------

SUPPORTED_VIT_MODELS = {
    "vit_base_patch16_224": {"embed_dim": 768, "img_size": 224},
    "vit_small_patch16_224": {"embed_dim": 384, "img_size": 224},
    "vit_tiny_patch16_224": {"embed_dim": 192, "img_size": 224},
    # Lightweight fallback when timm is unavailable (custom implementation below)
    "vit_custom_tiny": {"embed_dim": 256, "img_size": 224},
}


# ---------------------------------------------------------------------------
# Fallback minimal ViT implementation (no timm dependency)
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """
    Split an image into non-overlapping patches and linearly project each patch.

    Args:
        img_size:    Input image size (assumes square images).
        patch_size:  Size of each patch.
        in_channels: Number of input channels.
        embed_dim:   Embedding dimension per patch.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 256,
    ) -> None:
        super().__init__()
        n_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.n_patches = n_patches
        # Equivalent to a convolution that tiles non-overlapping patches
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, n_patches, embed_dim)
        x = self.projection(x)            # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)                  # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)            # (B, n_patches, embed_dim)
        return x


class MinimalViT(nn.Module):
    """
    Minimal ViT implementation for environments where timm is unavailable.

    This provides the correct interface without requiring pretrained weights.

    Args:
        img_size:    Spatial resolution of input images.
        patch_size:  Patch size (must divide img_size evenly).
        in_channels: Number of input spectral channels.
        embed_dim:   Transformer embedding dimension.
        depth:       Number of Transformer encoder layers.
        num_heads:   Number of attention heads.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        n_patches = (img_size // patch_size) ** 2

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        # Learnable [CLS] token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.num_features = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning the CLS token embedding.

        Args:
            x: (B, C, H, W) input tensor.

        Returns:
            Feature vector of shape (B, embed_dim).
        """
        B = x.size(0)
        tokens = self.patch_embed(x)  # (B, n_patches, D)
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embed
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        return tokens[:, 0]  # CLS token


# ---------------------------------------------------------------------------
# Main ViT model class
# ---------------------------------------------------------------------------

class ViTWeatherModel(nn.Module):
    """
    Vision Transformer-based multi-output weather prediction model.

    Prefers the `timm` library for pretrained ViT weights. Falls back to a
    lightweight custom ViT when timm is unavailable.

    Args:
        model_name:      ViT variant. See SUPPORTED_VIT_MODELS for options.
        pretrained:      Load pretrained ImageNet weights (requires timm).
        freeze_backbone: Freeze ViT encoder during initial training.
        in_channels:     Number of input spectral channels.
        dropout_rate:    Dropout in the classification head.
    """

    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        in_channels: int = 3,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        self.model_name = model_name

        # ----- Build ViT backbone -----
        if TIMM_AVAILABLE and model_name in SUPPORTED_VIT_MODELS and model_name != "vit_custom_tiny":
            self.vit = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,              # Remove classification head
                in_chans=in_channels,
            )
            embed_dim = self.vit.embed_dim
        else:
            if TIMM_AVAILABLE and model_name not in SUPPORTED_VIT_MODELS:
                print(f"[WARNING] '{model_name}' not in supported list. Using custom ViT.")
            elif not TIMM_AVAILABLE:
                print("[INFO] timm unavailable — using minimal ViT implementation.")
            self.vit = MinimalViT(
                img_size=224,
                patch_size=16,
                in_channels=in_channels,
                embed_dim=256,
                depth=4,
                num_heads=4,
            )
            embed_dim = self.vit.num_features

        # ----- Optional freeze -----
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

        # ----- Weather regression head -----
        self.weather_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, 3),  # [storm_prob, rainfall_intensity, flood_risk]
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> WeatherPrediction:
        """
        Forward pass producing structured weather state predictions.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            WeatherPrediction with per-sample scores in [0, 1].
        """
        if TIMM_AVAILABLE and hasattr(self.vit, "forward_features"):
            features = self.vit.forward_features(x)
            # timm ViT returns (B, n_tokens, D); take CLS token at index 0
            if features.ndim == 3:
                features = features[:, 0]
        else:
            features = self.vit(x)          # (B, embed_dim)

        logits = self.weather_head(features)  # (B, 3)
        outputs = self.sigmoid(logits)

        return WeatherPrediction(
            storm_probability=outputs[:, 0],
            rainfall_intensity=outputs[:, 1],
            flood_risk_score=outputs[:, 2],
            raw_logits=logits,
        )

    def unfreeze_backbone(self) -> None:
        """Unfreeze all ViT encoder parameters for full fine-tuning."""
        for param in self.vit.parameters():
            param.requires_grad = True

    def count_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== ViT Weather Model Smoke Test ===")
    model = ViTWeatherModel(model_name="vit_custom_tiny", pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    preds = model(dummy)
    print(f"storm_probability: {preds.storm_probability.detach()}")
    print(f"rainfall_intensity: {preds.rainfall_intensity.detach()}")
    print(f"flood_risk_score: {preds.flood_risk_score.detach()}")
    print(f"trainable params: {model.count_parameters():,}")
    print("Smoke test passed.")
