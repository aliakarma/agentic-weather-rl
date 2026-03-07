"""
Multi-Modal Weather Encoder
=============================
Purpose:
    Fuses radar and satellite image streams into a unified weather state
    representation. Accepts separate radar and satellite tensors, encodes
    each through a shared or independent backbone, then combines them via
    a fusion layer to produce a single structured WeatherPrediction output.

Input:
    radar_imgs:     Tensor (B, C_r, H, W) — normalized radar reflectivity frames
    satellite_imgs: Tensor (B, C_s, H, W) — normalized satellite imagery

Output:
    WeatherPrediction with storm_probability, rainfall_intensity, flood_risk_score

Example usage:
    encoder = MultiModalWeatherEncoder(
        backbone='vit',
        radar_channels=1,
        satellite_channels=3,
    )
    radar = torch.randn(4, 1, 224, 224)
    satellite = torch.randn(4, 3, 224, 224)
    preds = encoder(radar, satellite)
    state = encoder.to_rl_state(preds, regional_risk=0.4)
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Literal, Dict

from models.cnn_weather_model import CNNWeatherModel, WeatherPrediction
from models.transformer_weather_model import ViTWeatherModel


# ---------------------------------------------------------------------------
# RL state vector dataclass
# ---------------------------------------------------------------------------

@dataclass
class RLWeatherState:
    """
    Four-dimensional state vector used as input to the RL decision agent.

    Attributes:
        storm_probability:    Probability of a severe storm event [0, 1].
        rainfall_intensity:   Normalized rainfall level [0, 1].
        flood_risk_score:     Estimated flood risk score [0, 1].
        regional_risk_indicator: External contextual risk factor [0, 1].
        as_tensor:            Flat tensor of shape (B, 4) suitable for the RL env.
    """
    storm_probability: torch.Tensor
    rainfall_intensity: torch.Tensor
    flood_risk_score: torch.Tensor
    regional_risk_indicator: torch.Tensor

    def as_tensor(self) -> torch.Tensor:
        """Stack all fields into a (B, 4) float32 tensor."""
        return torch.stack(
            [
                self.storm_probability,
                self.rainfall_intensity,
                self.flood_risk_score,
                self.regional_risk_indicator,
            ],
            dim=-1,
        ).float()

    def to_numpy(self):
        """Return a numpy array of shape (B, 4) for use with gym environments."""
        return self.as_tensor().cpu().detach().numpy()


# ---------------------------------------------------------------------------
# Single-stream branch encoder
# ---------------------------------------------------------------------------

class _StreamEncoder(nn.Module):
    """
    Thin wrapper that encodes a single image stream (radar OR satellite)
    through either a CNN or ViT backbone and returns a feature embedding.

    Args:
        backbone:    'cnn' or 'vit'.
        in_channels: Number of input channels for this stream.
        embed_dim:   Output embedding dimension.
        pretrained:  Use ImageNet-pretrained weights.
    """

    def __init__(
        self,
        backbone: Literal["cnn", "vit"] = "cnn",
        in_channels: int = 3,
        embed_dim: int = 128,
        pretrained: bool = False,
    ) -> None:
        super().__init__()

        if backbone == "cnn":
            self.model = CNNWeatherModel(
                backbone="resnet50",
                pretrained=pretrained,
                freeze_backbone=False,
                in_channels=in_channels,
            )
            # Replace weather head with an embedding projector
            self.model.weather_head = nn.Sequential(
                nn.Linear(2048, embed_dim),
                nn.ReLU(),
            )
        elif backbone == "vit":
            self.model = ViTWeatherModel(
                model_name="vit_custom_tiny",
                pretrained=False,
                in_channels=in_channels,
            )
            vit_dim = self.model.vit.num_features
            self.model.weather_head = nn.Sequential(
                nn.LayerNorm(vit_dim),
                nn.Linear(vit_dim, embed_dim),
                nn.GELU(),
            )
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.backbone_type = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a single image stream.

        Args:
            x: Tensor (B, C, H, W).

        Returns:
            Feature embedding (B, embed_dim).
        """
        if self.backbone_type == "cnn":
            feats = self.model._extract_features(x)
            return self.model.weather_head(feats)
        else:
            if hasattr(self.model.vit, "forward_features"):
                feats = self.model.vit.forward_features(x)
                if feats.ndim == 3:
                    feats = feats[:, 0]
            else:
                feats = self.model.vit(x)
            return self.model.weather_head(feats)


# ---------------------------------------------------------------------------
# Multi-modal fusion encoder
# ---------------------------------------------------------------------------

class MultiModalWeatherEncoder(nn.Module):
    """
    Two-stream multi-modal encoder that fuses radar and satellite features.

    Architecture:
        1. Radar stream  → _StreamEncoder → embed_dim features
        2. Satellite stream → _StreamEncoder → embed_dim features
        3. Concatenate both streams → (2 * embed_dim,)
        4. Fusion MLP → 3 weather state scalars
        5. Sigmoid → [0, 1] output per scalar

    Args:
        backbone:            Backbone type for both streams ('cnn' or 'vit').
        radar_channels:      Number of input channels for radar stream.
        satellite_channels:  Number of input channels for satellite stream.
        embed_dim:           Feature embedding size per stream.
        pretrained:          Use pretrained weights (CNN only).
        dropout_rate:        Dropout in fusion head.
    """

    def __init__(
        self,
        backbone: Literal["cnn", "vit"] = "vit",
        radar_channels: int = 1,
        satellite_channels: int = 3,
        embed_dim: int = 128,
        pretrained: bool = False,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()

        self.radar_encoder = _StreamEncoder(
            backbone=backbone,
            in_channels=radar_channels,
            embed_dim=embed_dim,
            pretrained=pretrained,
        )
        self.satellite_encoder = _StreamEncoder(
            backbone=backbone,
            in_channels=satellite_channels,
            embed_dim=embed_dim,
            pretrained=pretrained,
        )

        fused_dim = 2 * embed_dim

        # Fusion MLP
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 64),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64, 3),   # storm_prob, rainfall_intensity, flood_risk
        )
        self.sigmoid = nn.Sigmoid()

    def forward(
        self,
        radar_imgs: torch.Tensor,
        satellite_imgs: torch.Tensor,
    ) -> WeatherPrediction:
        """
        Fuse radar and satellite inputs and predict weather state.

        Args:
            radar_imgs:     Tensor (B, C_r, H, W).
            satellite_imgs: Tensor (B, C_s, H, W).

        Returns:
            WeatherPrediction with per-sample scores in [0, 1].
        """
        radar_feats = self.radar_encoder(radar_imgs)          # (B, embed_dim)
        sat_feats = self.satellite_encoder(satellite_imgs)    # (B, embed_dim)
        fused = torch.cat([radar_feats, sat_feats], dim=-1)   # (B, 2*embed_dim)

        logits = self.fusion_head(fused)    # (B, 3)
        outputs = self.sigmoid(logits)

        return WeatherPrediction(
            storm_probability=outputs[:, 0],
            rainfall_intensity=outputs[:, 1],
            flood_risk_score=outputs[:, 2],
            raw_logits=logits,
        )

    @staticmethod
    def to_rl_state(
        prediction: WeatherPrediction,
        regional_risk: float = 0.0,
    ) -> RLWeatherState:
        """
        Convert a WeatherPrediction into an RLWeatherState by appending
        a contextual regional risk indicator.

        Args:
            prediction:    WeatherPrediction from forward().
            regional_risk: External scalar in [0, 1] representing regional
                           vulnerability factors (e.g., population density,
                           historical flood frequency).

        Returns:
            RLWeatherState with all four RL state dimensions populated.
        """
        B = prediction.storm_probability.shape[0]
        risk_tensor = torch.full(
            (B,),
            fill_value=regional_risk,
            dtype=torch.float32,
            device=prediction.storm_probability.device,
        )
        return RLWeatherState(
            storm_probability=prediction.storm_probability.detach(),
            rainfall_intensity=prediction.rainfall_intensity.detach(),
            flood_risk_score=prediction.flood_risk_score.detach(),
            regional_risk_indicator=risk_tensor,
        )

    def encode_to_state_vector(
        self,
        radar_imgs: torch.Tensor,
        satellite_imgs: torch.Tensor,
        regional_risk: float = 0.0,
    ) -> "RLWeatherState":
        """
        Convenience method: encode images directly to an RLWeatherState.

        Args:
            radar_imgs:     Radar input tensor (B, C_r, H, W).
            satellite_imgs: Satellite input tensor (B, C_s, H, W).
            regional_risk:  Scalar regional context value.

        Returns:
            RLWeatherState ready for the RL environment step().
        """
        with torch.no_grad():
            prediction = self.forward(radar_imgs, satellite_imgs)
        return self.to_rl_state(prediction, regional_risk)

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_encoder(config: Dict) -> MultiModalWeatherEncoder:
    """
    Build a MultiModalWeatherEncoder from a configuration dictionary.

    Args:
        config: Dictionary with optional keys matching __init__ arguments.
                Missing keys use defaults.

    Returns:
        Configured MultiModalWeatherEncoder instance.

    Example:
        cfg = {"backbone": "vit", "embed_dim": 256, "pretrained": False}
        encoder = build_encoder(cfg)
    """
    return MultiModalWeatherEncoder(
        backbone=config.get("backbone", "vit"),
        radar_channels=config.get("radar_channels", 1),
        satellite_channels=config.get("satellite_channels", 3),
        embed_dim=config.get("embed_dim", 128),
        pretrained=config.get("pretrained", False),
        dropout_rate=config.get("dropout_rate", 0.2),
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Multi-Modal Encoder Smoke Test ===")
    encoder = MultiModalWeatherEncoder(backbone="vit", radar_channels=1, satellite_channels=3)
    radar = torch.randn(2, 1, 224, 224)
    satellite = torch.randn(2, 3, 224, 224)
    preds = encoder(radar, satellite)
    state = MultiModalWeatherEncoder.to_rl_state(preds, regional_risk=0.3)
    print(f"RL state tensor: {state.as_tensor()}")
    print(f"Numpy state: {state.to_numpy()}")
    print(f"Total params: {encoder.count_parameters():,}")
    print("Smoke test passed.")
