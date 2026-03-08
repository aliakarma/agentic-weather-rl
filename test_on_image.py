"""
Test the trained weather model and RL agent on a real image.
Usage:
    python test_on_image.py --image "test image.jpg"
"""

import sys
import os
import argparse
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.cnn_weather_model import CNNWeatherModel
from rl_agent.agent_ppo import DisasterResponseAgent
from rl_agent.environment import DisasterResponseEnv

# ---- Action labels ----
ACTION_LABELS = {
    0: "No Action         (weather is safe)",
    1: "Issue Storm Warning",
    2: "Prepare Emergency Resources",
    3: "Recommend Evacuation (CRITICAL)",
}

RISK_LABELS = {
    0: "LOW",
    1: "MEDIUM",
    2: "HIGH",
    3: "CRITICAL",
}


def load_image(image_path: str, img_size: int = 64):
    """Load image and return both a tensor and raw numpy array for analysis."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((img_size, img_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0        # (H, W, 3) in [0,1]

    # Compute image-level weather signals from the photo:
    # Ground photos: dark + gray + high contrast = storm
    gray = arr.mean(axis=2)                               # (H, W)
    brightness   = float(gray.mean())                     # 0=dark, 1=bright
    contrast     = float(gray.std())                      # spread of tones
    blue_channel = float(arr[:, :, 2].mean())             # blue sky = high
    red_channel  = float(arr[:, :, 0].mean())
    # Greenness (green fields, not storm-related)
    colorfulness = float(np.abs(arr[:, :, 0] - arr[:, :, 2]).mean())

    arr_chw = arr.transpose(2, 0, 1)                      # (3, H, W)

    # Build a satellite-like IR cold-top signal:
    # Dark + low-saturation = storm cloud seen from ground → invert for satellite view
    # Satellite sees storm cloud tops as COLD (low temp) = bright white from above
    # Ground sees storm clouds as DARK gray
    # We simulate the cold-top IR signal by boosting dark, low-color areas
    coldtop_sim = (1.0 - gray) * (1.0 - colorfulness)    # dark + gray = storm
    coldtop_sim = np.clip(coldtop_sim * 2.0, 0.0, 1.0)   # amplify signal
    radar_channel = coldtop_sim[np.newaxis]               # (1, H, W)

    tensor = np.concatenate([radar_channel, arr_chw], axis=0).astype(np.float32)
    extra = {
        "brightness": brightness,
        "contrast": contrast,
        "blue_channel": blue_channel,
        "colorfulness": colorfulness,
    }
    return torch.from_numpy(tensor).unsqueeze(0), extra


def analyse_image_directly(extra: dict):
    """
    Derive weather scores directly from image visual properties.
    Works for ground-level photos (storm = dark, gray, high contrast).
    Returns (storm_prob, rainfall, flood_risk).
    """
    brightness   = extra["brightness"]
    contrast     = extra["contrast"]
    blue         = extra["blue_channel"]
    colorfulness = extra["colorfulness"]

    # Dark + gray (low colorfulness) + high contrast = severe storm
    darkness     = 1.0 - brightness                          # dark → storm
    grayness     = 1.0 - min(colorfulness * 4.0, 1.0)       # gray → storm
    turbulence   = min(contrast * 5.0, 1.0)                  # contrast → turbulence
    clearsky     = min(blue * 1.5, 1.0)                      # blue sky → clear

    storm_prob  = float(np.clip(darkness * 0.5 + grayness * 0.3 + turbulence * 0.2 - clearsky * 0.3, 0.0, 1.0))
    rainfall    = float(np.clip(storm_prob * 0.7 + turbulence * 0.3, 0.0, 1.0))
    flood_risk  = float(np.clip(storm_prob * 0.6 + rainfall * 0.4, 0.0, 1.0))
    return storm_prob, rainfall, flood_risk


def run_weather_model(image_tensor: torch.Tensor, extra: dict, model_path: str):
    """Run CNN model then blend with direct visual analysis for best results."""
    model = CNNWeatherModel(
        backbone="resnet50",
        pretrained=False,
        in_channels=4,
    )
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        preds = model(image_tensor)

    cnn_storm = preds.storm_probability.item()
    cnn_rain  = preds.rainfall_intensity.item()
    cnn_flood = preds.flood_risk_score.item()

    # Direct visual analysis (works well for ground photos)
    vis_storm, vis_rain, vis_flood = analyse_image_directly(extra)

    # Blend: 40% CNN + 60% visual analysis
    storm_prob = 0.4 * cnn_storm + 0.6 * vis_storm
    rainfall   = 0.4 * cnn_rain  + 0.6 * vis_rain
    flood_risk = 0.4 * cnn_flood + 0.6 * vis_flood
    return storm_prob, rainfall, flood_risk


def build_rl_observation(storm_prob, rainfall, flood_risk):
    """Build the 4-dim observation vector expected by the RL environment."""
    regional_risk = (storm_prob * 0.4 + rainfall * 0.3 + flood_risk * 0.3)
    obs = np.array([
        storm_prob,
        rainfall,
        flood_risk,
        regional_risk,
    ], dtype=np.float32)
    return np.clip(obs, 0.0, 1.0)


def run_rl_agent(obs: np.ndarray, model_path: str):
    """Load trained RL agent and predict action."""
    env = DisasterResponseEnv(max_steps=50, seed=42)
    agent = DisasterResponseAgent.load(model_path, env)
    action, _ = agent.predict(obs, deterministic=True)
    return int(action)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="test image.jpg")
    parser.add_argument("--weather_model", type=str, default="results/best_perception_model.pth")
    parser.add_argument("--rl_model", type=str, default="results/ppo_agent")
    parser.add_argument("--img_size", type=int, default=64)
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  WEATHER RL — IMAGE PREDICTION TEST")
    print("=" * 55)
    print(f"  Image : {args.image}")

    # Step 1: Load image
    print("\n[1/3] Loading image...")
    tensor, extra = load_image(args.image, args.img_size)
    print(f"      Shape: {tuple(tensor.shape)}  (batch=1, channels=4, {args.img_size}x{args.img_size})")

    # Step 2: Weather model prediction
    print("\n[2/3] Running weather perception model...")
    storm_prob, rainfall, flood_risk = run_weather_model(tensor, extra, args.weather_model)
    avg_risk = (storm_prob + rainfall + flood_risk) / 3.0
    risk_level = int(min(3, avg_risk * 4))

    print(f"\n  --- Weather Predictions ---")
    print(f"  Storm Probability   : {storm_prob:.2%}")
    print(f"  Rainfall Intensity  : {rainfall:.2%}")
    print(f"  Flood Risk Score    : {flood_risk:.2%}")
    print(f"  Overall Risk Level  : {RISK_LABELS[risk_level]}")

    # Step 3: RL agent decision
    print("\n[3/3] Running RL disaster response agent...")
    obs = build_rl_observation(storm_prob, rainfall, flood_risk)
    action = run_rl_agent(obs, args.rl_model)

    print(f"\n  --- Agent Decision ---")
    print(f"  Recommended Action  : {ACTION_LABELS[action]}")

    print("\n" + "=" * 55)
    print("  DONE")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
