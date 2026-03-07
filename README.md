# Multi-Modal Reinforcement Learning for Autonomous Extreme Weather Emergency Response

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/agentic-weather-rl/blob/main/notebooks/demo_pipeline.ipynb)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange.svg)](https://pytorch.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.0-green.svg)](https://stable-baselines3.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This repository implements a three-layer multi-agent AI architecture for intelligent weather monitoring and automated emergency response to extreme weather events. The system integrates multi-modal environmental perception, reinforcement learning-based decision making, and agentic orchestration for automated emergency response execution.

The work supports the methodology described in:

> **"Multi-Modal Reinforcement Learning for Autonomous Extreme Weather Emergency Response"**

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              LAYER 1: Multi-Modal Perception             │
│                                                         │
│   Radar Data (NEXRAD) + Satellite Images (GOES/SEVIR)   │
│                         ↓                               │
│          Multi-Modal Encoder (CNN / ViT)                │
│                         ↓                               │
│    storm_probability | rainfall_intensity | flood_risk  │
└─────────────────────────────────┬───────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────┐
│           LAYER 2: Reinforcement Learning Agent          │
│                                                         │
│   State: [storm_prob, rainfall, flood_risk, region]     │
│                         ↓                               │
│           PPO Policy (Stable-Baselines3)                │
│                         ↓                               │
│   Action: No Action | Warning | Emergency | Evacuation  │
└─────────────────────────────────┬───────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────┐
│          LAYER 3: Agentic Orchestration Layer            │
│                                                         │
│   send_alert() | notify_emergency_services()            │
│   update_disaster_dashboard() | log_event()             │
│                         ↓                               │
│              Emergency Response Actions                  │
└─────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
weather-rl-emergency-system/
│
├── README.md
├── requirements.txt
├── environment.yml
├── LICENSE
│
├── architecture/
│   └── system_architecture.png       # Architecture diagram
│
├── data/
│   └── dataset_links.md              # Dataset download instructions
│
├── preprocessing/
│   ├── process_radar_data.py         # NEXRAD radar preprocessing
│   └── process_satellite_images.py   # GOES/SEVIR satellite preprocessing
│
├── models/
│   ├── cnn_weather_model.py          # CNN-based perception model
│   ├── transformer_weather_model.py  # ViT-based perception model
│   └── multimodal_encoder.py         # Unified multi-modal encoder
│
├── rl_agent/
│   ├── environment.py                # Custom Gym environment
│   ├── agent_ppo.py                  # PPO agent wrapper
│   └── training.py                   # RL training loop
│
├── orchestration/
│   └── emergency_action_simulator.py # Simulated emergency actions
│
├── experiments/
│   ├── train_weather_model.py        # Perception model training
│   ├── train_rl_agent.py             # RL agent training
│   └── evaluate_system.py            # End-to-end evaluation
│
├── results/
│   ├── reward_curve_example.png
│   ├── accuracy_plot_example.png
│   └── experiment_results_template.csv
│
└── notebooks/
    └── demo_pipeline.ipynb           # Full pipeline demonstration
```

---

## Installation

### Option 1: pip

```bash
git clone https://github.com/your-username/weather-rl-emergency-system.git
cd weather-rl-emergency-system
pip install -r requirements.txt
```

### Option 2: Conda environment

```bash
conda env create -f environment.yml
conda activate weather-rl
```

**Requirements:** Python 3.10, CUDA (optional but recommended)

---

## Datasets

### SEVIR — Storm Event Imagery

- **Description:** Temporally aligned radar, satellite, and lightning observations for thousands of storm events.
- **Download:** https://registry.opendata.aws/sevir/
- **Usage:** Perception model training (storm event classification)

### GOES — Geostationary Operational Environmental Satellite

- **Description:** Continuous geostationary satellite imagery of cloud formations and atmospheric structures.
- **Download:** https://registry.opendata.aws/noaa-goes/
- **Usage:** Cloud pattern and temperature structure input

### NOAA NEXRAD — Next Generation Weather Radar

- **Description:** High-resolution atmospheric reflectivity measurements for precipitation and storm dynamics.
- **Download:** https://registry.opendata.aws/noaa-nexrad/
- **Usage:** Radar reflectivity and rainfall intensity input

> See `data/dataset_links.md` for detailed download and setup instructions.

---

## Training

### Step 1: Preprocess Data

```bash
python preprocessing/process_radar_data.py --data_dir data/nexrad --output_dir data/processed
python preprocessing/process_satellite_images.py --data_dir data/goes --output_dir data/processed
```

### Step 2: Train the Perception Model

```bash
# Train with CNN backbone
python experiments/train_weather_model.py --model cnn --epochs 30 --batch_size 32

# Train with Vision Transformer
python experiments/train_weather_model.py --model vit --epochs 30 --batch_size 16
```

### Step 3: Train the RL Agent

```bash
python experiments/train_rl_agent.py --timesteps 100000 --perception_model results/best_perception_model.pth
```

### Step 4: Evaluate the System

```bash
python experiments/evaluate_system.py --model_path results/ppo_agent --output_dir results/
```

---

## Results

| Scenario | Decision Accuracy |
|---|---|
| Storm Warning Decision | 0.91 |
| Flood Risk Response | 0.87 |
| Evacuation Recommendation | 0.84 |

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| CNN Baseline | 0.82 | 0.80 | 0.81 | 0.80 |
| Vision Transformer (Fine-tuned) | **0.89** | **0.88** | **0.87** | **0.88** |

---

## Reproducibility

All experiments use fixed random seeds (default: `42`). To reproduce results:

```bash
python experiments/train_rl_agent.py --seed 42 --timesteps 100000
python experiments/evaluate_system.py --seed 42
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
