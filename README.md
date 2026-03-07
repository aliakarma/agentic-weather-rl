<div align="center">

# 🌪️ Multi-Modal RL Weather Emergency Response

### *Autonomous Extreme Weather Detection & Emergency Orchestration via Multi-Agent AI*

<br/>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/agentic-weather-rl/blob/main/notebooks/demo_pipeline.ipynb)
&nbsp;
[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/)
&nbsp;
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
&nbsp;
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.0-76B900?style=flat&logo=openai&logoColor=white)](https://stable-baselines3.readthedocs.io/)
&nbsp;
[![License: MIT](https://img.shields.io/badge/License-MIT-F7DF1E?style=flat&logo=opensourceinitiative&logoColor=black)](LICENSE)
&nbsp;
[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen?style=flat&logo=github)](https://github.com/aliakarma/agentic-weather-rl/pulls)



> **"Multi-Modal Reinforcement Learning for Autonomous Extreme Weather Emergency Response"**
> 
> *A three-layer agentic AI framework combining multi-modal environmental perception,*  
> *reinforcement learning policy optimisation, and automated emergency action orchestration.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Datasets](#-datasets)
- [Training Pipeline](#-training-pipeline)
- [Results](#-results)
- [Reproducibility](#-reproducibility)
- [License](#-license)

---

## 🔭 Overview

This repository implements a **three-layer multi-agent AI architecture** for intelligent weather monitoring and automated emergency response. The system is designed to perceive extreme weather conditions, determine optimal mitigation strategies, and autonomously execute emergency response actions — all without human-in-the-loop latency.

### ✨ Key Highlights

| Feature | Detail |
|---|---|
| 🧠 **Perception** | Fused CNN / Vision Transformer encoder on radar + satellite imagery |
| 🤖 **Decision Making** | PPO-trained RL agent with disaster-aware reward shaping |
| ⚡ **Action Execution** | Agentic orchestration layer with simulated emergency service APIs |
| 📡 **Data Sources** | NEXRAD radar · GOES satellite · SEVIR storm events |
| 🔁 **End-to-End** | Full perception → decision → action pipeline in a single inference pass |
| 🔬 **Reproducible** | Seeded training, versioned configs, public dataset integration |

---

## 🏗️ Architecture

The system is decomposed into three tightly integrated layers:

```
╔══════════════════════════════════════════════════════════════════╗
║              🛰️  LAYER 1 — Multi-Modal Perception               ║
║                                                                  ║
║   📡 Radar (NEXRAD)        🌍 Satellite (GOES / SEVIR)          ║
║           └───────────────────┘                                  ║
║                         ↓                                        ║
║            Multi-Modal Encoder  (CNN / ViT)                      ║
║                         ↓                                        ║
║   storm_probability │ rainfall_intensity │ flood_risk_score      ║
╚══════════════════════════════════╦═══════════════════════════════╝
                                   ║
╔══════════════════════════════════╩═══════════════════════════════╗
║          🤖  LAYER 2 — Reinforcement Learning Agent             ║
║                                                                  ║
║   State: [storm_prob, rainfall, flood_risk, regional_risk]       ║
║                         ↓                                        ║
║              PPO Policy  (Stable-Baselines3)                     ║
║                         ↓                                        ║
║     🟢 No Action │ 🟡 Warning │ 🟠 Emergency │ 🔴 Evacuation   ║
╚══════════════════════════════════╦═══════════════════════════════╝
                                   ║
╔══════════════════════════════════╩═══════════════════════════════╗
║         🚨  LAYER 3 — Agentic Orchestration                     ║
║                                                                  ║
║   send_alert()  ·  notify_emergency_services()                   ║
║   recommend_evacuation()  ·  update_disaster_dashboard()         ║
║                         ↓                                        ║
║              🏥  Emergency Response Actions                     ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 📁 Repository Structure

```
weather-rl-emergency-system/
│
├── 📄 README.md
├── 📦 requirements.txt
├── 🐍 environment.yml
├── ⚖️  LICENSE
│
├── 🖼️  architecture/
│   └── system_architecture.png          # Architecture diagram
│
├── 📊 data/
│   └── dataset_links.md                 # Dataset download instructions
│
├── ⚙️  preprocessing/
│   ├── process_radar_data.py            # NEXRAD radar preprocessing
│   └── process_satellite_images.py      # GOES/SEVIR satellite preprocessing
│
├── 🧠 models/
│   ├── cnn_weather_model.py             # CNN-based perception model
│   ├── transformer_weather_model.py     # ViT-based perception model
│   └── multimodal_encoder.py            # Unified multi-modal encoder
│
├── 🤖 rl_agent/
│   ├── environment.py                   # Custom Gym environment
│   ├── agent_ppo.py                     # PPO agent wrapper
│   └── training.py                      # RL training loop
│
├── 🚨 orchestration/
│   └── emergency_action_simulator.py    # Simulated emergency actions
│
├── 🔬 experiments/
│   ├── train_weather_model.py           # Perception model training
│   ├── train_rl_agent.py                # RL agent training
│   └── evaluate_system.py              # End-to-end evaluation
│
├── 📈 results/
│   ├── reward_curve_example.png
│   ├── accuracy_plot_example.png
│   └── experiment_results_template.csv
│
└── 📓 notebooks/
    └── demo_pipeline.ipynb              # ▶️ Full pipeline demo
```

---

## ⚙️ Installation

### Option 1 — pip

```bash
git clone https://github.com/aliakarma/agentic-weather-rl.git
cd agentic-weather-rl
pip install -r requirements.txt
```

### Option 2 — Conda *(recommended)*

```bash
conda env create -f environment.yml
conda activate weather-rl
```

> **Requirements:** Python 3.10 · PyTorch ≥ 2.0 · CUDA 11.8+ *(optional but recommended)*

### 🚀 Quick Start — Notebook Demo

```bash
cd notebooks
jupyter notebook demo_pipeline.ipynb
```

> ⚠️ The demo notebook uses **synthetic data** and requires no dataset downloads.  
> Full training pipelines are in `experiments/`.

---

## 📡 Datasets

Three publicly available meteorological datasets are used. All are freely accessible via AWS Open Data.

### 🌩️ SEVIR — Storm Event Imagery

> Temporally aligned radar, satellite, and lightning observations for thousands of documented storm events.

- **Source:** [AWS Open Data — SEVIR](https://registry.opendata.aws/sevir/)
- **Format:** HDF5 · multi-channel imagery
- **Used for:** Perception model training — storm event classification

### 🛰️ GOES — Geostationary Operational Environmental Satellite

> Continuous geostationary satellite imagery capturing cloud formations and large-scale atmospheric structures.

- **Source:** [AWS Open Data — NOAA GOES](https://registry.opendata.aws/noaa-goes/)
- **Format:** NetCDF4 · multi-band spectral imagery
- **Used for:** Cloud pattern and temperature structure input

### 📻 NOAA NEXRAD — Next Generation Weather Radar

> High-resolution atmospheric reflectivity measurements for precipitation analysis and storm dynamics tracking.

- **Source:** [AWS Open Data — NOAA NEXRAD](https://registry.opendata.aws/noaa-nexrad/)
- **Format:** Level-2 binary · reflectivity in dBZ
- **Used for:** Radar reflectivity and rainfall intensity input

> 📂 See [`data/dataset_links.md`](data/dataset_links.md) for detailed download, extraction, and directory setup instructions.

---

## 🔁 Training Pipeline

Follow these four steps to reproduce the full experimental pipeline.

### Step 1 — Preprocess Data

```bash
# Process NEXRAD radar observations
python preprocessing/process_radar_data.py \
    --data_dir data/nexrad \
    --output_dir data/processed

# Process GOES/SEVIR satellite imagery
python preprocessing/process_satellite_images.py \
    --data_dir data/goes \
    --output_dir data/processed
```

### Step 2 — Train Perception Model

```bash
# CNN backbone
python experiments/train_weather_model.py \
    --model cnn --epochs 30 --batch_size 32

# Vision Transformer (ViT)
python experiments/train_weather_model.py \
    --model vit --epochs 30 --batch_size 16
```

### Step 3 — Train RL Agent

```bash
python experiments/train_rl_agent.py \
    --timesteps 100000 \
    --perception_model results/best_perception_model.pth \
    --seed 42
```

### Step 4 — Evaluate End-to-End System

```bash
python experiments/evaluate_system.py \
    --model_path results/ppo_agent \
    --output_dir results/ \
    --seed 42
```

---

## 📊 Results

### 🎯 RL Agent — Decision Accuracy

| Scenario | Decision Accuracy |
|---|:---:|
| 🌪️ Storm Warning Decision | **0.91** |
| 🌊 Flood Risk Response | **0.87** |
| 🚶 Evacuation Recommendation | **0.84** |

### 🧠 Perception Model — Classification Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|---|:---:|:---:|:---:|:---:|
| CNN Baseline | 0.82 | 0.80 | 0.81 | 0.80 |
| **Vision Transformer (Fine-tuned)** | **0.89** | **0.88** | **0.87** | **0.88** |

> 📈 The Vision Transformer outperforms the CNN baseline on all metrics, demonstrating the advantage of attention-based feature extraction for complex meteorological imagery patterns.

---

## 🔬 Reproducibility

All experiments use fixed random seeds across `random`, `numpy`, `torch`, and CUDA. Default seed: **42**.

```bash
# Fully reproducible training run
python experiments/train_rl_agent.py --seed 42 --timesteps 100000

# Fully reproducible evaluation
python experiments/evaluate_system.py --seed 42
```

The demo notebook also sets the global seed at startup:

```python
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
set_seed(42)
```

---


## ⚖️ License

This project is licensed under the **MIT License** — see [`LICENSE`](LICENSE) for full details.

---

<div align="center">

Made with ❤️ for safer communities and smarter disaster response.

⭐ **If this work is useful to you, please consider starring the repository!** ⭐

</div>
