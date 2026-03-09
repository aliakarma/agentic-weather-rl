<div align="center">

# 🌩️ Risk-Aware Multi-Agent Reinforcement Learning<br/>for Cloudburst Disaster Response

[![Paper](https://img.shields.io/badge/MDPI-Mathematics-2088FF?style=for-the-badge&logo=semanticscholar&logoColor=white)](https://www.mdpi.com/journal/mathematics)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)
[![NumPy](https://img.shields.io/badge/NumPy-Only-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)

[![Open Demo in Colab](https://img.shields.io/badge/▶%20Demo%20Notebook-Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/aliakarma/agentic-weather-rl/blob/main/notebooks/colab_demo.ipynb)
[![Open Train in Colab](https://img.shields.io/badge/🚀%20Train%20Notebook-Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/aliakarma/agentic-weather-rl/blob/main/notebooks/colab_train.ipynb)
[![Open Full in Colab](https://img.shields.io/badge/📊%20Full%20Reproduction-Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/aliakarma/agentic-weather-rl/blob/main/notebooks/colab_full.ipynb)
[![Open Perception in Colab](https://img.shields.io/badge/🔬%20Perception%20Ablation-Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/aliakarma/agentic-weather-rl/blob/main/notebooks/colab_perception.ipynb)

<br/>

*A constrained multi-agent reinforcement learning framework that coordinates Storm, Flood, and Evacuation response agents under formal safety constraints — achieving* ***81.5 ± 2.6 reward*** *with only* ***2.3% safety violations.***

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Architecture](#️-architecture)
- [Quick Start](#-quick-start)
- [Notebooks](#-notebooks)
- [Repository Structure](#-repository-structure)
- [Configuration](#️-configuration)
- [Citation](#-citation)
- [License](#-license)

---

## 🔍 Overview

This repository contains the full implementation of **LagrangianCTDE** — a risk-aware multi-agent reinforcement learning algorithm for coordinating emergency response during cloudburst disaster events. The system combines:

- 🌧️ **Multi-modal perception** — two-stream ViT encoder fusing NEXRAD radar and GOES-16 satellite imagery
- 🤝 **Centralised Training, Decentralised Execution (CTDE)** — agents share a joint critic during training but act independently at deployment
- ⚖️ **Lagrangian constraint satisfaction** — primal-dual updates enforce hard safety constraints (violation rate ≤ 10%)
- 🏙️ **Three-agent coordination** — Storm, Flood, and Evacuation agents with role-specific observation partitioning

The framework is benchmarked against six baselines (Heuristic, DQN, IPPO, QMIX, MAPPO, CPO) on a synthetic cloudburst disaster environment calibrated to real SEVIR weather event statistics.

---

## 📈 Key Results

### Table 1 — Perception Encoder Ablation

| Variant | Macro F1 | Accuracy | Notes |
|---|:---:|:---:|---|
| Radar CNN | 0.77 | 0.76 | Radar-only baseline |
| Multimodal CNN | 0.84 | 0.83 | Radar + satellite fusion |
| ViT Single-stream | 0.85 | 0.84 | Radar-only ViT |
| **ViT Multimodal** ⭐ | **0.88** | **0.87** | Two-stream ViT — **proposed** |

*All pairwise differences significant at p < 0.05 (paired t-test, 5 seeds).*

### Table 2 — MARL Method Comparison

| Method | Reward ↑ | Violation Rate ↓ | Constraint Met? |
|---|:---:|:---:|:---:|
| Heuristic | 42.1 ± 1.8 | 18.3% | ✗ |
| DQN | 55.6 ± 3.2 | 14.7% | ✗ |
| IPPO | 63.4 ± 4.1 | 12.1% | ✗ |
| QMIX | 69.8 ± 3.7 | 10.5% | ✗ |
| MAPPO | 74.3 ± 3.0 | 8.9% | ✓ |
| CPO | 71.2 ± 2.8 | 4.1% | ✓ |
| **LagrangianCTDE** ⭐ | **81.5 ± 2.6** | **2.3%** | **✓** |

*Results averaged over 5 random seeds × 500 evaluation episodes. Constraint threshold d = 10%.*

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     LagrangianCTDE System                       │
│                                                                 │
│  ┌──────────────┐    ┌──────────────────────────────────────┐   │
│  │  Perception  │    │           MARL Policy                │   │
│  │   Module     │    │                                      │   │
│  │              │    │  Agent 1 (Storm)   → Actor MLP-256   │   │
│  │  NEXRAD  ┐   │    │  Agent 2 (Flood)   → Actor MLP-256   │   │
│  │  Radar   ├──▶│φ_t │  Agent 3 (Evac.)  → Actor MLP-256   │    │
│  │  GOES-16 ┘   │    │                                      │   │
│  │  ViT-B/16    │    │  Joint Critic (CTDE)  → MLP-512      │   │
│  └──────────────┘    │  Lagrange Multipliers → λ₁, λ₂, λ₃   │   │
│                      └──────────────────────────────────────┘   │
│                                        │                        │
│                              ┌─────────▼──────────┐             │
│                              │  Orchestration Layer│            │
│                              │  (Emergency Alerts) │            │
│                              └────────────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

**Key components:**

| Component | File | Description |
|---|---|---|
| Environment | `src/environment/disaster_env.py` | 3-agent disaster benchmark, severity Markov chain |
| Actor | `src/models/actor.py` | Per-agent MLP (hidden=256), decentralised execution |
| Critic | `src/models/critic.py` | Joint-observation MLP (hidden=512), 4 value heads |
| Algorithm | `src/algorithms/lagrangian_ctde.py` | Primal-dual PPO, Algorithm 1 of paper |
| Perception | `src/models/vit_encoder.py` | Two-stream ViT encoder, 4 ablation variants |
| Orchestration | `src/orchestration/orchestration.py` | Action → emergency alert translation |

---

## 🚀 Quick Start

### Option 1 — Google Colab (recommended, zero setup)

Click any badge at the top of this page. Each notebook clones the repo and installs dependencies automatically in its first cell.

### Option 2 — Local installation

```bash
# 1. Clone
git clone https://github.com/aliakarma/agentic-weather-rl.git
cd agentic-weather-rl

# 2. Install dependencies (no PyTorch required)
pip install -r requirements.txt

# 3. Run the demo (~5 seconds)
python scripts/run_demo.py

# 4. Short training run (~30 seconds)
python scripts/run_train.py

# 5. Full Table 2 reproduction (~2 minutes)
python scripts/run_full.py

# 6. Perception ablation / Table 1 (~30 seconds)
python scripts/run_perception.py
```

### Option 3 — Shell scripts (GPU cluster / A100)

```bash
bash scripts/demo.sh                      # ~5 min
bash scripts/train_short.sh               # ~20 min
bash scripts/train_full.sh                # ~2 hrs on A100
bash scripts/train_baselines.sh           # ~2–3 hrs on A100
bash scripts/train_perception_ablation.sh # ~30 min on T4
```

---

## 📓 Notebooks

All notebooks are in `notebooks/` and self-contained — the first cell handles cloning and setup automatically on Colab.

| Notebook | Colab | Runtime | Description |
|---|:---:|:---:|---|
| `colab_demo_fixed.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/agentic-weather-rl/blob/main/notebooks/colab_demo.ipynb) | ~5 min · CPU | Evaluate pretrained policy, verify Table 2 targets |
| `colab_train_fixed.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/agentic-weather-rl/blob/main/notebooks/colab_train.ipynb) | ~20 min · T4 | PPO training from scratch, training curves |
| `colab_full_fixed.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/agentic-weather-rl/blob/main/notebooks/colab_full.ipynb) | ~4 hrs · A100 | Full Table 2: 5 seeds × all 7 methods |
| `colab_perception_fixed.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/agentic-weather-rl/blob/main/notebooks/colab_perception.ipynb) | ~30 min · T4 | Table 1: 4 encoder variants + t-test significance |

---

## 📁 Repository Structure

```
agentic-weather-rl/
│
├── 📄 README.md
├── 📄 LICENSE
├── 📄 CITATION.cff
├── 📄 requirements.txt
├── 📄 setup.py
├── 📄 .gitignore
│
├── ⚙️  configs/
│   ├── default.yaml          # Shared defaults
│   ├── training.yaml         # MARL hyperparameters (Table 3)
│   ├── environment.yaml      # Benchmark environment parameters
│   └── perception.yaml       # ViT encoder training config
│
├── 🧠 src/
│   ├── environment/
│   │   ├── disaster_env.py   # 3-agent disaster benchmark
│   │   ├── hazard_generator.py
│   │   ├── obs_router.py     # Agent-specific observation partitioning
│   │   └── reward.py         # Reward + constraint cost functions
│   │
│   ├── models/
│   │   ├── actor.py          # Decentralised actor MLP (hidden=256)
│   │   ├── critic.py         # Centralised critic MLP (hidden=512)
│   │   └── vit_encoder.py    # Two-stream ViT encoder (4 ablation variants)
│   │
│   ├── algorithms/
│   │   ├── lagrangian_ctde.py  # Primal-dual update (Algorithm 1)
│   │   ├── ppo.py
│   │   └── baselines/
│   │       ├── heuristic.py  │  dqn.py  │  ippo.py
│   │       ├── qmix.py       │  mappo.py │  cpo.py
│   │
│   ├── orchestration/
│   │   └── orchestration.py  # Action → emergency alert callbacks
│   │
│   ├── train.py              # Main training entry point
│   └── evaluate.py           # Evaluation runner
│
├── 📓 notebooks/
│   ├── colab_demo.ipynb
│   ├── colab_train.ipynb
│   ├── colab_full.ipynb
│   └── colab_perception.ipynb
│
├── 🔧 scripts/
│   ├── run_demo.py           ← python scripts/run_demo.py
│   ├── run_train.py          ← python scripts/run_train.py
│   ├── run_full.py           ← python scripts/run_full.py
│   ├── run_perception.py     ← python scripts/run_perception.py
│   ├── demo.sh  │  train_short.sh  │  train_full.sh
│   ├── train_baselines.sh
│   └── train_perception_ablation.sh
│
├── 💾 checkpoints/
│   ├── marl_policy.pt             # Pretrained MARL policy (best seed)
│   ├── marl_policy_seed{42..46}.pt
│   ├── perception_encoder.pt      # Best ViT encoder (vit_multimodal)
│   └── perception_encoder_{variant}.pt
│
└── 📊 results/
    └── example_results/
        ├── table2_reproduction.json
        ├── table2_comparison.pdf
        ├── perception_ablation.json
        └── table1_perception_ablation.pdf
```

---

## ⚙️ Configuration

Key hyperparameters from **Table 3** of the paper:

```yaml
# configs/training.yaml
lr:            3e-4    # Actor/critic learning rate
gamma:         0.99    # Discount factor
gae_lambda:    0.95    # GAE λ
clip_epsilon:  0.2     # PPO clip ratio
entropy_coef:  0.01    # Entropy bonus
constraint_d:  0.10    # Safety constraint threshold (10%)
lambda_lr:     1e-3    # Lagrange multiplier learning rate
lambda_init:   0.10    # Initial λ value
hidden_actor:  256     # Actor hidden dimension
hidden_critic: 512     # Critic hidden dimension
```

---

## 📦 Requirements

```
numpy>=1.23
matplotlib>=3.5
scipy>=1.9
```

> **No PyTorch required** for CPU inference and the demo. GPU training with PyTorch is supported — install via `pip install torch` separately.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Made with ❤️ for safer disaster response systems

[![GitHub stars](https://img.shields.io/github/stars/aliakarma/agentic-weather-rl?style=social)](https://github.com/aliakarma/agentic-weather-rl/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/aliakarma/agentic-weather-rl?style=social)](https://github.com/aliakarma/agentic-weather-rl/network/members)

</div>
