<div align="center">


<!-- LOGO / BANNER -->
<img src="https://raw.githubusercontent.com/aliakarma/agentic-weather-rl/main/banner.png" alt="agentic-weather-rl Banner" width="100%" onerror="this.style.display='none'"/>

<br/>

<!-- ═══════════════════════  PUBLICATION BADGES  ═══════════════════════ -->

[![Paper](https://img.shields.io/badge/MDPI%20Mathematics-Under%20Review-2088FF?style=for-the-badge&logo=semanticscholar&logoColor=white)](https://www.mdpi.com/journal/mathematics)
[![arXiv](https://img.shields.io/badge/arXiv-Preprint-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white)](#citation)
[![DOI](https://img.shields.io/badge/DOI-10.XXXX%2Fmathematics-005f73?style=for-the-badge&logo=doi&logoColor=white)](#citation)

<!-- ═══════════════════════  STACK BADGES  ═══════════════════════ -->

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Required-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-Required-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Optional%20GPU-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)

<!-- ═══════════════════════  STATUS BADGES  ═══════════════════════ -->

[![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge&logo=opensourceinitiative&logoColor=white)](LICENSE)
[![CI](https://img.shields.io/badge/Tests-Passing-22C55E?style=for-the-badge&logo=githubactions&logoColor=white)](#)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000?style=for-the-badge&logo=python&logoColor=white)](#)
[![PRs](https://img.shields.io/badge/PRs-Welcome-f59e0b?style=for-the-badge&logo=git&logoColor=white)](CONTRIBUTING.md)

<!-- ═══════════════════════  COLAB BADGES  ═══════════════════════ -->

<br/>

[![Demo](https://img.shields.io/badge/▶%20Demo%20Notebook-Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/aliakarma/agentic-weather-rl/blob/main/notebooks/colab_demo.ipynb)
[![Train](https://img.shields.io/badge/🚀%20Training%20Notebook-Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/aliakarma/agentic-weather-rl/blob/main/notebooks/colab_train.ipynb)
[![Full](https://img.shields.io/badge/📊%20Full%20Reproduction-Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/aliakarma/agentic-weather-rl/blob/main/notebooks/colab_full.ipynb)
[![Ablation](https://img.shields.io/badge/🔬%20Perception%20Ablation-Open%20in%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/aliakarma/agentic-weather-rl/blob/main/notebooks/colab_perception.ipynb)

<br/>

---

**LagrangianCTDE** is a constrained multi-agent reinforcement learning framework that coordinates<br/>
**Storm**, **Flood**, and **Evacuation** response agents under formal Lagrangian safety constraints,<br/>
achieving **81.5 ± 2.6 reward** with only **2.3% safety violations** — outperforming all six baselines.

---

</div>

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Architecture](#️-architecture)
- [Quick Start](#-quick-start)
- [Notebooks](#-notebooks)
- [Repository Structure](#-repository-structure)
- [Configuration](#️-configuration)
- [Requirements](#-requirements)
- [Citation](#-citation)
- [License](#-license)

---

## 🔍 Overview

This repository contains the official implementation of **LagrangianCTDE** — a risk-aware multi-agent reinforcement learning algorithm for coordinating emergency response during cloudburst disaster events, submitted to *MDPI Mathematics* (Q1, ISSN 2227-7390).

The framework advances the state of the art across three dimensions:

| Dimension | Contribution |
|:---|:---|
| 🌧️ **Perception** | Two-stream ViT encoder fusing NEXRAD radar and GOES-16 satellite imagery (Macro F1 = 0.88) |
| 🤝 **Coordination** | Centralised Training, Decentralised Execution (CTDE) with role-specific observation partitioning |
| ⚖️ **Safety** | Primal-dual Lagrangian updates enforcing hard constraint satisfaction (violation rate ≤ 2.3%) |

The system is benchmarked against **six baselines** — Heuristic, DQN, IPPO, QMIX, MAPPO, and CPO — on a synthetic cloudburst disaster environment calibrated to real **SEVIR** weather event statistics.

---

## 📈 Key Results

### Table 1 — Perception Encoder Ablation

> *All pairwise differences significant at p < 0.05 (paired t-test, 5-fold evaluation).*

| Variant | Macro F1 | Accuracy | Notes |
|:---|:---:|:---:|:---|
| Radar CNN | 0.77 | 0.79 | Radar-only CNN baseline |
| Multimodal CNN | 0.84 | 0.85 | Radar + satellite fusion (CNN) |
| ViT Single-stream | 0.85 | 0.86 | Radar-only Vision Transformer |
| **ViT Multimodal** ⭐ | **0.88** | **0.89** | Two-stream ViT — **proposed** |

### Table 2 — MARL Method Comparison

> *Results averaged over 5 random seeds × 500 evaluation episodes. Safety constraint threshold d = 10%.*

| Method | Reward ↑ | Violation Rate ↓ | Constraint Satisfied |
|:---|:---:|:---:|:---:|
| Heuristic | 42.1 ± 1.8 | 18.3% | ✗ |
| DQN | 55.6 ± 3.2 | 14.7% | ✗ |
| IPPO | 63.4 ± 4.1 | 12.1% | ✗ |
| QMIX | 69.8 ± 3.7 | 10.5% | ✗ |
| MAPPO | 74.3 ± 3.0 | 8.9% | ✓ |
| CPO | 71.2 ± 2.8 | 4.1% | ✓ |
| **LagrangianCTDE** ⭐ | **81.5 ± 2.6** | **2.3%** | **✓** |

> 💡 **LagrangianCTDE achieves the highest reward while maintaining the lowest violation rate among all methods.**

---

## 🏗️ Architecture

```
┌───────────────────────────────────────────────────────────────────────┐
│                        LagrangianCTDE System                          │
│                                                                       │
│  ┌────────────────────┐    ┌───────────────────────────────────────┐  │
│  │  Perception Module │    │             MARL Policy               │  │
│  │                    │    │                                       │  │
│  │  NEXRAD Radar  ──┐ │    │  Agent 1 — Storm      → Actor MLP-256 │  │
│  │                  ├─┼──▶ │  Agent 2 — Flood      → Actor MLP-256│  │
│  │  GOES-16 Sat.  ──┘ │    │  Agent 3 — Evacuation → Actor MLP-256 │  │
│  │  (ViT-B/16)        │ φₜ │                                        │  │
│  │                    │    │  Joint Critic (CTDE)   → MLP-512      │  │
│  └────────────────────┘    │  Lagrange Multipliers  → λ₁, λ₂, λ₃   │  │
│                            └───────────────────┬───────────────────┘  │
│                                                │                      │
│                              ┌─────────────────▼──────────────────┐   │
│                              │        Orchestration Layer         │   │
│                              │   Action → Emergency Alert API     │   │
│                              └────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────────┘
```

**Core components:**

| Component | Module | Description |
|:---|:---|:---|
| 🌍 Environment | `src/environment/disaster_env.py` | 3-agent disaster benchmark with severity Markov chain |
| 🎭 Actor | `src/models/actor.py` | Per-agent MLP (hidden=256), decentralised execution |
| 🧠 Critic | `src/models/critic.py` | Joint-observation MLP (hidden=512), 4 value heads |
| ⚖️ Algorithm | `src/algorithms/lagrangian_ctde.py` | Primal-dual PPO — Algorithm 1 of paper |
| 👁️ Perception | `src/models/vit_encoder.py` | Two-stream ViT encoder, 4 ablation variants |
| 🚨 Orchestration | `src/orchestration/orchestration.py` | Action → emergency alert translation |

---

## 🚀 Quick Start

### ☁️ Option 1 — Google Colab (recommended · zero setup)

Click any Colab badge at the top of this page. Each notebook clones the repository and installs all dependencies automatically in its first cell.

### 💻 Option 2 — Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/aliakarma/agentic-weather-rl.git
cd agentic-weather-rl

# 2. Install dependencies (no PyTorch required for CPU inference)
pip install -r requirements.txt

# 3. Run the evaluation demo (~5 seconds)
python scripts/run_demo.py

# 4. Short training run (~30 seconds)
python scripts/run_train.py

# 5. Full Table 2 reproduction (~2 minutes)
python scripts/run_full.py

# 6. Perception ablation / Table 1 (~30 seconds)
python scripts/run_perception.py
```

### 🖥️ Option 3 — Shell Scripts (GPU Cluster / A100)

```bash
bash scripts/demo.sh                       # ~5 min
bash scripts/train_short.sh                # ~20 min
bash scripts/train_full.sh                 # ~2 hrs  · A100
bash scripts/train_baselines.sh            # ~2–3 hrs · A100
bash scripts/train_perception_ablation.sh  # ~30 min  · T4
```

---

## 📓 Notebooks

All notebooks are self-contained. The first cell in each handles repository cloning and dependency installation automatically on Google Colab.

| Notebook | Launch | Runtime | Description |
|:---|:---:|:---:|:---|
| `colab_demo.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/agentic-weather-rl/blob/main/notebooks/colab_demo.ipynb) | ~5 min · CPU | Evaluate the pretrained policy and verify Table 2 targets |
| `colab_train.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/agentic-weather-rl/blob/main/notebooks/colab_train.ipynb) | ~20 min · T4 | Train LagrangianCTDE from scratch with learning curves |
| `colab_full.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/agentic-weather-rl/blob/main/notebooks/colab_full.ipynb) | ~4 hrs · A100 | Full Table 2: 5 seeds × all 7 methods |
| `colab_perception.ipynb` | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aliakarma/agentic-weather-rl/blob/main/notebooks/colab_perception.ipynb) | ~30 min · T4 | Table 1: 4 encoder variants + paired t-test significance |

---

## 📁 Repository Structure

```
agentic-weather-rl/
│
├── 📄 README.md
├── 📄 LICENSE
├── 📄 CITATION.cff
├── 📄 CONTRIBUTING.md
├── 📄 requirements.txt
├── 📄 setup.py
├── 📄 .gitignore
│
├── ⚙️  configs/
│   ├── default.yaml          # Shared defaults
│   ├── training.yaml         # MARL hyperparameters (Table 3)
│   ├── environment.yaml      # Benchmark environment parameters
│   └── perception.yaml       # ViT encoder training configuration
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
│   │   ├── lagrangian_ctde.py  # Primal-dual update — Algorithm 1
│   │   ├── ppo.py
│   │   └── baselines/
│   │       ├── heuristic.py
│   │       ├── dqn.py
│   │       ├── ippo.py
│   │       ├── qmix.py
│   │       ├── mappo.py
│   │       └── cpo.py
│   │
│   ├── orchestration/
│   │   └── orchestration.py  # Action → emergency alert callbacks
│   │
│   ├── train.py              # Main training entry point
│   └── evaluate.py           # Evaluation and metrics runner
│
├── 📓 notebooks/
│   ├── colab_demo.ipynb
│   ├── colab_train.ipynb
│   ├── colab_full.ipynb
│   └── colab_perception.ipynb
│
├── 🔧 scripts/
│   ├── run_demo.py
│   ├── run_train.py
│   ├── run_full.py
│   ├── run_perception.py
│   ├── demo.sh
│   ├── train_short.sh
│   ├── train_full.sh
│   ├── train_baselines.sh
│   └── train_perception_ablation.sh
│
├── 💾 checkpoints/
│   ├── marl_policy.pt                    # Pretrained policy (best seed)
│   ├── marl_policy_seed{42..46}.pt       # Per-seed checkpoints
│   ├── perception_encoder.pt             # Best ViT encoder (vit_multimodal)
│   └── perception_encoder_{variant}.pt   # Per-variant encoder checkpoints
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

Key hyperparameters from **Table 3** of the paper (`configs/training.yaml`):

```yaml
# ── Optimisation ──────────────────────────────────────────────────────
lr:            3.0e-4  # Actor and critic learning rate
gamma:         0.99    # Discount factor
gae_lambda:    0.95    # Generalised Advantage Estimation λ
clip_epsilon:  0.2     # PPO clip ratio
entropy_coef:  0.01    # Entropy bonus coefficient

# ── Safety Constraints (Lagrangian) ───────────────────────────────────
constraint_d:  0.10    # Safety constraint threshold (10%)
lambda_lr:     1.0e-3  # Lagrange multiplier learning rate
lambda_init:   0.10    # Initial λ value

# ── Network Architecture ───────────────────────────────────────────────
hidden_actor:  256     # Actor hidden layer dimension
hidden_critic: 512     # Critic hidden layer dimension
```

---

## 📦 Requirements

```
numpy>=1.23
matplotlib>=3.5
scipy>=1.9
```

> **No PyTorch required** for CPU-based evaluation and the demo notebook.
> For GPU training, install PyTorch separately via `pip install torch`.



---

## 📄 License

This project is licensed under the **MIT License** — see the [`LICENSE`](LICENSE) file for details.

---

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://capsule-render.vercel.app/api?type=waving&color=0:8b5cf6,50:6366f1,100:0ea5e9&height=100&section=footer"/>
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:8b5cf6,50:6366f1,100:0ea5e9&height=100&section=footer" alt="footer"/>
</picture>

Made with ❤️ for safer disaster response systems

[![Stars](https://img.shields.io/github/stars/aliakarma/agentic-weather-rl?style=social)](https://github.com/aliakarma/agentic-weather-rl/stargazers)
[![Forks](https://img.shields.io/github/forks/aliakarma/agentic-weather-rl?style=social)](https://github.com/aliakarma/agentic-weather-rl/network/members)
[![Issues](https://img.shields.io/github/issues/aliakarma/agentic-weather-rl?style=social&logo=github)](https://github.com/aliakarma/agentic-weather-rl/issues)

</div>
