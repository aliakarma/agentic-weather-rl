# Risk-Aware MARL for Cloudburst Disaster Response

> **Paper:** *Risk-Aware Multi-Agent Reinforcement Learning for Coordinated Cloudburst Disaster Response*
> **Journal:** MDPI Mathematics — Algorithmic Advances in Reinforcement Learning

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.1](https://img.shields.io/badge/pytorch-2.1-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Demo Notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/risk-aware-marl-cloudburst/blob/main/notebooks/colab_demo.ipynb)
[![Full Training](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/risk-aware-marl-cloudburst/blob/main/notebooks/colab_train.ipynb)

A constrained multi-agent reinforcement learning (MARL) framework for coordinated
disaster response under extreme weather. Three specialised agents — Storm Detection,
Flood Risk Assessment, and Evacuation Planning — are trained jointly using
Lagrangian-relaxed Centralised-Training Decentralised-Execution (CTDE) PPO inside
the custom `DisasterResponseBenchmark` environment.

The system follows a **three-layer architecture** (Paper Sections 3.1–3.5):
- **Layer 1 — Multi-Modal Weather Perception:** two-stream ViT-B/16 encoder
  fusing NEXRAD radar and GOES-16 satellite inputs into a 128-dimensional
  state feature vector φ_t.
- **Layer 2 — MARL Decision Layer:** three cooperative agents trained under
  Lagrangian-relaxed CTDE-PPO with heterogeneous, agent-specific observations.
- **Layer 3 — Orchestration Interface:** translates the joint action a_t into
  simulated downstream emergency response callbacks.

![Architecture](figures/architecture.png)

> **Reviewers:** The fastest way to verify results is via Google Colab —
> no local setup required. See the [Run on Google Colab](#run-on-google-colab--no-setup-required) section below.

---

## Quick Start — 5 Minutes

No training needed. Evaluate the pretrained model immediately:

```bash
git clone https://github.com/your-username/risk-aware-marl-cloudburst.git
cd risk-aware-marl-cloudburst
pip install -r requirements.txt
bash scripts/demo.sh
```

Expected output:

```
Loading checkpoint: checkpoints/marl_policy.pt  (single best seed)
Running 500 evaluation episodes...

Storm Warning Accuracy    :  0.91
Flood Risk Accuracy       :  0.87
Evacuation Accuracy       :  0.84

Reward (this seed)        :  81.3
Violation Rate (this seed):   2.4%

Note: Paper Table 2 reports 81.5 ± 2.6 and VR = 2.3% as the mean
      across all 5 seeds. Single-seed results vary within ± std.

Done. Results saved to results/demo_output/
```

Runtime: **3–5 minutes** on CPU or GPU.

---

## Run on Google Colab — No Setup Required

Reviewers and readers with no local GPU can run everything directly in
the browser. No installation, no downloads, no environment setup.

| Notebook | What it does | Runtime | GPU needed |
|----------|-------------|---------|------------|
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/risk-aware-marl-cloudburst/blob/main/notebooks/colab_demo.ipynb) **Demo** | Loads pretrained checkpoint, runs 500 eval episodes, prints Table 2 & 3 results | ~5 min | No (CPU runtime) |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/risk-aware-marl-cloudburst/blob/main/notebooks/colab_train.ipynb) **Short Training** | Trains proposed method for 100 episodes, plots live reward curve | ~20 min | T4 (free tier) |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/risk-aware-marl-cloudburst/blob/main/notebooks/colab_full.ipynb) **Full Reproduction** | Trains all 5 seeds + all 6 baselines, exports Tables 2–4 as CSV | ~4–5 hrs | A100 (Colab Pro) |
| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/risk-aware-marl-cloudburst/blob/main/notebooks/colab_perception.ipynb) **Perception Ablation** | Trains all 4 encoder variants from Table 1 on the SEVIR sample | ~30 min | T4 (free tier) |

### What each Colab notebook does step by step

**`colab_demo.ipynb` — Recommended starting point for reviewers**

1. Clones the repository and installs all dependencies automatically
2. Downloads `checkpoints/marl_policy.pt` from the repo
3. Runs `src/evaluate.py` over 500 episodes against the synthetic benchmark
4. Prints per-scenario decision accuracy (Table 3) and reward/VR matching Table 2
5. Plots the architecture diagram (`figures/architecture.png`)

No Google account storage is used. Runs in under 5 minutes on a free CPU runtime.

---

**`colab_train.ipynb` — Short training run**

1. Sets up the environment and installs dependencies
2. Connects to a free T4 GPU (Runtime → Change runtime type → T4)
3. Trains the Lagrangian CTDE agent for 100 episodes on one seed
4. Plots a live training curve (reward vs. episodes)
5. Optionally saves the checkpoint to Google Drive

Expected reward at episode 100: ~55–65. Full convergence occurs around
episode 600 per Fig. 2(a) of the paper.

---

**`colab_full.ipynb` — Full paper reproduction (Colab Pro required)**

1. Trains the proposed method across all 5 seeds (1,000 episodes each)
2. Trains all 6 baseline methods across 5 seeds each
3. Runs evaluation and assembles Tables 2, 3, and 4 automatically
4. Exports `tables.csv` and all training curve figures to Google Drive
5. Prints a summary comparing reproduced numbers to the paper's reported values

> **Colab Pro note:** Requires an A100 or V100 session (~$10/month) and
> approximately 4–5 hours of runtime. On the free T4 tier, use
> Runtime → Save to Drive every 200 episodes to guard against disconnects.

---

**`colab_perception.ipynb` — Table 1 encoder ablation**

1. Downloads the 50-storm SEVIR sample (`data/sample/sevir_subset.h5`, ~150 MB)
   directly from the repository — no 30 GB full dataset needed
2. Trains all four encoder variants: radar-only CNN, multi-modal CNN,
   single-modal ViT, and multi-modal ViT (ours)
3. Reports accuracy, precision, recall, and F1 for each variant

> **Note:** Numbers on the 50-storm sample will be approximate. Full
> Table 1 results require the complete SEVIR dataset (see Full Datasets section).

---

### Colab tips for reviewers

**Free tier is sufficient** for `colab_demo.ipynb` and
`colab_perception.ipynb`. Only `colab_full.ipynb` needs Colab Pro.

Connect Google Drive in training notebooks to persist checkpoints
across sessions:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Verify GPU allocation before running any training:

```python
import torch
print(torch.cuda.get_device_name(0))  # T4, V100, or A100
print(torch.cuda.is_available())       # Must be True
```

All notebooks are fully self-contained. Every cell that requires a
package installs it inline with `!pip install`. Click **Runtime →
Run All** and the notebook handles everything.

The notebooks also correspond to the `notebooks/` folder in the
repository structure below, so the same cells can be run locally
if preferred.

---

## Repository Structure

```
risk-aware-marl-cloudburst/
│
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
│
├── configs/
│   ├── default.yaml              # Shared defaults
│   ├── training.yaml             # MARL hyperparameters (Table 3 of paper)
│   ├── environment.yaml          # Benchmark environment parameters (Table 1)
│   └── perception.yaml           # ViT encoder training config (Section 4.2)
│
├── src/
│   ├── environment/
│   │   ├── disaster_env.py       # DisasterResponseBenchmark (20×20 grid)
│   │   ├── hazard_generator.py   # Poisson + Gaussian-process hazard model
│   │   ├── obs_router.py         # Agent-specific observation partitioning
│   │   │                         # (o_{i,t} ⊂ s_t per agent; Section 3.3)
│   │   └── reward.py             # Reward and constraint cost functions (Eq. 4–5)
│   │
│   ├── models/
│   │   ├── actor.py              # Decentralised actor MLP (hidden=256);
│   │   │                         # each agent receives its own observation
│   │   │                         # slice from obs_router.py (Section 3.3)
│   │   ├── critic.py             # Centralised critic MLP (hidden=512)
│   │   └── vit_encoder.py        # Two-stream ViT-B/16 encoder (Section 3.1)
│   │                             # fuses NEXRAD radar + GOES-16 satellite
│   │
│   ├── algorithms/
│   │   ├── lagrangian_ctde.py    # Primal-dual update logic (Algorithm 1)
│   │   ├── ppo.py                # PPO with cost-aware advantage (Eq. 7–8)
│   │   └── baselines/
│   │       ├── heuristic.py      # Rule-based threshold policy
│   │       ├── dqn.py            # Single-agent DQN (flat joint action space)
│   │       ├── ippo.py           # Independent PPO (no shared information)
│   │       ├── qmix.py           # QMIX (Rashid et al., 2018)
│   │       ├── mappo.py          # MAPPO (Yu et al., 2022)
│   │       └── cpo.py            # CPO extended to joint action space
│   │
│   ├── orchestration/
│   │   └── orchestration.py      # Layer 3: translates joint action a_t into
│   │                             # simulated emergency response callbacks
│   │                             # (Section 3.5 of paper)
│   │
│   ├── train.py                  # Main training entry point
│   └── evaluate.py               # Evaluation runner
│
├── scripts/
│   ├── demo.sh                   # Evaluate pretrained model       (~5 min)
│   ├── train_short.sh            # Short training run              (~20 min)
│   ├── train_full.sh             # Full MARL experiment, 5 seeds   (~2 hrs on A100)
│   ├── train_baselines.sh        # Train all 6 baseline methods    (~2–3 hrs on A100)
│   └── train_perception_ablation.sh  # Reproduce Table 1 (4 encoder variants)
│
├── checkpoints/
│   ├── perception_encoder.pt     # Pretrained ViT encoder (SEVIR fine-tuned)
│   └── marl_policy.pt            # Pretrained MARL policy (best single seed)
│
├── notebooks/
│   ├── colab_demo.ipynb          # ▶ Evaluate pretrained model (~5 min, CPU)
│   ├── colab_train.ipynb         # ▶ Short training run (~20 min, T4 free)
│   ├── colab_full.ipynb          # ▶ Full paper reproduction (~4–5 hrs, Colab Pro)
│   └── colab_perception.ipynb    # ▶ Table 1 encoder ablation (~30 min, T4 free)
│
├── data/
│   └── sample/
│       └── sevir_subset.h5       # 50-storm SEVIR sample (~150 MB)
│
├── results/
│   └── example_results/
│       ├── reward_curve.png      # Fig. 2(a): mean episodic reward vs. episodes
│       ├── violation_rate.png    # Fig. 2(c): VR across training
│       ├── lambda_curves.png     # Fig. 2(b): dual variable λ_i over training
│       └── tables.csv            # Tables 2–4 numerical results
│
└── figures/
    ├── architecture.png          # Fig. 1: three-layer system architecture
    └── training_dynamics.png     # Fig. 2: training curves (reward, λ, VR)
```

**Total repo size:** ~400 MB (checkpoints + sample data).

---

## Installation

```bash
# 1. Clone
git clone https://github.com/aliakarma/risk-aware-marl-cloudburst.git
cd risk-aware-marl-cloudburst

# 2. Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify
python -c "import torch; from src.environment.disaster_env import DisasterEnv; print('OK')"
```

`requirements.txt`:

```
torch==2.1.0
torchvision==0.16.0
timm==0.9.12
stable-baselines3==2.2.1
gymnasium==0.29.1
numpy==1.26.2
scipy==1.11.4
pandas==2.1.3
matplotlib==3.8.2
pyyaml==6.0.1
tqdm==4.66.1
h5py==3.10.0
netCDF4==1.6.5
tensorboard==2.15.1
```

**Requirements:** Python 3.10, any NVIDIA GPU (or CPU for demo and short run).
No multi-GPU setup required.

---

## Four Ways to Run

### Option 1 — Demo with pretrained model (~5 min, no training)

Loads `checkpoints/marl_policy.pt` (best single seed) and evaluates
over 500 episodes. Per-scenario decision accuracy matches Table 3 of
the paper. Reward and VR are single-seed values; the paper's Table 2
figures are 5-seed means.

```bash
bash scripts/demo.sh
```

---

### Option 2 — Short training run (~20 min on A100)

Trains the proposed Lagrangian CTDE agent for 100 episodes. Useful
for verifying the training loop before committing to the full run.

```bash
bash scripts/train_short.sh
```

Expected reward at episode 100: ~55–65 (policy is still converging;
full convergence occurs around episode 600 per Fig. 2(a)).

---

### Option 3 — Full training, proposed method (~2 hrs on A100)

Reproduces Table 2 results for the proposed Lagrangian CTDE method
across all 5 random seeds.

```bash
bash scripts/train_full.sh
```

Expected final result (mean ± std, 5 seeds, 500 eval episodes):
**Reward = 81.5 ± 2.6, VR = 2.3%**

> **Hardware note:** Timed on a single NVIDIA A100 40 GB. Expect
> ~3–4 hours on an RTX 3090 24 GB.

---

### Option 4 — Baseline comparisons (~2–3 hrs on A100)

Trains all six baseline methods from Table 2 across 5 seeds each.
Must be run after Option 3 (or standalone) to populate the full
comparison table.

```bash
bash scripts/train_baselines.sh
```

This trains: Heuristic, DQN, IPPO, QMIX, MAPPO, and CPO.
Results are written to `results/baseline_results/` and merged
into `results/example_results/tables.csv` alongside the proposed
method's results.

---

## Configuration

### MARL Training (`configs/training.yaml`)

```yaml
training:
  episodes: 1000          # Total training episodes (Table 3)
  gamma: 0.99             # Discount factor γ
  lr: 3.0e-4              # Actor and critic learning rate
  clip: 0.2               # PPO clipping threshold ε
  gae_lambda: 0.95        # GAE λ parameter
  entropy_coef: 0.01      # Entropy coefficient
  lagrangian_lr: 1.0e-3   # Dual variable step size α_λ
  constraint_d: 0.10      # Per-agent constraint threshold d_i
  seeds: [0, 1, 2, 3, 4]  # 5 independent seeds
  batch_size: 64
  mini_batches: 4
```

### Benchmark Environment (`configs/environment.yaml`)

```yaml
environment:
  grid_size: 20           # 20×20 grid (Table 1)
  n_agents: 3
  episode_length: 100     # Steps per episode T
  hazard_lambda: 0.05     # Poisson arrival rate
  reward_alpha: 1.0       # Mitigation weight α
  reward_beta: 0.5        # False-alarm penalty β
  reward_eta: 0.3         # Delay penalty η
  state_dim: 24           # Synthetic mode only (see State Dimension note below)
```

### Perception Encoder (`configs/perception.yaml`)

```yaml
perception:
  backbone: vit_base_patch16_224
  pretrained: true        # ImageNet-21k weights (~330 MB, auto-downloaded)
  output_dim: 128         # d_φ = 128 (Section 4.2)
  epochs: 50
  lr: 1.0e-4
  weight_decay: 1.0e-2
  scheduler: cosine
  optimizer: adamw
  batch_size: 32
  input_size: 224         # Radar and satellite frames resized to 224×224
```

To override any value without editing files:

```bash
python src/train.py --episodes 200 --lr 1e-4 --seeds 0 1 2
```

### State Dimension Note

By default, MARL training runs in **synthetic mode**, where the
perception encoder output φ_t is replaced by a 24-dimensional analytic
risk representation from the hazard generator. The `state_dim: 24` in
`environment.yaml` refers to this synthetic mode only.

When the perception encoder is enabled (see Full Datasets section),
the state dimension grows to 128 (d_φ = 128 from the ViT encoder) plus
scalar contextual variables. Update `state_dim` in `environment.yaml`
accordingly when switching modes.

---

## Monitoring Training

```bash
tensorboard --logdir results/logs/
```

Three key signals from the paper's Fig. 2 to watch during training:

| Plot | What to expect |
|------|----------------|
| `train/mean_reward` | Rises and plateaus around episode 600 |
| `train/violation_rate` | Drops below 0.10 by episode 350; stabilises near 2% |
| `train/lambda_agent_1/2/3` | Rises sharply, then decays to near-zero at convergence |

If violation rate stays above 0.10 after episode 500, increase
`lagrangian_lr` to `3.0e-3` in `configs/training.yaml`.

---

## Full Datasets (Perception Encoder Retraining)

The pretrained `checkpoints/perception_encoder.pt` is included —
**no dataset downloads are needed** to run MARL training or evaluation.

To **retrain the perception encoder from scratch**, all three datasets
described in Section 4.2 of the paper are required. The encoder is a
two-stream ViT that jointly processes radar (NEXRAD) and satellite
(GOES-16) inputs. SEVIR provides paired and temporally-aligned samples
of both modalities; GOES-16 and NEXRAD archives provide additional
standalone coverage.

### SEVIR — Primary (~30 GB)

~10,000 storm events with aligned NEXRAD radar reflectivity and
GOES-16 visible/IR imagery at 384×384 px. Used for supervised
pre-training of the ViT encoder.

```bash
python src/models/vit_encoder.py \
    --download sevir \
    --data_dir data/sevir/
```

### GOES-16 ABI — Auxiliary satellite context (~50 GB, optional)

Multi-year geostationary imagery at 4 km resolution. Supplements
SEVIR with broader synoptic context.

```bash
python src/models/vit_encoder.py \
    --download goes16 \
    --data_dir data/goes16/ \
    --channels C02 C13 \
    --start 2018-01-01 --end 2021-12-31
```

### NEXRAD WSR-88D — Auxiliary radar features (~40 GB, optional)

Multi-year Level-III radar composites at ~1 km radial resolution.
Supplements SEVIR for precipitation structure characterisation.

```bash
python src/models/vit_encoder.py \
    --download nexrad \
    --data_dir data/nexrad/ \
    --start 2018-01-01 --end 2021-12-31
```

### Train the encoder

```bash
python src/models/vit_encoder.py \
    --train \
    --sevir_dir data/sevir/ \
    --goes_dir data/goes16/ \
    --nexrad_dir data/nexrad/ \
    --config configs/perception.yaml \
    --output checkpoints/perception_encoder.pt
```

Training time: ~4–6 hours on a single A100 40 GB / RTX 3090.
Expected F1 on SEVIR test split: **0.88** (Table 1 of paper).

---

## Reproducing Table 1 — Perception Encoder Ablation

Table 1 of the paper compares four encoder configurations.
All four can be trained via the `--model_type` flag:

```bash
bash scripts/train_perception_ablation.sh

# Equivalent individual commands:
python src/models/vit_encoder.py --model_type radar_cnn      # Radar-only CNN
python src/models/vit_encoder.py --model_type multimodal_cnn # Multi-modal CNN
python src/models/vit_encoder.py --model_type vit_single     # ViT, radar only
python src/models/vit_encoder.py --model_type vit_multimodal # ViT, multi-modal (ours)
```

Expected results on SEVIR test split (Table 1):

| Model | Acc | Prec | Recall | F1 |
|-------|-----|------|--------|----|
| Radar-only CNN | 0.79 | 0.77 | 0.78 | 0.77 |
| Multi-modal CNN | 0.85 | 0.84 | 0.83 | 0.84 |
| ViT (single-modal, radar) | 0.86 | 0.85 | 0.84 | 0.85 |
| **ViT (multi-modal, fine-tuned)** | **0.89** | **0.88** | **0.87** | **0.88** |

All pairwise F1 differences significant at p < 0.05 (paired t-test,
5 evaluation folds).

---

## Expected Results

### Table 2 — RL Policy Comparison

Running `bash scripts/train_full.sh` followed by
`bash scripts/train_baselines.sh` reproduces the following
(mean ± std, 5 seeds, 500 evaluation episodes):

| Method | Reward ↑ | Violation Rate ↓ |
|--------|----------|------------------|
| Heuristic | 42.1 ± 1.8 | 18.3% |
| DQN | 55.6 ± 3.2 | 14.7% |
| IPPO | 63.4 ± 4.1 | 12.1% |
| QMIX | 69.8 ± 3.7 | 10.5% |
| MAPPO | 74.3 ± 3.0 | 8.9% |
| CPO | 71.2 ± 2.8 | 4.1% |
| **Ours (Lagrangian CTDE)** | **81.5 ± 2.6** | **2.3%** |

All improvements of the proposed method over baselines are
statistically significant at p < 0.05 (paired t-test across seeds).

### Table 3 — Decision Accuracy (Proposed Method)

| Scenario | Decision Accuracy |
|----------|-------------------|
| Storm warning | 0.91 ± 0.02 |
| Flood risk response | 0.87 ± 0.03 |
| Evacuation recommendation | 0.84 ± 0.04 |

### Table 4 — Ablation Study

| Variant | Reward | VR |
|---------|--------|----|
| w/o CTDE (independent critics) | 66.3 ± 3.9 | 11.8% |
| w/o Lagrangian (unconstrained) | 75.1 ± 2.9 | 9.2% |
| w/o multi-modal perception* | 77.4 ± 3.1 | 3.8% |
| **Full model (Ours)** | **81.5 ± 2.6** | **2.3%** |

*Scalar state uses only contextual variables (l_t, v_t) without φ_t.

Example training curves are in `results/example_results/`.

---

## Related Repositories

This work builds on:

- [MAPPO](https://github.com/marlbenchmark/on-policy) — Yu et al., 2022
- [QMIX / PyMARL](https://github.com/oxwhirl/pymarl) — Rashid et al., 2018
- [CleanRL](https://github.com/vwxyzjn/cleanrl) — Clean single-file RL implementations
- [SEVIR Dataset](https://github.com/MIT-AI-Accelerator/neurips-2020-sevir) — Veillette et al., 2020

---


## License

MIT License. See [LICENSE](LICENSE) for details.
