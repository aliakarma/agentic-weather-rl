"""
run_train.py — mirrors colab_train.ipynb cell by cell.
All 7 code cells execute sequentially; produces:
  - results/example_results/training_curves.pdf
  - checkpoints/marl_policy_short_seed42.pt
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)

import sys
import os
import time

# ── mock torch so "import torch" / torch.cuda.is_available() works ─────────
import importlib.util
if importlib.util.find_spec("torch") is None:
    sys.path.insert(0, os.path.dirname(__file__))   # pick up torch.py mock

sys.path.insert(0, ".")
os.makedirs("results/example_results", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

import shutil, subprocess, platform
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

# ── CELL 0 · Setup ────────────────────────────────────────────────────────────
print("=" * 60)
print("CELL 0 · Setup")
print("=" * 60)
if shutil.which("nvidia-smi"):
    r = subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],
                       capture_output=True, text=True)
    print("GPU:", r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else "Unavailable")
else:
    print("GPU: nvidia-smi not found (CPU mode)")
print("Python:", sys.version.split()[0])

# ── CELL 1 · Setup (torch / modules) ─────────────────────────────────────────
print()
print("=" * 60)
print("CELL 1 · Modules & Device")
print("=" * 60)
import torch   # uses mock if real torch absent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
N_EPISODES = 100

print(f"Device     : {DEVICE}")
print(f"Seed       : {SEED}")
print(f"Episodes   : {N_EPISODES}")
print()
print("Key hyperparameters (Table 3):")
print("  lr=3e-4  γ=0.99  GAE λ=0.95  ε=0.2  entropy=0.01")
print("  actor hidden=256  critic hidden=512  mini-batches=4")
print("  constraint d_i=0.10  λ_lr=1e-3")

from src.environment.disaster_env import DisasterEnv
from src.algorithms.lagrangian_ctde import LagrangianCTDE, LagrangianCTDEConfig
from src.train import train
print("✓ Modules loaded")

# ── CELL 2 · Initialise Environment & Agent ───────────────────────────────────
print()
print("=" * 60)
print("CELL 2 · Initialise Environment & Agent")
print("=" * 60)
env = DisasterEnv(seed=SEED)

config = LagrangianCTDEConfig(
    n_episodes    = N_EPISODES,
    device        = DEVICE,
    seed          = SEED,
    log_interval  = 5,
    eval_interval = 20,
    save_interval = 50,
    checkpoint_dir = "checkpoints/",
)

agent = LagrangianCTDE(env=env, config=config)

n_params = sum(
    sum(p.size for p in net.parameters())
    for net in list(agent._actors.values()) + [agent._critic]
)
print(f"✓ LagrangianCTDE initialised")
print(f"  Total parameters: {n_params:,}")
print(f"  Actors: {len(agent._actors)} × ActorNetwork (obs_dim=12 → 4 actions)")
print(f"  Critic: CriticNetwork (state_dim=36 → V^R + 3×V^C)")

# ── CELL 3 · Training Loop ────────────────────────────────────────────────────
print()
print("=" * 60)
print("CELL 3 · Training Loop")
print("=" * 60)
print(f"Starting training: {N_EPISODES} episodes on {DEVICE}...")
print("─" * 55)

t0 = time.time()
history = agent.train(n_episodes=N_EPISODES)
elapsed = time.time() - t0

print()
print(f"Training complete in {elapsed/60:.1f} min")
print(f"Final 10-ep mean reward        : {sum(s.total_reward  for s in history[-10:]) / 10:.2f}")
print(f"Final 10-ep mean violation rate: {sum(s.violation_rate for s in history[-10:]) / 10:.4f}")

# ── CELL 4 · Plot Training Curves ─────────────────────────────────────────────
print()
print("=" * 60)
print("CELL 4 · Plot Training Curves")
print("=" * 60)

episodes = [s.episode       for s in history]
rewards  = [s.total_reward  for s in history]
vrs      = [s.violation_rate for s in history]
kls      = [s.approx_kl     for s in history]
v_losses = [s.value_loss     for s in history]
lambdas  = [[s.lambdas[i] for s in history] for i in range(len(history[0].lambdas))]

def smooth(x, w=10):
    if len(x) < w: return np.array(x)
    return np.convolve(x, np.ones(w)/w, mode='valid')

sw = 5

fig = plt.figure(figsize=(15, 10))
fig.suptitle(
    f"LagrangianCTDE Training Dynamics\n"
    f"({N_EPISODES} episodes, seed={SEED}, device={DEVICE})",
    fontsize=13, fontweight="bold",
)
gs = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.38)

# 1. Reward
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(episodes, rewards, alpha=0.3, lw=1, color="#2563EB")
if len(rewards) >= sw:
    ax1.plot(episodes[sw-1:], smooth(rewards, sw), lw=2, color="#2563EB", label="Smoothed")
ax1.set_title("Cumulative Reward"); ax1.set_xlabel("Episode"); ax1.set_ylabel("Reward")
ax1.xaxis.set_major_locator(MaxNLocator(5)); ax1.grid(alpha=0.3)

# 2. Violation Rate
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(episodes, vrs, alpha=0.3, lw=1, color="#DC2626")
if len(vrs) >= sw:
    ax2.plot(episodes[sw-1:], smooth(vrs, sw), lw=2, color="#DC2626")
ax2.axhline(0.10, ls="--", color="gray", lw=1.2, label="Threshold d=0.10")
ax2.set_title("Violation Rate"); ax2.set_xlabel("Episode"); ax2.set_ylabel("VR")
ax2.legend(fontsize=8); ax2.xaxis.set_major_locator(MaxNLocator(5)); ax2.grid(alpha=0.3)

# 3. Lagrange Multipliers
ax3 = fig.add_subplot(gs[0, 2])
AGENT_COLORS = ["#7C3AED", "#0891B2", "#059669"]
AGENT_LABELS = ["λ₁ Storm", "λ₂ Flood", "λ₃ Evacuation"]
for lam_series, color, label in zip(lambdas, AGENT_COLORS, AGENT_LABELS):
    ax3.plot(episodes, lam_series, lw=1.5, color=color, label=label, alpha=0.85)
ax3.set_title("Lagrange Multipliers λᵢ"); ax3.set_xlabel("Episode"); ax3.set_ylabel("λᵢ")
ax3.legend(fontsize=7); ax3.xaxis.set_major_locator(MaxNLocator(5)); ax3.grid(alpha=0.3)

# 4. Value Loss
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(episodes, v_losses, alpha=0.3, lw=1, color="#D97706")
if len(v_losses) >= sw:
    ax4.plot(episodes[sw-1:], smooth(v_losses, sw), lw=2, color="#D97706")
ax4.set_title("Centralised Critic Loss"); ax4.set_xlabel("Episode"); ax4.set_ylabel("Value Loss")
ax4.xaxis.set_major_locator(MaxNLocator(5)); ax4.grid(alpha=0.3)

# 5. KL Divergence
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(episodes, kls, alpha=0.3, lw=1, color="#0F766E")
if len(kls) >= sw:
    ax5.plot(episodes[sw-1:], smooth(kls, sw), lw=2, color="#0F766E")
ax5.axhline(0.02, ls="--", color="gray", lw=1.2, label="Target KL=0.02")
ax5.set_title("Approx. KL Divergence"); ax5.set_xlabel("Episode"); ax5.set_ylabel("KL")
ax5.legend(fontsize=8); ax5.xaxis.set_major_locator(MaxNLocator(5)); ax5.grid(alpha=0.3)

# 6. Reward vs VR scatter
ax6 = fig.add_subplot(gs[1, 2])
sc = ax6.scatter(vrs, rewards, c=episodes, cmap="viridis", s=15, alpha=0.6)
plt.colorbar(sc, ax=ax6, label="Episode")
ax6.axvline(0.10, ls="--", color="gray", lw=1.2, label="d=0.10")
ax6.set_title("Reward vs Violation Rate"); ax6.set_xlabel("Violation Rate"); ax6.set_ylabel("Reward")
ax6.legend(fontsize=8); ax6.grid(alpha=0.3)

out_plot = "results/example_results/training_curves.pdf"
plt.savefig(out_plot, bbox_inches="tight", dpi=150)
plt.close()
print(f"Training curves saved to {out_plot}")

# ── CELL 5 · Quick Evaluation ─────────────────────────────────────────────────
print()
print("=" * 60)
print("CELL 5 · Quick Evaluation (20 episodes)")
print("=" * 60)
from src.evaluate import evaluate

eval_env = DisasterEnv(seed=10000)
metrics = evaluate(
    agent=agent, env=eval_env, n_episodes=20, seed_offset=10000,
    deterministic=True, output_dir="results/example_results",
    algo_name="lagrangian_ctde_short",
)

print()
print("Post-training evaluation (20 episodes):")
print(f"  Reward         : {metrics['reward_mean']:.2f} ± {metrics['reward_std']:.2f}")
print(f"  Violation Rate : {metrics['violation_rate_mean']:.4f}")
print()
print("Note: Full paper numbers use 1000 training episodes × 5 seeds × 500 eval episodes.")
print("See colab_full.ipynb for complete reproduction.")

# ── CELL 6 · Save Checkpoint ──────────────────────────────────────────────────
print()
print("=" * 60)
print("CELL 6 · Save Checkpoint")
print("=" * 60)
CKPT_PATH = f"checkpoints/marl_policy_short_seed{SEED}.pt"
agent.save_checkpoint(CKPT_PATH)
print(f"✓ Checkpoint saved: {CKPT_PATH}")
print()
print("To evaluate this checkpoint later:")
print("  python -m src.evaluate --algo lagrangian_ctde \\")
print(f"      --checkpoint {CKPT_PATH} --episodes 50")

print()
print("=" * 60)
print("Training demo complete. All cells executed successfully.")
print(f"Training curves → {out_plot}")
print(f"Checkpoint      → {CKPT_PATH}")
print("=" * 60)
