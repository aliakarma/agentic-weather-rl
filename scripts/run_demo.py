"""
Full demo runner — executes all notebook cells sequentially.
Produces results/example_results/demo_trajectory.pdf and console output.
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)

import sys
import os
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, ".")
os.makedirs("results/example_results", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── Cell 0: Runtime Check ─────────────────────────────────────────────────────
import platform, shutil, subprocess
print("=" * 60)
print("CELL 0 · Runtime Check")
print("=" * 60)
if shutil.which("nvidia-smi"):
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                       capture_output=True, text=True)
    print("GPU detected:", r.stdout.strip() if r.returncode == 0 else "none")
else:
    print("nvidia-smi not found. CPU will be used (fine for evaluation).")
print(f"Python {sys.version}")
print(f"Platform: {platform.platform()}")

# ── Cell 1: Module verification ───────────────────────────────────────────────
print()
print("=" * 60)
print("CELL 1 · Module Verification")
print("=" * 60)
from src.environment.disaster_env import DisasterEnv
from src.models.actor import ActorNetwork
from src.models.critic import CriticNetwork
from src.algorithms.lagrangian_ctde import LagrangianCTDE, LagrangianCTDEConfig
from src.evaluate import evaluate
print("✓ All core modules import successfully")

# ── Cell 2: Checkpoint ────────────────────────────────────────────────────────
print()
print("=" * 60)
print("CELL 2 · Checkpoint")
print("=" * 60)
from pathlib import Path
CHECKPOINT = "checkpoints/marl_policy.pt"
if not Path(CHECKPOINT).exists():
    print("Checkpoint not found — generating...")
    subprocess.run([sys.executable, "scripts/create_checkpoints.py"], check=True)
print(f"✓ Checkpoint found: {CHECKPOINT}")
print(f"  Size: {Path(CHECKPOINT).stat().st_size / 1e6:.2f} MB")

# ── Cell 3: Load Agent ────────────────────────────────────────────────────────
print()
print("=" * 60)
print("CELL 3 · Load Agent")
print("=" * 60)
from src.evaluate import load_agent_from_checkpoint

DEVICE = "cpu"
print(f"Using device: {DEVICE}")
agent = load_agent_from_checkpoint(CHECKPOINT, algo="lagrangian_ctde", device=DEVICE)
print("✓ Agent loaded from checkpoint")

# ── Cell 4: Evaluate (20 episodes) ───────────────────────────────────────────
print()
print("=" * 60)
print("CELL 4 · Evaluation (20 episodes)")
print("=" * 60)
env = DisasterEnv(seed=10000)
metrics = evaluate(
    agent=agent, env=env, n_episodes=20, seed_offset=10000,
    deterministic=True, output_dir="results/example_results",
    algo_name="lagrangian_ctde_demo",
)
print()
print("=" * 55)
print("DEMO RESULTS (20 episodes)")
print("=" * 55)
print(f"  Reward          : {metrics['reward_mean']:.2f} ± {metrics['reward_std']:.2f}")
print(f"  Violation Rate  : {metrics['violation_rate_mean']:.4f} ± {metrics['violation_rate_std']:.4f}")
for i, name in zip(range(1, 4), ["Storm", "Flood", "Evacuation"]):
    m = metrics[f'decision_accuracy_{i}_mean']
    s = metrics[f'decision_accuracy_{i}_std']
    print(f"  DA Agent {i} ({name:<10}): {m:.4f} ± {s:.4f}")
print()
print("Paper Table 2 targets (5-seed, 500 eps):")
print("  Reward=81.5±2.6  VR=2.3%  DA: 0.91/0.87/0.84")
print("=" * 55)

# ── Cell 5: Trajectory Visualisation ─────────────────────────────────────────
print()
print("=" * 60)
print("CELL 5 · Trajectory Visualisation")
print("=" * 60)
env_vis = DisasterEnv(seed=99999)
obs_dict, _ = env_vis.reset(seed=99999)
agent._rng = np.random.default_rng(99999)

rewards_ep, severity_ep, actions_ep = [], [], []
ACTION_NAMES = {0: "No-op", 1: "Warn", 2: "Deploy", 3: "Evacuate"}
AGENT_NAMES  = {1: "Storm", 2: "Flood", 3: "Evacuation"}

for t in range(env_vis.episode_length):
    action_dict = agent.get_actions(obs_dict, deterministic=True)
    obs_dict, reward, terminated, truncated, info = env_vis.step(action_dict)
    rewards_ep.append(reward)
    severity_ep.append(info.get("risk_level", 0.0))
    actions_ep.append([action_dict[i] for i in range(1, 4)])
    if terminated or truncated:
        break

actions_arr = np.array(actions_ep)
T = len(rewards_ep)
ts = np.arange(T)

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
fig.suptitle("Single Episode — Pretrained LagrangianCTDE Agent", fontsize=13, fontweight="bold")

ax1 = axes[0]
ax1.plot(ts, np.cumsum(rewards_ep), color="#2563EB", lw=2, label="Cumulative Reward")
ax1r = ax1.twinx()
ax1r.fill_between(ts, severity_ep, alpha=0.25, color="#DC2626", label="Risk Level")
ax1r.set_ylim(0, 1.1)
ax1r.set_ylabel("Risk Level", color="#DC2626", fontsize=9)
ax1.set_ylabel("Cumulative Reward")
ax1.set_title("Reward & Risk Level")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1r.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

ax2 = axes[1]
AGENT_COLORS = {1: "#7C3AED", 2: "#0891B2", 3: "#059669"}
for i in range(1, 4):
    ax2.step(ts, actions_arr[:, i - 1] + (i - 1) * 0.1,
             where="post", color=AGENT_COLORS[i],
             lw=1.5, label=f"Agent {i} ({AGENT_NAMES[i]})", alpha=0.85)
ax2.set_yticks([0, 1, 2, 3])
ax2.set_yticklabels(["No-op", "Warn", "Deploy", "Evacuate"])
ax2.set_ylabel("Action")
ax2.set_title("Per-Agent Actions")
ax2.legend(fontsize=8, loc="upper right")
ax2.set_ylim(-0.3, 3.5)

ax3 = axes[2]
colors = ["#16A34A" if r >= 0 else "#DC2626" for r in rewards_ep]
ax3.bar(ts, rewards_ep, color=colors, alpha=0.8, width=0.8)
ax3.axhline(0, color="gray", lw=0.8, ls="--")
ax3.set_ylabel("Step Reward")
ax3.set_xlabel("Timestep")
ax3.set_title("Per-Step Reward")

plt.tight_layout()
out_pdf = "results/example_results/demo_trajectory.pdf"
plt.savefig(out_pdf, bbox_inches="tight", dpi=150)
plt.close()
print(f"Figure saved to {out_pdf}")

# ── Cell 6: Orchestration Demo ────────────────────────────────────────────────
print()
print("=" * 60)
print("CELL 6 · Orchestration Layer Demo")
print("=" * 60)
from src.orchestration.orchestration import Orchestrator, ResponseType

orch = Orchestrator(policy=agent)
env_orch = DisasterEnv(seed=12345)
obs_dict_orch, _ = env_orch.reset(seed=12345)
agent._rng = np.random.default_rng(12345)
orch.reset()

print("Orchestration responses (first 10 timesteps):")
print("-" * 65)
for t in range(10):
    action_dict = agent.get_actions(obs_dict_orch, deterministic=True)
    obs_dict_orch, _, terminated, truncated, info = env_orch.step(action_dict)
    response = orch.step(obs_dict_orch, info=info)
    print(response.summary())
    if terminated or truncated:
        break

print()
print("Episode summary:")
summary = orch.get_episode_summary()
print(f"  Mean severity    : {summary.get('mean_severity', 0):.2f}")
print(f"  Escalations      : {summary.get('escalations', 0)}")
print(f"  Evacuation rate  : {summary.get('evacuation_rate', 0):.3f}")
print(f"  Storm warning    : {summary.get('storm_warning_rate', 0):.3f}")

print()
print("=" * 60)
print("Demo complete. All cells executed successfully.")
print(f"Trajectory plot → {out_pdf}")
print("=" * 60)
