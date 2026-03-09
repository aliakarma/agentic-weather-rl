"""
run_full.py — mirrors colab_full.ipynb (all 6 code cells) end-to-end.
Produces:
  - checkpoints/marl_policy_seed{N}.pt          (5 seeds)
  - checkpoints/{baseline}_final.pt             (6 baselines)
  - results/example_results/table2_reproduction.json
  - results/example_results/table2_comparison.pdf
"""
import os as _os, sys as _sys
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0, _ROOT)
_os.chdir(_ROOT)

import os, sys, time, json, shutil, subprocess, platform

# ── torch mock if PyTorch not installed ──────────────────────────────────────
import importlib.util
if importlib.util.find_spec("torch") is None:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

os.makedirs("results/example_results", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

import torch    # mock or real
from src.environment.disaster_env import DisasterEnv
from src.algorithms.lagrangian_ctde import LagrangianCTDE, LagrangianCTDEConfig
from src.train import train
from src.evaluate import evaluate, load_agent_from_checkpoint

# ── CELL 0 · Setup ────────────────────────────────────────────────────────────
print("=" * 65)
print("CELL 0 · Setup")
print("=" * 65)
if shutil.which("nvidia-smi"):
    r = subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],
                       capture_output=True, text=True)
    print("GPU:", r.stdout.strip() if r.returncode==0 and r.stdout.strip() else "Unavailable")
else:
    print("GPU: nvidia-smi not found (CPU mode is supported)")
print("Python:", sys.version.split()[0])
print("Platform:", platform.platform())

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✓ Setup complete | device={DEVICE}")

# ── CELL 1 · Configuration ────────────────────────────────────────────────────
print()
print("=" * 65)
print("CELL 1 · Configuration")
print("=" * 65)
SEEDS            = [42, 43, 44, 45, 46]
N_EPISODES_TRAIN = 1000
N_EPISODES_EVAL  = 500
CHECKPOINT_DIR   = "checkpoints"
RESULTS_DIR      = "results/example_results"

print(f"Seeds          : {SEEDS}")
print(f"Train episodes : {N_EPISODES_TRAIN}")
print(f"Eval episodes  : {N_EPISODES_EVAL}")
print(f"Device         : {DEVICE}")

# ── CELL 2 · Train LagrangianCTDE (5 seeds) ───────────────────────────────────
print()
print("=" * 65)
print("CELL 2 · Train LagrangianCTDE (5 Seeds)")
print("=" * 65)
all_seed_metrics = {}
t_total = time.time()

for seed in SEEDS:
    print(f"\n{'─'*55}")
    print(f"  Training LagrangianCTDE  seed={seed}  ({SEEDS.index(seed)+1}/{len(SEEDS)})")
    print(f"{'─'*55}")

    env = DisasterEnv(seed=seed)
    config = LagrangianCTDEConfig(
        n_episodes    = N_EPISODES_TRAIN,
        device        = DEVICE,
        seed          = seed,
        log_interval  = 50,
        eval_interval = 100,
        save_interval = 500,
        checkpoint_dir = CHECKPOINT_DIR,
    )
    agent = LagrangianCTDE(env=env, config=config)

    t0 = time.time()
    history = agent.train(n_episodes=N_EPISODES_TRAIN)
    elapsed = time.time() - t0

    ckpt = f"{CHECKPOINT_DIR}/marl_policy_seed{seed}.pt"
    agent.save_checkpoint(ckpt)

    eval_env = DisasterEnv(seed=10000 + seed)
    metrics = evaluate(
        agent=agent, env=eval_env,
        n_episodes=N_EPISODES_EVAL,
        seed_offset=10000 + seed,
        deterministic=True,
        output_dir=RESULTS_DIR,
        algo_name=f"lagrangian_ctde_seed{seed}",
    )
    all_seed_metrics[seed] = metrics
    print(f"  Seed {seed} | R={metrics['reward_mean']:.2f} "
          f"VR={metrics['violation_rate_mean']:.4f} | {elapsed/60:.1f} min")

print(f"\nAll seeds complete. Total time: {(time.time()-t_total)/60:.1f} min")

# ── CELL 3 · Train All Baselines ──────────────────────────────────────────────
print()
print("=" * 65)
print("CELL 3 · Train All Baselines")
print("=" * 65)

from src.algorithms.baselines.heuristic import HeuristicPolicy
from src.algorithms.baselines.dqn       import DQNAgent
from src.algorithms.baselines.ippo      import IPPOAgent
from src.algorithms.baselines.qmix      import QMIXAgent
from src.algorithms.baselines.mappo     import MAPPOAgent
from src.algorithms.baselines.cpo       import CPOAgent

BASELINES = {
    "heuristic": None,
    "dqn":       DQNAgent,
    "ippo":      IPPOAgent,
    "qmix":      QMIXAgent,
    "mappo":     MAPPOAgent,
    "cpo":       CPOAgent,
}
BASELINE_KWARGS = dict(obs_dim=12, n_actions=4, n_agents=3, device=DEVICE)
QMIX_KWARGS  = dict(**BASELINE_KWARGS, state_dim=24)
MAPPO_KWARGS = dict(obs_dim=12, state_dim=24, n_actions=4, n_agents=3, device=DEVICE)

baseline_metrics = {}

for name, cls in BASELINES.items():
    print(f"\n{'─'*50}")
    print(f"  Baseline: {name.upper()}")
    print(f"{'─'*50}")
    t0 = time.time()

    env = DisasterEnv(seed=42)

    if name == "heuristic":
        baseline_agent = HeuristicPolicy()
        class _HWrapper:
            def __init__(self, p): self._p = p
            def get_actions(self, obs, deterministic=True): return self._p.act(obs)
        eval_agent = _HWrapper(baseline_agent)
    else:
        kwargs = QMIX_KWARGS if name=="qmix" else (MAPPO_KWARGS if name=="mappo" else BASELINE_KWARGS)
        baseline_agent = cls(**kwargs)
        baseline_agent.train(
            env=env,
            n_episodes=N_EPISODES_TRAIN,
            checkpoint_dir=CHECKPOINT_DIR,
            save_interval=500,
        )
        ckpt = f"{CHECKPOINT_DIR}/{name}_final.pt"
        if hasattr(baseline_agent, "save"):
            baseline_agent.save(ckpt)
        class _AWrapper:
            def __init__(self, a): self._a = a
            def get_actions(self, obs, deterministic=True):
                return self._a.act(obs, deterministic=deterministic)
        eval_agent = _AWrapper(baseline_agent)

    eval_env = DisasterEnv(seed=10042)
    metrics = evaluate(
        agent=eval_agent, env=eval_env,
        n_episodes=N_EPISODES_EVAL,
        seed_offset=10042,
        deterministic=True,
        output_dir=RESULTS_DIR,
        algo_name=name,
    )
    baseline_metrics[name] = metrics
    print(f"  {name}: R={metrics['reward_mean']:.2f}  VR={metrics['violation_rate_mean']:.4f}  "
          f"({(time.time()-t0)/60:.1f} min)")

print("\nAll baselines complete.")

# ── CELL 4 · Export Table 2 ───────────────────────────────────────────────────
print()
print("=" * 65)
print("CELL 4 · Export Table 2")
print("=" * 65)

PAPER_TABLE2 = {
    "heuristic":       {"reward": 42.1, "reward_std": 1.8,  "vr": 18.3},
    "dqn":             {"reward": 55.6, "reward_std": 3.2,  "vr": 14.7},
    "ippo":            {"reward": 63.4, "reward_std": 4.1,  "vr": 12.1},
    "qmix":            {"reward": 69.8, "reward_std": 3.7,  "vr": 10.5},
    "mappo":           {"reward": 74.3, "reward_std": 3.0,  "vr":  8.9},
    "cpo":             {"reward": 71.2, "reward_std": 2.8,  "vr":  4.1},
    "lagrangian_ctde": {"reward": 81.5, "reward_std": 2.6,  "vr":  2.3},
}

lctde_rewards = [all_seed_metrics[s]["reward_mean"]         for s in SEEDS]
lctde_vrs     = [all_seed_metrics[s]["violation_rate_mean"] for s in SEEDS]
agg_ctde = {
    "reward_mean":          np.mean(lctde_rewards),
    "reward_std":           np.std(lctde_rewards),
    "violation_rate_mean":  np.mean(lctde_vrs),
    "violation_rate_std":   np.std(lctde_vrs),
}

all_results = {**baseline_metrics, "lagrangian_ctde": agg_ctde}

print("=" * 75)
print(f"  {'Method':<22} {'Reward':>10} {'Std':>8} {'VR%':>10} {'vs Heuristic':>14}")
print("  " + "─" * 68)

ORDER = ["heuristic","dqn","ippo","qmix","mappo","cpo","lagrangian_ctde"]
baseline_r = all_results.get("heuristic", {}).get("reward_mean", 1.0)
for method in ORDER:
    m  = all_results.get(method, {})
    r  = m.get("reward_mean", float("nan"))
    rs = m.get("reward_std",  float("nan"))
    vr = m.get("violation_rate_mean", float("nan")) * 100
    delta = ((r - baseline_r) / baseline_r * 100) if baseline_r else 0
    marker = "◀ proposed" if method == "lagrangian_ctde" else ""
    print(f"  {method:<22} {r:>10.1f} ±{rs:<6.1f} {vr:>9.1f}% {delta:>+13.1f}%  {marker}")

print("=" * 75)

agg_all = {}
for method in ORDER:
    m = all_results.get(method, {})
    agg_all[method] = {
        "reward_mean":          m.get("reward_mean", None),
        "reward_std":           m.get("reward_std",  None),
        "violation_rate_mean":  m.get("violation_rate_mean", None),
    }

out_path = Path(RESULTS_DIR) / "table2_reproduction.json"
with open(out_path, "w") as f:
    json.dump(agg_all, f, indent=2)
print(f"\nResults saved to: {out_path}")

# ── CELL 5 · Plot Table 2 Comparison ─────────────────────────────────────────
print()
print("=" * 65)
print("CELL 5 · Plot Table 2 Comparison")
print("=" * 65)

ORDER_DISPLAY = ["Heuristic","DQN","IPPO","QMIX","MAPPO","CPO","LagrangianCTDE\n(Proposed)"]
ORDER_KEYS    = ["heuristic","dqn","ippo","qmix","mappo","cpo","lagrangian_ctde"]

ours_r  = [all_results.get(k,{}).get("reward_mean", 0)             for k in ORDER_KEYS]
ours_rs = [all_results.get(k,{}).get("reward_std",  0)             for k in ORDER_KEYS]
ours_vr = [all_results.get(k,{}).get("violation_rate_mean", 0)*100 for k in ORDER_KEYS]

x = np.arange(len(ORDER_KEYS))
# Colour scheme: unconstrained baselines blue, CPO orange, proposed red
COLORS = ["#60A5FA","#60A5FA","#60A5FA","#60A5FA","#60A5FA","#F59E0B","#DC2626"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Table 2 — MARL Method Comparison", fontsize=13, fontweight="bold")

# ── Reward ────────────────────────────────────────────────────────────────────
bars1 = ax1.bar(x, ours_r, color=COLORS, alpha=0.88,
                yerr=ours_rs, capsize=4,
                error_kw=dict(lw=1.5, ecolor="#374151"),
                edgecolor="white", linewidth=0.8)
# Value labels
for bar, val in zip(bars1, ours_r):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 1,
             f"{val:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
# Heuristic baseline reference line
ax1.axhline(ours_r[0], ls="--", color="#6B7280", lw=1.1,
            label=f"Heuristic baseline={ours_r[0]:.1f}")
ax1.annotate("★ Proposed",
             xy=(x[-1], ours_r[-1]),
             xytext=(x[-1] - 1.2, ours_r[-1] * 1.04),
             fontsize=8, color="#DC2626",
             arrowprops=dict(arrowstyle="->", color="#DC2626", lw=1.2))
ax1.set_xticks(x)
ax1.set_xticklabels(ORDER_DISPLAY, fontsize=8, ha="center")
ax1.set_ylabel("Mean Cumulative Reward")
ax1.set_title("Reward ↑ (higher is better)")
ax1.grid(axis="y", alpha=0.3)
ax1.set_ylim(0, max(ours_r) * 1.18)

# Legend for colour coding
ax1.legend(handles=[
    mpatches.Patch(facecolor="#60A5FA", alpha=0.88, label="Unconstrained baselines"),
    mpatches.Patch(facecolor="#F59E0B", alpha=0.88, label="Constrained baseline (CPO)"),
    mpatches.Patch(facecolor="#DC2626", alpha=0.88, label="Proposed (LagrangianCTDE)"),
    ax1.get_lines()[0],
], fontsize=8, loc="upper left")

# ── Violation Rate ────────────────────────────────────────────────────────────
bars2 = ax2.bar(x, ours_vr, color=COLORS, alpha=0.88,
                edgecolor="white", linewidth=0.8)
# Value labels
for bar, val in zip(bars2, ours_vr):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 0.2,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
ax2.axhline(10.0, ls="--", color="#DC2626", lw=1.4, label="Constraint threshold d=10%")
ax2.set_xticks(x)
ax2.set_xticklabels(ORDER_DISPLAY, fontsize=8, ha="center")
ax2.set_ylabel("Constraint Violation Rate (%)")
ax2.set_title("Violation Rate ↓ (lower is better)")
ax2.legend(fontsize=8)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
out_fig = f"{RESULTS_DIR}/table2_comparison.pdf"
plt.savefig(out_fig, bbox_inches="tight", dpi=150)
plt.close()
print(f"Figure saved to {out_fig}")

print()
print("=" * 65)
print("Full experiment complete. All cells executed successfully.")
print(f"  Table 2 JSON → {out_path}")
print(f"  Comparison plot → {out_fig}")
print("=" * 65)
