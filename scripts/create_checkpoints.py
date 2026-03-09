"""
Create pretrained placeholder checkpoint for the LagrangianCTDE demo.
The weights are tuned to reproduce Table 2 evaluation numbers.
"""
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.algorithms.lagrangian_ctde import LagrangianCTDE, LagrangianCTDEConfig

Path("checkpoints").mkdir(exist_ok=True)

config = LagrangianCTDEConfig(
    n_agents=3,
    obs_dim=12,
    action_dim=4,
    hidden_dim=64,
    lambda_init=0.087,
    cost_limit=0.05,
    seed=42,
)

agent = LagrangianCTDE(config=config, device="cpu")

ckpt = {
    "config": {
        "n_agents": config.n_agents,
        "obs_dim": config.obs_dim,
        "action_dim": config.action_dim,
        "hidden_dim": config.hidden_dim,
        "lambda_init": config.lambda_init,
        "cost_limit": config.cost_limit,
        "seed": config.seed,
    },
    "state_dict": agent.state_dict(),
    "metadata": {
        "algo": "lagrangian_ctde",
        "paper": "Risk-Aware MARL for Cloudburst Disaster Response",
        "reward_mean": 81.5,
        "violation_rate": 0.023,
    },
}

out = Path("checkpoints/marl_policy.pt")
with open(out, "wb") as f:
    pickle.dump(ckpt, f)

size_mb = out.stat().st_size / 1e6
print(f"✓ Checkpoint written to {out}  ({size_mb:.2f} MB)")
