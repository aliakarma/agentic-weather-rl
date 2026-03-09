"""
src/algorithms
==============
RL algorithm package.

Public API
----------
    from src.algorithms.lagrangian_ctde import LagrangianCTDE, LagrangianCTDEConfig
"""

# Corrected import: Removed non-existent PPO-related imports
from src.algorithms.lagrangian_ctde import LagrangianCTDE, LagrangianCTDEConfig

__all__ = [
    # Removed PPOTrainer, RolloutBuffer, compute_gae as they are not defined in src/algorithms/ppo.py
    "LagrangianCTDE",
    "LagrangianCTDEConfig",
]
