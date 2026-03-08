"""
src/algorithms
==============
RL algorithm package.

Public API
----------
    from src.algorithms.ppo             import PPOTrainer, RolloutBuffer, compute_gae
    from src.algorithms.lagrangian_ctde import LagrangianCTDE, LagrangianCTDEConfig
"""

from src.algorithms.ppo import PPOTrainer, RolloutBuffer, compute_gae
from src.algorithms.lagrangian_ctde import LagrangianCTDE, LagrangianCTDEConfig

__all__ = [
    "PPOTrainer",
    "RolloutBuffer",
    "compute_gae",
    "LagrangianCTDE",
    "LagrangianCTDEConfig",
]
