"""Random multi-agent policy baseline for diagnostics."""
from __future__ import annotations

from typing import Dict

import numpy as np


class RandomAgent:
    """Uniform random decentralized policy used as a learnability baseline."""

    def __init__(self, n_agents: int = 3, n_actions: int = 4, seed: int = 42):
        self.n_agents = int(n_agents)
        self.n_actions = int(n_actions)
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

    def get_actions(self, obs_dict: Dict[int, np.ndarray], deterministic: bool = False) -> Dict[int, int]:
        del deterministic
        return {
            i: int(self._rng.integers(0, self.n_actions))
            for i in range(1, self.n_agents + 1)
        }

    def state_dict(self) -> dict:
        return {
            "type": "random",
            "seed": self.seed,
            "n_agents": self.n_agents,
            "n_actions": self.n_actions,
        }
