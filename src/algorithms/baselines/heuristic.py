"""Heuristic rule-based policy (no learning). Threshold-based severity response."""
import numpy as np

from .base import BaseAgent

class HeuristicPolicy(BaseAgent):
    """
    Simple threshold heuristic: maps severity estimate → fixed action per agent.
    No training required.  Paper Table 2: R=42.1, VR=18.3%.
    """
    _ALGO_NAME  = 'heuristic'

    def act(self, obs_dict, deterministic=True):
        acts = {}
        for i in range(1, self.n_agents + 1):
            sev = int(np.clip(round(float(obs_dict[i][0]) * 4), 0, 4))
            if i == 1:
                acts[i] = int(min(3, sev))
            elif i == 2:
                acts[i] = int(max(0, sev - 1))
            else:
                acts[i] = int(max(0, sev - 2))
        return acts

    def update(self):
        return 0.0

    def train(self, env=None, n_episodes=0, checkpoint_dir='checkpoints', save_interval=500):
        """Heuristic requires no training."""
        print(f"    [heuristic] No training needed — rule-based policy.", flush=True)
        self._trained = True
