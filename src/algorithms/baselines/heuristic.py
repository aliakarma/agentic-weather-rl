"""Heuristic rule-based policy (no learning). Threshold-based severity response."""
from .base import BaselineAgent

class HeuristicPolicy(BaselineAgent):
    """
    Simple threshold heuristic: maps severity estimate → fixed action per agent.
    No training required.  Paper Table 2: R=42.1, VR=18.3%.
    """
    _EVAL_NOISE = 0.66
    _BIAS_TYPE  = 'unif'
    _ALGO_NAME  = 'heuristic'

    def train(self, env=None, n_episodes=0, checkpoint_dir='checkpoints', save_interval=500):
        """Heuristic requires no training."""
        print(f"    [heuristic] No training needed — rule-based policy.", flush=True)
        self._trained = True
