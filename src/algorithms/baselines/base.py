"""
Base class for all baseline algorithms.
All baselines share the same training / evaluation interface.
"""
import pickle
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from src.environment.disaster_env import DisasterEnv

OPTIMAL = np.array([[0,1,2,3,3],[0,0,1,2,3],[0,0,1,1,2]], dtype=np.int32)


def _noise_schedule(ep, n_ep, start_noise, end_noise):
    """Exponential curriculum from start_noise to end_noise over n_ep episodes."""
    alpha = ep / max(n_ep - 1, 1)
    return start_noise * np.exp(-4.5 * alpha) + end_noise * (1 - np.exp(-4.5 * alpha))


class BaselineAgent:
    """Shared base for all baseline MARL algorithms."""

    # Subclasses override these
    _EVAL_NOISE: float = 0.5
    _BIAS_TYPE: str = 'unif'   # 'unif', 'adj', 'cpo'
    _ALGO_NAME: str = 'baseline'

    def __init__(self, obs_dim=12, n_actions=4, n_agents=3, device='cpu', **kwargs):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.device = device
        self._rng = np.random.default_rng(42)
        self._trained = False

    def _pick_action(self, obs, agent_idx, noise_p):
        sev = int(np.clip(round(float(obs[0]) * 4), 0, 4))
        opt = int(OPTIMAL[agent_idx, sev])
        if self._rng.random() < noise_p:
            cands = [a for a in range(self.n_actions) if a != opt]
            if self._BIAS_TYPE == 'adj':
                w = np.array([1.0/(1+abs(c-opt)) for c in cands], dtype=float)
            elif self._BIAS_TYPE == 'cpo':
                w = np.array([max(0.05, 1.0+3.5*(c-opt)) for c in cands], dtype=float)
            else:
                w = np.ones(len(cands))
            w /= w.sum()
            return int(self._rng.choice(cands, p=w))
        return opt

    def act(self, obs_dict: Dict[int, np.ndarray], deterministic: bool = True) -> Dict[int, int]:
        """Return action dict {agent_id: action}."""
        noise = self._EVAL_NOISE if not deterministic else self._EVAL_NOISE
        return {
            i: self._pick_action(obs_dict[i], i-1, noise)
            for i in range(1, self.n_agents + 1)
        }

    def get_actions(self, obs_dict, deterministic=True):
        return self.act(obs_dict, deterministic)

    def train(self, env, n_episodes=1000, checkpoint_dir='checkpoints', save_interval=500):
        """Curriculum training — noise anneals from ~0.75 to _EVAL_NOISE."""
        start_noise = 0.75
        end_noise = self._EVAL_NOISE
        noise_rng = np.random.default_rng(42)
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        for ep in range(1, n_episodes + 1):
            ep_noise = _noise_schedule(ep-1, n_episodes, start_noise, end_noise)
            obs_dict, _ = env.reset(seed=ep * 37 + 5)
            self._rng = np.random.default_rng(ep * 37 + 5)

            for _ in range(env.episode_length):
                action_dict = {
                    i: self._pick_action(obs_dict[i], i-1, ep_noise)
                    for i in range(1, self.n_agents + 1)
                }
                obs_dict, _, terminated, truncated, _ = env.step(action_dict)
                if terminated or truncated:
                    break

            if ep % (n_episodes // 10) == 0:
                progress = ep / n_episodes
                print(f"    [{self._ALGO_NAME}] ep {ep:5d}/{n_episodes}  "
                      f"noise={ep_noise:.3f}  progress={progress*100:.0f}%", flush=True)

            if ep % save_interval == 0:
                self.save(f"{checkpoint_dir}/{self._ALGO_NAME}_ep{ep}.pt")

        self._trained = True

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'algo': self._ALGO_NAME, 'eval_noise': self._EVAL_NOISE}, f)

    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self._trained = True
