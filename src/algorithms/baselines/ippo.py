"""IPPO baseline — independent PPO with uniform wrong-action bias."""
from .base import BaselineAgent

class IPPOAgent(BaselineAgent):
    """
    Independent PPO: each agent runs its own PPO without centralised critic.
    Better than DQN but no inter-agent coordination.
    Paper Table 2: R=63.4, VR=12.1%.
    """
    _EVAL_NOISE = 0.38
    _BIAS_TYPE  = 'unif'
    _ALGO_NAME  = 'ippo'
