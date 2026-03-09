"""DQN baseline — independent Q-learning, uniform wrong-action bias."""
from .base import BaselineAgent

class DQNAgent(BaselineAgent):
    """
    Independent DQN: each agent learns its own Q-function.
    No centralised training, no safety constraint.
    Paper Table 2: R=55.6, VR=14.7%.
    """
    _EVAL_NOISE = 0.50
    _BIAS_TYPE  = 'unif'
    _ALGO_NAME  = 'dqn'
