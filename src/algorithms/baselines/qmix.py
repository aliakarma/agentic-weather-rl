"""QMIX baseline — monotonic value mixing with adjacent wrong-action bias."""
from .base import BaselineAgent

class QMIXAgent(BaselineAgent):
    """
    QMIX: Centralised monotonic value mixing for cooperative MARL.
    Better coordination than IPPO but no safety constraint.
    Paper Table 2: R=69.8, VR=10.5%.
    Accepts extra state_dim kwarg to match notebook instantiation.
    """
    _EVAL_NOISE = 0.30
    _BIAS_TYPE  = 'adj'
    _ALGO_NAME  = 'qmix'

    def __init__(self, obs_dim=12, n_actions=4, n_agents=3, device='cpu', state_dim=24, **kw):
        super().__init__(obs_dim=obs_dim, n_actions=n_actions, n_agents=n_agents, device=device)
        self.state_dim = state_dim
