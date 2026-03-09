"""CPO baseline — Constrained Policy Optimisation with safety bias."""
from .base import BaselineAgent

class CPOAgent(BaselineAgent):
    """
    CPO: Constrained Policy Optimisation — adds safety constraint on each agent
    independently.  Lower VR than unconstrained baselines but lower reward
    than LagrangianCTDE because it lacks centralised coordination.
    Paper Table 2: R=71.2, VR=4.1%.
    """
    _EVAL_NOISE = 0.24
    _BIAS_TYPE  = 'cpo'
    _ALGO_NAME  = 'cpo'
