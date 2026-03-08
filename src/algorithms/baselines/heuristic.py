"""
src/algorithms/baselines/heuristic.py
=======================================
Rule-based heuristic policy for the disaster response environment.

Implements the heuristic baseline from Table 2 of the paper
(reward=42.1±1.8, VR=18.3%).

Each agent applies a fixed threshold rule on the component of the
observation most relevant to its role:

  Agent 1 (Storm Detection):
      if storm_probability > 0.6  → evacuate (3)
      elif storm_probability > 0.3 → warn (1)
      else                         → no-op (0)

  Agent 2 (Flood Risk):
      if rainfall + river_level > 0.8 → deploy (2)
      elif rainfall + river_level > 0.4 → warn (1)
      else                              → no-op (0)

  Agent 3 (Evacuation Planning):
      if vulnerability * compound_risk > 0.5 → evacuate (3)
      elif vulnerability * compound_risk > 0.2 → deploy (2)
      else                                      → no-op (0)

Observation layout (from obs_router.py):
  Agent 1: [phi(0..3), storm_prob(4), rainfall(5), quad_storm(6..9), t(10), res(11)]
  Agent 2: [rainfall(0), river(1), quad_flood_mean(2..5), quad_flood_max(6..9), compound(10), t(11)]
  Agent 3: [agent1_summary(0..3), agent2_summary(4..7), vuln(8), max_combined(9), n_events(10), t(11)]
"""

from __future__ import annotations

import numpy as np
from typing import Dict


class HeuristicPolicy:
    """
    Deterministic threshold-based heuristic policy.

    No training required. Used as the weakest baseline in Table 2.

    Parameters
    ----------
    storm_high  : float  Agent 1 threshold for evacuation (default 0.6)
    storm_low   : float  Agent 1 threshold for warning    (default 0.3)
    flood_high  : float  Agent 2 combined threshold for deployment (default 0.8)
    flood_low   : float  Agent 2 combined threshold for warning    (default 0.4)
    evac_high   : float  Agent 3 compound-risk threshold for evacuation (default 0.5)
    evac_low    : float  Agent 3 compound-risk threshold for deployment (default 0.2)
    """

    # Observation index constants (must match obs_router.py)
    # Agent 1
    A1_STORM_PROB = 4
    A1_RAINFALL   = 5
    # Agent 2
    A2_RAINFALL   = 0
    A2_RIVER      = 1
    A2_COMPOUND   = 10
    # Agent 3
    A3_VULN        = 8
    A3_MAX_COMBINED = 9

    def __init__(
        self,
        storm_high: float = 0.6,
        storm_low:  float = 0.3,
        flood_high: float = 0.8,
        flood_low:  float = 0.4,
        evac_high:  float = 0.5,
        evac_low:   float = 0.2,
    ) -> None:
        self.storm_high = storm_high
        self.storm_low  = storm_low
        self.flood_high = flood_high
        self.flood_low  = flood_low
        self.evac_high  = evac_high
        self.evac_low   = evac_low

    def act(self, obs_dict: Dict[int, np.ndarray]) -> Dict[int, int]:
        """
        Compute heuristic actions for all three agents.

        Parameters
        ----------
        obs_dict : dict[int, np.ndarray]
            Local observations keyed by agent_id {1, 2, 3}.

        Returns
        -------
        actions : dict[int, int]
            Joint action keyed by agent_id.
        """
        return {
            1: self._agent1_action(obs_dict[1]),
            2: self._agent2_action(obs_dict[2]),
            3: self._agent3_action(obs_dict[3]),
        }

    # -----------------------------------------------------------------------

    def _agent1_action(self, obs: np.ndarray) -> int:
        """Storm Detection Agent heuristic."""
        storm_prob = float(obs[self.A1_STORM_PROB])
        if storm_prob > self.storm_high:
            return 3  # evacuate
        elif storm_prob > self.storm_low:
            return 1  # warn
        return 0

    def _agent2_action(self, obs: np.ndarray) -> int:
        """Flood Risk Assessment Agent heuristic."""
        combined = float(obs[self.A2_RAINFALL]) + float(obs[self.A2_RIVER])
        if combined > self.flood_high:
            return 2  # deploy
        elif combined > self.flood_low:
            return 1  # warn
        return 0

    def _agent3_action(self, obs: np.ndarray) -> int:
        """Evacuation Planning Agent heuristic."""
        compound = float(obs[self.A3_VULN]) * float(obs[self.A3_MAX_COMBINED])
        if compound > self.evac_high:
            return 3  # evacuate
        elif compound > self.evac_low:
            return 2  # deploy
        return 0

    # -----------------------------------------------------------------------
    # Compatibility shim so train.py can call .train() on any baseline

    def train(self, *args, **kwargs) -> Dict:
        """No-op: heuristic policy requires no training."""
        return {"note": "HeuristicPolicy requires no training."}

    def save(self, path: str) -> None:
        """No-op: nothing to serialise."""
        pass

    def load(self, path: str) -> None:
        """No-op: nothing to deserialise."""
        pass

    def __repr__(self) -> str:
        return (
            f"HeuristicPolicy("
            f"storm=[{self.storm_low},{self.storm_high}], "
            f"flood=[{self.flood_low},{self.flood_high}], "
            f"evac=[{self.evac_low},{self.evac_high}])"
        )
