"""
src/environment/obs_router.py
==============================
Heterogeneous observation routing for the three cooperative MARL agents.

Implements the observation partitioning described in Section 3.3 of the paper:

  "Each agent receives a partial view of the global state.
   Agent 1 observes atmospheric features and storm probability.
   Agent 2 observes rainfall intensity and river-level indicators.
   Agent 3 receives the outputs forwarded by Agents 1 and 2 alongside
   the regional vulnerability indicator."

Formally: o_{i,t} ⊂ s_t for each i ∈ {1, 2, 3}.
Agents cannot access the full state s_t at execution time (Dec-POMDP).

The observation dimension per agent is 12 (obs_dim from environment.yaml).
Layout is documented precisely in ObservationRouter.OBS_LAYOUT.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

from src.environment.hazard_generator import HazardState


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AgentObservations:
    """Typed container for the three agents' local observations."""
    agent_1: np.ndarray   # (12,) Storm Detection observation
    agent_2: np.ndarray   # (12,) Flood Risk Assessment observation
    agent_3: np.ndarray   # (12,) Evacuation Planning observation

    def as_dict(self) -> Dict[int, np.ndarray]:
        return {1: self.agent_1, 2: self.agent_2, 3: self.agent_3}

    def as_list(self) -> list[np.ndarray]:
        return [self.agent_1, self.agent_2, self.agent_3]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ObservationRouter:
    """
    Routes components of the global state s_t to each agent's local
    observation space o_{i,t} according to Section 3.3 of the paper.

    Each observation vector has dimension OBS_DIM = 12.

    Observation layouts
    -------------------
    Agent 1 — Storm Detection (dims 0–11):
        [0..3]   ViT encoder summary or storm field statistics (4 dims)
        [4]      storm_probability  p^storm_t
        [5]      rainfall_intensity i^rain_t
        [6..9]   storm spatial features (quadrant max values)
        [10]     timestep normalised t / T
        [11]     resource_availability (placeholder, always 1.0 at init)

    Agent 2 — Flood Risk Assessment (dims 0–11):
        [0]      rainfall_intensity i^rain_t
        [1]      river_level        l^river_t
        [2..5]   flood spatial features (quadrant mean values)
        [6..9]   flood spatial features (quadrant max values)
        [10]     combined_risk  (storm_prob * rainfall)
        [11]     timestep normalised t / T

    Agent 3 — Evacuation Planning (dims 0–11):
        [0..3]   agent_1_summary  (forwarded from agent 1 observation [0..3])
        [4..7]   agent_2_summary  (forwarded from agent 2 observation [0..3])
        [8]      vulnerability  v^region_t
        [9]      max_combined_risk  max(storm_field * flood_field)
        [10]     n_active_events (normalised)
        [11]     timestep normalised t / T

    Parameters
    ----------
    grid_size : int
        Side length of the square grid.
    episode_length : int
        Total steps per episode T (used for timestep normalisation).
    obs_dim : int
        Observation vector dimension per agent (must be 12).
    """

    OBS_DIM: int = 12
    N_AGENTS: int = 3

    def __init__(
        self,
        grid_size: int = 20,
        episode_length: int = 100,
        obs_dim: int = 12,
    ) -> None:
        if obs_dim != self.OBS_DIM:
            raise ValueError(
                f"obs_dim must be {self.OBS_DIM}; got {obs_dim}. "
                "Changing observation dimension requires updating all agent networks."
            )
        self.grid_size = grid_size
        self.episode_length = episode_length
        self.obs_dim = obs_dim

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def route(
        self,
        hazard_state: HazardState,
        flat_state: np.ndarray,
        timestep: int,
        resource_availability: float = 1.0,
    ) -> AgentObservations:
        """
        Compute the three agent observations from the global state.

        Parameters
        ----------
        hazard_state : HazardState
            Full hazard state from HazardGenerator.step().
        flat_state : np.ndarray
            24-dimensional flat state vector from HazardGenerator.get_flat_state().
        timestep : int
            Current simulation step (0-indexed).
        resource_availability : float
            Fraction of resources remaining in [0, 1].

        Returns
        -------
        AgentObservations
            Named container of three (12,) observation arrays.
        """
        t_norm = timestep / max(self.episode_length, 1)  # normalised timestep

        obs1 = self._agent1_obs(hazard_state, flat_state, t_norm, resource_availability)
        obs2 = self._agent2_obs(hazard_state, flat_state, t_norm)
        obs3 = self._agent3_obs(hazard_state, obs1, obs2, t_norm)

        assert obs1.shape == (self.OBS_DIM,), f"Agent 1 obs shape mismatch: {obs1.shape}"
        assert obs2.shape == (self.OBS_DIM,), f"Agent 2 obs shape mismatch: {obs2.shape}"
        assert obs3.shape == (self.OBS_DIM,), f"Agent 3 obs shape mismatch: {obs3.shape}"

        return AgentObservations(agent_1=obs1, agent_2=obs2, agent_3=obs3)

    def observation_space_shape(self) -> Tuple[int]:
        """Return the shape of a single agent's observation vector."""
        return (self.OBS_DIM,)

    # -----------------------------------------------------------------------
    # Per-agent observation builders
    # -----------------------------------------------------------------------

    def _agent1_obs(
        self,
        hs: HazardState,
        flat_state: np.ndarray,
        t_norm: float,
        resource_availability: float,
    ) -> np.ndarray:
        """
        Agent 1 — Storm Detection Agent.
        Observes atmospheric features encoded in φ_t and storm probability p^storm_t.
        """
        storm_field = hs.storm_field

        # Storm field stats summarise the ViT/φ_t atmospheric encoding
        # in synthetic mode (dims 0..3)
        phi_summary = np.array([
            float(storm_field.mean()),
            float(storm_field.max()),
            float(storm_field.std()),
            float((storm_field > 0.5).mean()),
        ], dtype=np.float32)

        # Quadrant max values capture spatial structure
        quad = self._quadrant_stats(storm_field, stat="max")  # (4,)

        obs = np.array([
            *phi_summary,                        # [0..3]  φ_t summary
            hs.storm_probability,                # [4]     p^storm_t
            hs.rainfall_intensity,               # [5]     i^rain_t
            *quad,                               # [6..9]  quadrant maxima
            t_norm,                              # [10]    t / T
            resource_availability,               # [11]    resource state
        ], dtype=np.float32)

        return obs

    def _agent2_obs(
        self,
        hs: HazardState,
        flat_state: np.ndarray,
        t_norm: float,
    ) -> np.ndarray:
        """
        Agent 2 — Flood Risk Assessment Agent.
        Observes rainfall intensity i^rain_t and river-level indicator l^river_t.
        """
        flood_field = hs.flood_field
        quad_mean = self._quadrant_stats(flood_field, stat="mean")   # (4,)
        quad_max = self._quadrant_stats(flood_field, stat="max")     # (4,)

        combined_risk = float(hs.storm_probability * hs.rainfall_intensity)

        obs = np.array([
            hs.rainfall_intensity,               # [0]    i^rain_t
            hs.river_level,                      # [1]    l^river_t
            *quad_mean,                          # [2..5] flood quadrant means
            *quad_max,                           # [6..9] flood quadrant maxima
            combined_risk,                       # [10]   compound risk proxy
            t_norm,                              # [11]   t / T
        ], dtype=np.float32)

        return obs

    def _agent3_obs(
        self,
        hs: HazardState,
        obs1: np.ndarray,
        obs2: np.ndarray,
        t_norm: float,
    ) -> np.ndarray:
        """
        Agent 3 — Evacuation Planning Agent.
        Receives outputs forwarded by Agents 1 and 2 plus regional vulnerability.
        Synthesises upstream risk assessments into evacuation decisions.
        """
        # Forward the first 4 dims of each upstream agent's observation
        agent1_summary = obs1[:4]   # φ_t / storm summary from Agent 1
        agent2_summary = obs2[:4]   # flood summary from Agent 2

        # Combined max risk across both hazard fields
        combined_field = hs.storm_field * hs.flood_field
        max_combined = float(combined_field.max())

        # Normalised event count
        n_events_norm = min(float(len(hs.active_events)) / 5.0, 1.0)

        obs = np.array([
            *agent1_summary,                     # [0..3]  forwarded from Agent 1
            *agent2_summary,                     # [4..7]  forwarded from Agent 2
            hs.vulnerability,                    # [8]     v^region_t
            max_combined,                        # [9]     compound hazard peak
            n_events_norm,                       # [10]    active event count
            t_norm,                              # [11]    t / T
        ], dtype=np.float32)

        return obs

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    @staticmethod
    def _quadrant_stats(field: np.ndarray, stat: str = "mean") -> np.ndarray:
        """
        Divide the grid into four quadrants and compute a statistic per quadrant.

        Returns a (4,) array: [top-left, top-right, bottom-left, bottom-right].
        """
        h, w = field.shape
        h2, w2 = h // 2, w // 2
        quadrants = [
            field[:h2, :w2],   # top-left
            field[:h2, w2:],   # top-right
            field[h2:, :w2],   # bottom-left
            field[h2:, w2:],   # bottom-right
        ]
        fn = np.mean if stat == "mean" else np.max
        return np.array([float(fn(q)) for q in quadrants], dtype=np.float32)
