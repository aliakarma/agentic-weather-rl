"""
src/environment/disaster_env.py
================================
DisasterResponseBenchmark — the primary simulation environment.

Implements a Gymnasium environment for the three-agent cooperative
disaster response task described in Section 4.1 of the paper.

Key properties (Table 1 of paper):
  - 20×20 grid world
  - 3 cooperative agents (Storm Detection, Flood Risk, Evacuation Planning)
  - Episode length T = 100 steps
  - Joint action space: 4^3 = 64 discrete actions
  - Global state dimension: 24 (synthetic mode)
  - Local observation dimension: 12 per agent
  - Hazard onset: Poisson(λ = 0.05)

Agents act simultaneously each step. The environment returns a shared
team reward and per-agent constraint costs following Eq. 4–5 of the paper.

Usage
-----
    from src.environment.disaster_env import DisasterEnv

    env = DisasterEnv()
    obs, info = env.reset(seed=42)
    for _ in range(100):
        actions = {1: 0, 2: 1, 3: 0}   # dict keyed by agent id
        obs, reward, terminated, truncated, info = env.step(actions)
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, Optional, Tuple

from src.environment.hazard_generator import HazardGenerator, HazardState
from src.environment.obs_router import ObservationRouter, AgentObservations
from src.environment.reward import RewardCalculator, RewardInfo


# ---------------------------------------------------------------------------
# Constants — must match environment.yaml exactly
# ---------------------------------------------------------------------------

GRID_SIZE: int = 20
N_AGENTS: int = 3
EPISODE_LENGTH: int = 100
STATE_DIM: int = 24        # global state dimension (synthetic mode)
OBS_DIM: int = 12          # local observation dimension per agent
N_ACTIONS: int = 4         # {0: no-op, 1: warn, 2: deploy, 3: evacuate}
HAZARD_LAMBDA: float = 0.05


# ---------------------------------------------------------------------------
# DisasterEnv
# ---------------------------------------------------------------------------

class DisasterEnv(gym.Env):
    """
    Multi-agent disaster response environment (DisasterResponseBenchmark).

    Follows the Gymnasium API with multi-agent extensions:
      - reset() → Dict[int, np.ndarray], info
      - step(actions: Dict[int, int]) → obs, reward, terminated, truncated, info

    Observation and action spaces are defined per agent. All agents share
    the same action space; observations are heterogeneous (see obs_router.py).

    Parameters
    ----------
    grid_size : int
        Side length of the square grid (default 20).
    n_agents : int
        Number of cooperative agents (default 3; must be 3).
    episode_length : int
        Maximum steps per episode T (default 100).
    hazard_lambda : float
        Poisson arrival rate for hazard events (default 0.05).
    reward_alpha : float
        Mitigation reward weight α (default 1.0).
    reward_beta : float
        False-alarm penalty weight β (default 0.5).
    reward_eta : float
        Delay penalty weight η (default 0.3).
    constraint_threshold : float
        Per-agent safety constraint threshold d_i (default 0.10).
    seed : int | None
        Master random seed (can also be set via reset(seed=...)).
    render_mode : str | None
        Currently only None is supported.
    """

    metadata = {"render_modes": [], "render_fps": 4}

    def __init__(
        self,
        grid_size: int = GRID_SIZE,
        n_agents: int = N_AGENTS,
        episode_length: int = EPISODE_LENGTH,
        hazard_lambda: float = HAZARD_LAMBDA,
        reward_alpha: float = 1.0,
        reward_beta: float = 0.5,
        reward_eta: float = 0.3,
        constraint_threshold: float = 0.10,
        seed: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()

        if n_agents != N_AGENTS:
            raise ValueError(
                f"DisasterEnv supports exactly {N_AGENTS} agents; got {n_agents}."
            )

        self.grid_size = grid_size
        self.n_agents = n_agents
        self.episode_length = episode_length
        self.constraint_threshold = constraint_threshold
        self.render_mode = render_mode

        # --- Sub-components ---
        self._hazard_gen = HazardGenerator(
            grid_size=grid_size,
            hazard_lambda=hazard_lambda,
            seed=seed,
        )
        self._obs_router = ObservationRouter(
            grid_size=grid_size,
            episode_length=episode_length,
            obs_dim=OBS_DIM,
        )
        self._reward_calc = RewardCalculator(
            alpha=reward_alpha,
            beta=reward_beta,
            eta=reward_eta,
            constraint_threshold=constraint_threshold,
        )

        # --- Gymnasium spaces ---
        # Action space: each agent selects from {0, 1, 2, 3}
        self.action_space = spaces.Discrete(N_ACTIONS)

        # Per-agent observation space: flat vector of dimension OBS_DIM
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )

        # Global state space (for centralised critic during training)
        self.state_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(STATE_DIM,),
            dtype=np.float32,
        )

        # --- Internal state ---
        self._timestep: int = 0
        self._hazard_state: HazardState | None = None
        self._prev_hazard_state: HazardState | None = None
        self._agent_obs: AgentObservations | None = None
        self._global_state: np.ndarray | None = None
        self._resource_availability: float = 1.0

        # Episode-level accumulators for constraint violation tracking
        self._episode_constraint_costs: Dict[int, float] = {
            i: 0.0 for i in range(1, N_AGENTS + 1)
        }
        self._episode_violation_steps: int = 0
        self._episode_total_steps: int = 0

    # -----------------------------------------------------------------------
    # Gymnasium API — reset
    # -----------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> Tuple[Dict[int, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment to the start of a new episode.

        Returns
        -------
        observations : dict[int, np.ndarray]
            Initial local observations keyed by agent id {1, 2, 3}.
        info : dict
            Auxiliary info including the initial global state.
        """
        super().reset(seed=seed)

        self._timestep = 0
        self._prev_hazard_state = None
        self._resource_availability = 1.0

        # Reset sub-components
        self._hazard_state = self._hazard_gen.reset(seed=seed)
        self._reward_calc.reset()

        # Reset episode accumulators
        self._episode_constraint_costs = {i: 0.0 for i in range(1, N_AGENTS + 1)}
        self._episode_violation_steps = 0
        self._episode_total_steps = 0

        # Build initial observations and global state
        self._global_state = self._hazard_gen.get_flat_state()
        self._agent_obs = self._obs_router.route(
            hazard_state=self._hazard_state,
            flat_state=self._global_state,
            timestep=self._timestep,
            resource_availability=self._resource_availability,
        )

        observations = self._agent_obs.as_dict()
        info = self._build_info(reward_info=None)

        return observations, info

    # -----------------------------------------------------------------------
    # Gymnasium API — step
    # -----------------------------------------------------------------------

    def step(
        self,
        actions: Dict[int, int],
    ) -> Tuple[Dict[int, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Advance the environment by one step.

        Parameters
        ----------
        actions : dict[int, int]
            Joint action mapping agent_id → action ∈ {0, 1, 2, 3}.
            Must contain keys 1, 2, and 3.

        Returns
        -------
        observations : dict[int, np.ndarray]
            Updated local observations for each agent.
        reward : float
            Shared team reward r(s_t, a_t).
        terminated : bool
            True if a terminal condition is reached (not used here).
        truncated : bool
            True when the episode length T is exceeded.
        info : dict
            Auxiliary information: global state, constraint costs,
            risk level, reward breakdown.
        """
        if self._hazard_state is None:
            raise RuntimeError("Call reset() before step().")

        self._validate_actions(actions)

        # Unpack joint action as a tuple (agent_1_action, ..., agent_3_action)
        joint_action = (actions[1], actions[2], actions[3])

        # Advance hazard dynamics
        self._prev_hazard_state = self._hazard_state
        self._hazard_state = self._hazard_gen.step()

        # Compute reward and constraint costs
        reward_info: RewardInfo = self._reward_calc.compute(
            hazard_state=self._hazard_state,
            actions=joint_action,
            prev_hazard_state=self._prev_hazard_state,
            timestep=self._timestep,
        )

        # Update resource availability (simple depletion model)
        self._resource_availability = self._update_resources(
            self._resource_availability, joint_action
        )

        # Build new observations and global state
        self._global_state = self._hazard_gen.get_flat_state()
        self._agent_obs = self._obs_router.route(
            hazard_state=self._hazard_state,
            flat_state=self._global_state,
            timestep=self._timestep + 1,
            resource_availability=self._resource_availability,
        )

        # Accumulate episode-level constraint statistics
        self._episode_total_steps += 1
        for agent_id, cost in reward_info.constraint_costs.items():
            self._episode_constraint_costs[agent_id] += cost

        violated = self._reward_calc.compute_violation(
            reward_info.constraint_costs, self.constraint_threshold
        )
        if violated:
            self._episode_violation_steps += 1

        self._timestep += 1

        # Termination and truncation
        terminated = False
        truncated = self._timestep >= self.episode_length

        observations = self._agent_obs.as_dict()
        info = self._build_info(reward_info=reward_info)

        return observations, reward_info.total_reward, terminated, truncated, info

    # -----------------------------------------------------------------------
    # Additional API — used by centralised critic during training
    # -----------------------------------------------------------------------

    def get_global_state(self) -> np.ndarray:
        """
        Return the current 24-dimensional global state vector s_t.
        Used by the centralised critic V_ψ(s_t) during CTDE training.
        """
        if self._global_state is None:
            raise RuntimeError("Call reset() before get_global_state().")
        return self._global_state.copy()

    def get_episode_stats(self) -> Dict[str, Any]:
        """Return cumulative statistics for the current or last episode."""
        total = max(self._episode_total_steps, 1)
        return {
            "violation_rate": self._episode_violation_steps / total,
            "constraint_costs": dict(self._episode_constraint_costs),
            "total_steps": self._episode_total_steps,
        }

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _update_resources(
        current: float,
        actions: Tuple[int, int, int],
    ) -> float:
        """
        Simple resource depletion model.
        Deploy / evacuate actions consume a fraction of available resources.
        """
        depletion = sum(
            0.02 if a == 2 else (0.04 if a == 3 else 0.0)
            for a in actions
        )
        return float(np.clip(current - depletion, 0.0, 1.0))

    def _validate_actions(self, actions: Dict[int, int]) -> None:
        """Validate that actions dict contains the correct agent ids and values."""
        for agent_id in range(1, self.n_agents + 1):
            if agent_id not in actions:
                raise ValueError(
                    f"Missing action for agent {agent_id}. "
                    f"Expected keys: {{1, 2, 3}}, got: {set(actions.keys())}"
                )
            a = actions[agent_id]
            if not (0 <= a < N_ACTIONS):
                raise ValueError(
                    f"Action {a} for agent {agent_id} is out of range [0, {N_ACTIONS - 1}]."
                )

    def _build_info(
        self,
        reward_info: RewardInfo | None,
    ) -> Dict[str, Any]:
        """Assemble the info dict returned by reset() and step()."""
        info: Dict[str, Any] = {
            "timestep": self._timestep,
            "global_state": self._global_state,
            "resource_availability": self._resource_availability,
            "episode_violation_rate": (
                self._episode_violation_steps / max(self._episode_total_steps, 1)
            ),
        }
        if reward_info is not None:
            info.update({
                "reward_mitigation": reward_info.mitigation,
                "reward_false_alarm": reward_info.false_alarm,
                "reward_delay": reward_info.delay,
                "constraint_costs": reward_info.constraint_costs,
                "risk_level": reward_info.risk_level,
            })
        if self._hazard_state is not None:
            info["storm_probability"] = self._hazard_state.storm_probability
            info["rainfall_intensity"] = self._hazard_state.rainfall_intensity
            info["river_level"] = self._hazard_state.river_level
        return info

    # -----------------------------------------------------------------------
    # Gymnasium render / close (stub)
    # -----------------------------------------------------------------------

    def render(self) -> None:
        """Rendering is not implemented in the current version."""
        pass

    def close(self) -> None:
        pass
