"""
Disaster Response Reinforcement Learning Environment
======================================================
Purpose:
    A custom Gymnasium environment that simulates a disaster management
    decision problem. The RL agent observes a 4-dimensional weather state
    and selects from four emergency response actions.

    This environment is designed to be used with Stable-Baselines3 PPO
    and is compatible with the standard gym interface.

State Space (Box, 4-dimensional, all in [0, 1]):
    [storm_probability, rainfall_intensity, flood_risk_score, regional_risk_indicator]

Action Space (Discrete, 4 actions):
    0 = No action
    1 = Issue early warning
    2 = Prepare emergency resources
    3 = Recommend evacuation

Reward function:
    +20  correct disaster mitigation (right action for high-risk scenario)
    +10  early warning issued appropriately
    -10  false alarm (warning or action taken for low-risk scenario)
    -25  disaster damage (missed high-risk event, no action taken)

Example usage:
    from rl_agent.environment import DisasterResponseEnv
    env = DisasterResponseEnv()
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action=1)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Action index → label mapping
ACTION_LABELS = {
    0: "No Action",
    1: "Issue Early Warning",
    2: "Prepare Emergency Resources",
    3: "Recommend Evacuation",
}

# Reward values
REWARD_CORRECT_MITIGATION: float = 20.0
REWARD_EARLY_WARNING: float = 10.0
PENALTY_FALSE_ALARM: float = -10.0
PENALTY_DISASTER_DAMAGE: float = -25.0
REWARD_CORRECT_NO_ACTION: float = 2.0   # Small reward for correctly doing nothing

# Risk thresholds (based on storm_probability + flood_risk weighted average)
THRESHOLD_LOW_RISK: float = 0.35
THRESHOLD_HIGH_RISK: float = 0.65
THRESHOLD_CRITICAL_RISK: float = 0.80


# ---------------------------------------------------------------------------
# Scenario generator
# ---------------------------------------------------------------------------

class WeatherScenarioGenerator:
    """
    Generates synthetic weather state observations for RL training.

    Scenarios are parameterized by risk level (low, medium, high, critical)
    and sample realistic distributions for each state dimension.

    Args:
        seed: Random seed for reproducibility.
    """

    RISK_LEVELS = ["low", "medium", "high", "critical"]

    def __init__(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed=seed)

    def sample(self, risk_level: Optional[str] = None) -> np.ndarray:
        """
        Sample a weather state vector for a given risk level.

        Args:
            risk_level: One of 'low', 'medium', 'high', 'critical'.
                        If None, randomly selected with equal probability.

        Returns:
            float32 array of shape (4,):
            [storm_prob, rainfall_intensity, flood_risk, regional_risk]
        """
        if risk_level is None:
            risk_level = self.rng.choice(self.RISK_LEVELS)

        if risk_level == "low":
            storm_prob = self.rng.uniform(0.0, 0.25)
            rainfall = self.rng.uniform(0.0, 0.25)
            flood_risk = self.rng.uniform(0.0, 0.20)
            regional = self.rng.uniform(0.0, 0.30)

        elif risk_level == "medium":
            storm_prob = self.rng.uniform(0.25, 0.55)
            rainfall = self.rng.uniform(0.20, 0.55)
            flood_risk = self.rng.uniform(0.20, 0.50)
            regional = self.rng.uniform(0.20, 0.55)

        elif risk_level == "high":
            storm_prob = self.rng.uniform(0.55, 0.80)
            rainfall = self.rng.uniform(0.50, 0.80)
            flood_risk = self.rng.uniform(0.50, 0.75)
            regional = self.rng.uniform(0.40, 0.70)

        else:  # critical
            storm_prob = self.rng.uniform(0.80, 1.00)
            rainfall = self.rng.uniform(0.75, 1.00)
            flood_risk = self.rng.uniform(0.70, 1.00)
            regional = self.rng.uniform(0.60, 1.00)

        state = np.array([storm_prob, rainfall, flood_risk, regional], dtype=np.float32)
        return state

    def compute_risk_level(self, state: np.ndarray) -> str:
        """
        Determine the qualitative risk level of a state vector.

        Args:
            state: float32 array of shape (4,).

        Returns:
            Risk level string: 'low', 'medium', 'high', or 'critical'.
        """
        composite = 0.4 * state[0] + 0.3 * state[1] + 0.2 * state[2] + 0.1 * state[3]
        if composite < THRESHOLD_LOW_RISK:
            return "low"
        elif composite < THRESHOLD_HIGH_RISK:
            return "medium"
        elif composite < THRESHOLD_CRITICAL_RISK:
            return "high"
        else:
            return "critical"


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def compute_reward(state: np.ndarray, action: int) -> Tuple[float, Dict[str, Any]]:
    """
    Compute the reward for taking an action in a given weather state.

    The reward function is designed to:
      - Encourage early warnings for medium-risk situations
      - Enforce full emergency response for high/critical situations
      - Penalize false alarms (over-responding to low-risk)
      - Penalize inaction during high-risk situations (disaster damage)

    Args:
        state:  Weather state vector [storm_prob, rainfall, flood_risk, regional].
        action: Integer action index (0–3).

    Returns:
        (reward, info_dict) where info_dict contains diagnostic details.
    """
    composite = 0.4 * state[0] + 0.3 * state[1] + 0.2 * state[2] + 0.1 * state[3]
    risk_str = (
        "low" if composite < THRESHOLD_LOW_RISK else
        "medium" if composite < THRESHOLD_HIGH_RISK else
        "high" if composite < THRESHOLD_CRITICAL_RISK else
        "critical"
    )

    reward = 0.0
    outcome = "neutral"

    if risk_str == "low":
        if action == 0:
            reward = REWARD_CORRECT_NO_ACTION
            outcome = "correct_no_action"
        else:
            reward = PENALTY_FALSE_ALARM
            outcome = "false_alarm"

    elif risk_str == "medium":
        if action == 1:
            reward = REWARD_EARLY_WARNING
            outcome = "early_warning"
        elif action == 0:
            reward = PENALTY_FALSE_ALARM * 0.5  # Mild penalty for inaction
            outcome = "mild_underresponse"
        else:
            reward = REWARD_EARLY_WARNING * 0.5  # Acceptable but over-prepared
            outcome = "over_prepared"

    elif risk_str == "high":
        if action == 2:
            reward = REWARD_CORRECT_MITIGATION
            outcome = "correct_mitigation"
        elif action == 1:
            reward = REWARD_EARLY_WARNING  # Warning is better than nothing
            outcome = "underresponse"
        elif action == 0:
            reward = PENALTY_DISASTER_DAMAGE
            outcome = "disaster_damage"
        elif action == 3:
            reward = REWARD_CORRECT_MITIGATION * 0.8  # Evacuation is reasonable
            outcome = "acceptable_overresponse"

    else:  # critical
        if action == 3:
            reward = REWARD_CORRECT_MITIGATION
            outcome = "correct_mitigation"
        elif action == 2:
            reward = REWARD_EARLY_WARNING
            outcome = "underresponse_critical"
        elif action == 1:
            reward = PENALTY_DISASTER_DAMAGE * 0.5
            outcome = "underresponse"
        elif action == 0:
            reward = PENALTY_DISASTER_DAMAGE
            outcome = "disaster_damage"

    info = {
        "composite_risk": float(composite),
        "risk_level": risk_str,
        "action_taken": ACTION_LABELS[action],
        "outcome": outcome,
    }
    return reward, info


# ---------------------------------------------------------------------------
# Gymnasium Environment
# ---------------------------------------------------------------------------

class DisasterResponseEnv(gym.Env):
    """
    Custom Gymnasium environment for disaster management decision making.

    The environment simulates a sequence of weather scenarios. At each step:
      1. A weather state is sampled from the scenario generator.
      2. The agent selects an emergency response action.
      3. A reward is returned based on the appropriateness of the action.
      4. The episode terminates after max_steps steps.

    Args:
        max_steps:    Number of time steps per episode.
        seed:         Random seed.
        render_mode:  'human' to print scenario summaries, None otherwise.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        max_steps: int = 50,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.max_steps = max_steps
        self.render_mode = render_mode
        self._step_count: int = 0
        self._current_state: Optional[np.ndarray] = None
        self._episode_rewards: list = []
        self._last_info: Dict[str, Any] = {}

        # --- Spaces ---
        # Observation: 4 continuous values in [0, 1]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )
        # Action: 4 discrete options
        self.action_space = spaces.Discrete(4)

        # Scenario generator
        self.scenario_gen = WeatherScenarioGenerator(seed=seed)

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment for a new episode.

        Args:
            seed:    Optional seed override for this episode.
            options: Optional config dict (unused).

        Returns:
            (initial_observation, info_dict)
        """
        super().reset(seed=seed)
        if seed is not None:
            self.scenario_gen = WeatherScenarioGenerator(seed=seed)

        self._step_count = 0
        self._episode_rewards = []
        self._current_state = self.scenario_gen.sample()
        self._last_info = {}

        return self._current_state.copy(), {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one environment step.

        Args:
            action: Integer in [0, 3].

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if self._current_state is None:
            raise RuntimeError("Call reset() before step().")

        reward, info = compute_reward(self._current_state, action)
        self._episode_rewards.append(reward)
        self._step_count += 1

        # Sample next state
        self._current_state = self.scenario_gen.sample()
        info["step"] = self._step_count
        info["cumulative_reward"] = float(sum(self._episode_rewards))
        self._last_info = info

        terminated = False  # Episode ends only via truncation
        truncated = self._step_count >= self.max_steps

        if self.render_mode == "human":
            self.render()

        return self._current_state.copy(), reward, terminated, truncated, info

    def render(self) -> None:
        """Print a human-readable summary of the last step."""
        if self._last_info:
            print(
                f"  Step {self._last_info.get('step', '?'):>3} | "
                f"Risk: {self._last_info.get('risk_level', '?'):>8} | "
                f"Action: {self._last_info.get('action_taken', '?'):>30} | "
                f"Outcome: {self._last_info.get('outcome', '?'):>25} | "
                f"Cumulative reward: {self._last_info.get('cumulative_reward', 0.0):>8.1f}"
            )

    def close(self) -> None:
        """Clean up environment resources."""
        pass

    def get_episode_stats(self) -> Dict[str, float]:
        """
        Return summary statistics for the current episode.

        Returns:
            Dictionary with total_reward, mean_reward, and step count.
        """
        return {
            "total_reward": float(sum(self._episode_rewards)),
            "mean_reward": float(np.mean(self._episode_rewards)) if self._episode_rewards else 0.0,
            "steps": self._step_count,
        }


# ---------------------------------------------------------------------------
# Register with gymnasium
# ---------------------------------------------------------------------------
try:
    gym.register(
        id="DisasterResponse-v0",
        entry_point="rl_agent.environment:DisasterResponseEnv",
        max_episode_steps=50,
    )
except Exception:
    pass  # Already registered


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Disaster Response Environment Smoke Test ===")
    env = DisasterResponseEnv(max_steps=10, seed=42, render_mode="human")
    obs, info = env.reset()
    print(f"Initial observation: {obs}")

    total_reward = 0.0
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(f"\nEpisode stats: {env.get_episode_stats()}")
    print("Smoke test passed.")
