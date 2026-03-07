"""
PPO Agent for Disaster Response
=================================
Purpose:
    Wrapper around Stable-Baselines3's PPO algorithm configured for the
    DisasterResponseEnv. Provides a clean interface for training, saving,
    loading, and inference.

Input:
    Gymnasium environment (DisasterResponseEnv or any compatible env)

Output:
    Trained PPO policy capable of selecting optimal emergency response actions

Example usage:
    from rl_agent.environment import DisasterResponseEnv
    from rl_agent.agent_ppo import DisasterResponseAgent

    env = DisasterResponseEnv(max_steps=50, seed=42)
    agent = DisasterResponseAgent(env, seed=42)
    agent.train(total_timesteps=100_000)
    agent.save("results/ppo_agent")
    action = agent.predict(observation)
"""

import os
import numpy as np
from typing import Optional, Tuple

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym


# ---------------------------------------------------------------------------
# Custom callback for logging episode rewards
# ---------------------------------------------------------------------------

class RewardLoggingCallback(BaseCallback):
    """
    Callback that accumulates per-episode rewards for later plotting.

    Attributes:
        episode_rewards: List of cumulative reward values at episode end.
        episode_lengths: List of episode lengths.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.episode_rewards: list = []
        self.episode_lengths: list = []

    def _on_step(self) -> bool:
        """Called at every environment step."""
        # SB3 stores episode info in self.locals['infos']
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
        return True


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class DisasterResponseAgent:
    """
    PPO-based emergency response agent for the DisasterResponseEnv.

    Wraps Stable-Baselines3 PPO with sensible defaults for this domain,
    and adds convenience methods for training, evaluation, and export.

    Args:
        env:             Gymnasium environment instance (or VecEnv).
        seed:            Random seed for reproducibility.
        learning_rate:   PPO learning rate.
        n_steps:         Number of steps per PPO update rollout.
        batch_size:      Mini-batch size for gradient updates.
        n_epochs:        Number of optimization epochs per update.
        gamma:           Discount factor for future rewards.
        gae_lambda:      GAE lambda for advantage estimation.
        clip_range:      PPO clipping parameter.
        verbose:         SB3 verbosity level (0=silent, 1=info, 2=debug).
        tensorboard_log: Directory for TensorBoard logs. None to disable.
    """

    DEFAULT_POLICY = "MlpPolicy"

    def __init__(
        self,
        env: gym.Env,
        seed: int = 42,
        learning_rate: float = 3e-4,
        n_steps: int = 256,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        verbose: int = 1,
        tensorboard_log: Optional[str] = None,
    ) -> None:
        self.seed = seed
        self.reward_callback = RewardLoggingCallback()

        # Wrap env in Monitor for SB3 episode logging
        if not isinstance(env, VecEnv):
            env = Monitor(env)

        self.model = PPO(
            policy=self.DEFAULT_POLICY,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            seed=seed,
            verbose=verbose,
            tensorboard_log=tensorboard_log,
            policy_kwargs={
                # 3-layer MLP head for the 4-dimensional state space
                "net_arch": [dict(pi=[128, 64], vf=[128, 64])]
            },
        )

    def train(
        self,
        total_timesteps: int = 100_000,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 5_000,
        save_best_path: Optional[str] = None,
    ) -> "DisasterResponseAgent":
        """
        Train the PPO agent.

        Args:
            total_timesteps: Total environment steps for training.
            eval_env:        Optional evaluation environment for EvalCallback.
            eval_freq:       Evaluation frequency in timesteps.
            save_best_path:  Path to save the best model found during evaluation.

        Returns:
            self (for method chaining)
        """
        callbacks = [self.reward_callback]

        if eval_env is not None:
            if save_best_path:
                os.makedirs(save_best_path, exist_ok=True)
            eval_callback = EvalCallback(
                eval_env=Monitor(eval_env),
                best_model_save_path=save_best_path,
                log_path=save_best_path,
                eval_freq=eval_freq,
                n_eval_episodes=10,
                deterministic=True,
                verbose=0,
            )
            callbacks.append(eval_callback)

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
        return self

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[int, Optional[np.ndarray]]:
        """
        Predict the best action for a given observation.

        Args:
            observation:   State array of shape (4,) in [0, 1].
            deterministic: Use greedy (argmax) policy if True.

        Returns:
            (action, state) tuple. State is None for MlpPolicy.
        """
        action, state = self.model.predict(observation, deterministic=deterministic)
        return int(action), state

    def evaluate(
        self,
        env: gym.Env,
        n_episodes: int = 20,
        deterministic: bool = True,
    ) -> dict:
        """
        Evaluate the trained policy over multiple episodes.

        Args:
            env:          Environment to evaluate on.
            n_episodes:   Number of evaluation episodes.
            deterministic: Use deterministic policy.

        Returns:
            Dictionary with mean_reward, std_reward, and mean_episode_length.
        """
        episode_rewards = []
        episode_lengths = []

        for _ in range(n_episodes):
            obs, _ = env.reset()
            ep_reward = 0.0
            ep_len = 0
            done = False
            while not done:
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                ep_len += 1
                done = terminated or truncated
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_len)

        return {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "mean_episode_length": float(np.mean(episode_lengths)),
            "n_episodes": n_episodes,
        }

    def save(self, path: str) -> None:
        """
        Save the model weights and policy to disk.

        Args:
            path: File path without extension (SB3 adds .zip automatically).
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self.model.save(path)
        print(f"[INFO] Agent saved to {path}.zip")

    @classmethod
    def load(
        cls,
        path: str,
        env: gym.Env,
    ) -> "DisasterResponseAgent":
        """
        Load a previously saved agent.

        Args:
            path: Path to saved model (with or without .zip extension).
            env:  Environment instance to associate with the loaded model.

        Returns:
            DisasterResponseAgent with loaded weights.
        """
        instance = cls.__new__(cls)
        instance.seed = 0
        instance.reward_callback = RewardLoggingCallback()
        instance.model = PPO.load(path, env=Monitor(env))
        print(f"[INFO] Agent loaded from {path}")
        return instance

    def get_reward_history(self) -> list:
        """Return list of episode rewards recorded during training."""
        return self.reward_callback.episode_rewards

    def get_episode_length_history(self) -> list:
        """Return list of episode lengths recorded during training."""
        return self.reward_callback.episode_lengths


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from rl_agent.environment import DisasterResponseEnv

    print("=== PPO Agent Smoke Test ===")
    env = DisasterResponseEnv(max_steps=50, seed=42)
    agent = DisasterResponseAgent(env, seed=42, verbose=0)
    agent.train(total_timesteps=2_000)

    print(f"Reward history (last 5): {agent.get_reward_history()[-5:]}")

    eval_env = DisasterResponseEnv(max_steps=50, seed=99)
    stats = agent.evaluate(eval_env, n_episodes=5)
    print(f"Evaluation: {stats}")
    print("Smoke test passed.")
