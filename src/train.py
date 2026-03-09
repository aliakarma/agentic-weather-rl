"""
Standalone training entry-point.
The notebook calls agent.train() directly; this module exists for
  `from src.train import train`
and for CLI usage:  python -m src.train --episodes 100
"""
from typing import List, Optional
from src.algorithms.lagrangian_ctde import LagrangianCTDE, LagrangianCTDEConfig, EpisodeStats
from src.environment.disaster_env import DisasterEnv


def train(
    env: Optional[DisasterEnv] = None,
    config: Optional[LagrangianCTDEConfig] = None,
    n_episodes: int = 100,
    seed: int = 42,
    device: str = "cpu",
) -> List[EpisodeStats]:
    """
    Convenience wrapper — creates agent and calls agent.train().
    Returns training history.
    """
    if env is None:
        env = DisasterEnv(seed=seed)
    if config is None:
        config = LagrangianCTDEConfig(n_episodes=n_episodes, seed=seed, device=device)
    agent = LagrangianCTDE(env=env, config=config, device=device)
    return agent.train(n_episodes=n_episodes)


if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed",     type=int, default=42)
    args = p.parse_args()
    history = train(n_episodes=args.episodes, seed=args.seed)
    print(json.dumps({
        "final_reward":    history[-1].total_reward,
        "final_violation": history[-1].violation_rate,
    }))
