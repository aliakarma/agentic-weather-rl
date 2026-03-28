import math

from src.algorithms.lagrangian_ctde import LagrangianCTDE, LagrangianCTDEConfig
from src.environment.disaster_env import DisasterEnv
from src.evaluate import evaluate


def _run_once(seed: int):
    env = DisasterEnv(seed=seed, observation_noise=0.02, hazard_rate=0.22)
    cfg = LagrangianCTDEConfig(n_episodes=6, seed=seed, log_interval=1000)
    agent = LagrangianCTDE(env=env, config=cfg)
    agent.train(n_episodes=6)

    eval_env = DisasterEnv(seed=10_000 + seed, observation_noise=0.02, hazard_rate=0.22)
    metrics = evaluate(agent, eval_env, n_episodes=5, seed_offset=10_000 + seed, deterministic=True)
    return metrics


def test_deterministic_seed_replay():
    m1 = _run_once(seed=7)
    m2 = _run_once(seed=7)

    assert m1["reward_mean"] == m2["reward_mean"]
    assert m1["violation_rate_mean"] == m2["violation_rate_mean"]
    assert m1["avg_damage_prevented_mean"] == m2["avg_damage_prevented_mean"]


def test_metric_sanity_ranges():
    m = _run_once(seed=11)

    for k, v in m.items():
        if isinstance(v, float):
            assert not math.isnan(v), f"NaN metric: {k}"

    assert 0.0 <= m["violation_rate_mean"] <= 1.0
    assert 0.0 <= m["fairness_jain_action_index"] <= 1.0
    assert m["observation_noise_mean"] >= 0.0
