"""Final gradient-flow debug for NumPy PPO actor update.

This script mirrors torch-style checks with NumPy equivalents:
- logp_new gradient-path check (manual backprop path)
- per-parameter gradient magnitudes
- parameter linkage (named parameter shapes)
- parameter delta after step
- old vs new log-prob difference
- forced large update debug pass
"""
from __future__ import annotations

import numpy as np

from src.algorithms.lagrangian_ctde import LagrangianCTDE, LagrangianCTDEConfig
from src.environment.disaster_env import DisasterEnv


def collect_rollout(agent: LagrangianCTDE, env: DisasterEnv, seed: int):
    obs, _ = env.reset(seed=seed)
    agent._rng = np.random.default_rng(seed)
    obs_list, act_list, rew_list, cost_list = [], [], [], []
    ep_reward = 0.0

    for _ in range(env.episode_length):
        actions = {i: int(agent._actors[i].get_action(obs[i], deterministic=False)) for i in range(1, 4)}
        next_obs, reward, terminated, truncated, info = env.step(actions)

        obs_list.append(obs)
        act_list.append(actions)
        rew_list.append(float(reward))
        cost_list.append([float(info.get("violation", 0.0))] * 3)
        ep_reward += float(reward)

        obs = next_obs
        if terminated or truncated:
            break

    return ep_reward, obs_list, act_list, rew_list, cost_list


def compute_returns(rew_list, cost_list, gamma: float):
    T = len(rew_list)
    returns_r = np.zeros(T, dtype=np.float64)
    returns_c = np.zeros((T, 3), dtype=np.float64)
    g_r = 0.0
    g_c = np.zeros(3, dtype=np.float64)

    for t in reversed(range(T)):
        g_r = rew_list[t] + gamma * g_r
        g_c = np.asarray(cost_list[t], dtype=np.float64) + gamma * g_c
        returns_r[t] = g_r
        returns_c[t] = g_c

    return returns_r, returns_c


def eval_reward(agent: LagrangianCTDE, seed: int) -> float:
    env = DisasterEnv(seed=agent.config.seed, observation_noise=0.02, hazard_rate=0.22)
    obs, _ = env.reset(seed=seed)
    agent._rng = np.random.default_rng(seed)
    rsum = 0.0
    for _ in range(env.episode_length):
        actions = {i: int(agent._actors[i].get_action(obs[i], deterministic=True)) for i in range(1, 4)}
        obs, reward, terminated, truncated, _ = env.step(actions)
        rsum += float(reward)
        if terminated or truncated:
            break
    return rsum


def main() -> None:
    cfg = LagrangianCTDEConfig(
        seed=7,
        n_episodes=1,
        bc_pretrain_enabled=False,
        warmup_phase_enabled=False,
        exploration_noise_enabled=False,
        freeze_actor_steps=0,
        actor_lr_scale=0.01,
        adv_clip_abs=2.0,
        ppo_update_epochs=1,
        kl_early_stop_threshold=10.0,
        clip_eps=10.0,
    )

    env = DisasterEnv(seed=cfg.seed, observation_noise=0.02, hazard_rate=0.22)
    agent = LagrangianCTDE(env=env, config=cfg)

    # Baseline before update
    eval_seed = 424242
    r_before = eval_reward(agent, eval_seed)

    # Collect update batch
    _, obs_list, act_list, rew_list, cost_list = collect_rollout(agent, env, seed=7001)
    returns_r, returns_c = compute_returns(rew_list, cost_list, gamma=cfg.gamma)
    adv = agent._compute_global_advantages(returns_r, agent._compute_global_values(obs_list))

    # One forced large debug update (debug only)
    kls = []
    debug_rows = []
    T = min(len(rew_list), 20)
    for t in range(T):
        for idx, i in enumerate(range(1, 4)):
            old_lp = agent._actors[i].log_prob(obs_list[t][i], act_list[t][i])
            cav = float(adv[t]) - float(agent._lambdas[idx]) * float(returns_c[t, idx])
            out = agent._actors[i].update(
                obs=obs_list[t][i],
                action=act_list[t][i],
                advantage=cav,
                lr=cfg.lr * 10.0,  # forced large debug update
                entropy_coef=0.0,
                clip_eps=10.0,
                old_log_prob=old_lp,
                grad_clip_norm=0.5,
                kl_loss_scale_threshold=1e9,
                kl_loss_scale_factor=1.0,
                debug=True,
            )
            kls.append(abs(float(out["approx_kl"])))
            debug_rows.append(out)

    # Reward after update
    r_after = eval_reward(agent, eval_seed)

    delta_param = float(np.mean([row["delta_param"] for row in debug_rows])) if debug_rows else 0.0
    mean_kl = float(np.mean(kls)) if kls else 0.0
    mean_logp_shift = float(np.mean([row["logp_new"] - row["logp_old"] for row in debug_rows])) if debug_rows else 0.0

    print("SANITY RESULTS:")
    print(f"R_before={r_before:.6f}")
    print(f"R_after={r_after:.6f}")
    print(f"delta={r_after - r_before:.6f}")
    print(f"KL={mean_kl:.6f}")

    if debug_rows:
        sample = debug_rows[0]
        print(f"logp_new.requires_grad={sample['logp_new_requires_grad']}")
        print(f"manual_gradient_flow={sample['manual_gradient_flow']}")

        print("gradient_abs_means:")
        for name, g in sample["grad_abs_means"].items():
            print(f"  {name}: {g:.8f}")

        print("optimizer_param_shapes_equivalent:")
        for name, shp in sample["parameter_shapes"].items():
            print(f"  {name}: {tuple(shp)}")

    print(f"delta_param={delta_param:.10f}")
    print(f"mean(logp_new - logp_old)={mean_logp_shift:.8f}")

    all_true = (
        delta_param > 0.0
        and mean_kl > 0.0
        and bool(debug_rows and debug_rows[0]["manual_gradient_flow"])
    )

    if all_true:
        print("GRADIENT FLOW FIXED")
    else:
        print("GRADIENT FLOW STILL BROKEN")

    print("Final gradient debug completed.")


if __name__ == "__main__":
    main()
