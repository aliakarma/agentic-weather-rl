"""
Multi-agent disaster response environment.
3 agents: Storm (1), Flood (2), Evacuation (3)
4 discrete actions: 0=No-op, 1=Warn, 2=Deploy, 3=Evacuate
"""
import numpy as np


class DisasterEnv:
    N_AGENTS = 3
    N_ACTIONS = 4
    OBS_DIM = 12
    episode_length = 50

    def __init__(self, seed=None, observation_noise=0.02, hazard_rate=0.22):
        self._init_seed = seed
        self.observation_noise = float(observation_noise)
        self.hazard_rate = float(hazard_rate)
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._severity = 1
        self._prev_severity = 1
        self._hazard_pressure = 0.35
        self._high_risk_age = 0
        self._resource_budget = 4.5

        # Episode aggregates for trajectory-level evaluation.
        self._episode_reward = 0.0
        self._episode_damage = 0.0
        self._episode_damage_prevented = 0.0
        self._episode_violations = 0

    def reset(self, seed=None):
        s = seed if seed is not None else self._init_seed
        self._rng = np.random.default_rng(s)
        self._t = 0
        self._severity = int(self._rng.integers(0, 3))
        self._prev_severity = self._severity
        self._hazard_pressure = float(self._rng.uniform(0.2, 0.5))
        self._high_risk_age = 0

        self._episode_reward = 0.0
        self._episode_damage = 0.0
        self._episode_damage_prevented = 0.0
        self._episode_violations = 0

        return self._get_obs(), {}

    def _get_obs(self):
        obs = {}
        for i in range(1, 4):
            o = np.zeros(self.OBS_DIM, dtype=np.float32)
            o[0] = self._severity / 4.0
            o[1] = self._t / self.episode_length
            o[2] = (self._severity - self._prev_severity) / 4.0

            # Observation noise scales with hazard pressure to make sensing realistic.
            sigma = self.observation_noise * (1.0 + 0.8 * self._hazard_pressure)
            noise = self._rng.normal(0, sigma, self.OBS_DIM - 3).astype(np.float32)
            o[3:] = np.clip(o[0] + noise, 0.0, 1.0)

            if i == 1:
                o[4] = float(np.clip(self._hazard_pressure + 0.08 * o[4], 0, 1))
            elif i == 2:
                o[5] = float(np.clip(0.8 * self._hazard_pressure + 0.12 * o[5], 0, 1))
            else:
                o[6] = float(np.clip(0.6 * self._hazard_pressure + 0.10 * o[6], 0, 1))
            obs[i] = o
        return obs

    def _preparedness_score(self, action_dict):
        a1 = int(action_dict.get(1, 0))
        a2 = int(action_dict.get(2, 0))
        a3 = int(action_dict.get(3, 0))

        weighted = 0.35 * a1 + 0.30 * a2 + 0.45 * a3
        coordination = 0.20 if min(a1, a2, a3) >= 1 else 0.0
        return float(weighted + coordination), (a1, a2, a3)

    def step(self, action_dict):
        obs_severity = self._severity

        preparedness, acts = self._preparedness_score(action_dict)
        total_action = float(sum(acts))

        # Stochastic hazard realization and outcome variables.
        shock_prob = np.clip(
            self.hazard_rate + 0.18 * (obs_severity / 4.0) + 0.25 * self._hazard_pressure,
            0.0,
            0.95,
        )
        hazard_shock = 1.0 if self._rng.random() < shock_prob else 0.0

        damage_raw = float((obs_severity + 1.0) * (0.9 + 0.7 * hazard_shock))
        mitigation = float(np.clip(0.8 * preparedness, 0.0, damage_raw))
        damage_after = float(max(0.0, damage_raw - mitigation))
        damage_prevented = float(damage_raw - damage_after)

        resource_cost = float(0.55 * total_action)
        false_alarm = float(0.45 * max(0.0, total_action - (obs_severity + 0.8)))

        if obs_severity >= 3:
            self._high_risk_age += 1
        else:
            self._high_risk_age = max(0, self._high_risk_age - 1)

        response_gap = max(0.0, (obs_severity + 0.5) - preparedness)
        response_penalty = float(0.20 * self._high_risk_age * response_gap)

        violation = 1 if resource_cost > self._resource_budget else 0

        # Outcome-based reward: no oracle action supervision.
        reward = (
            1.30 * damage_prevented
            - 1.10 * damage_after
            - 0.40 * resource_cost
            - 0.80 * false_alarm
            - response_penalty
            - (0.60 if violation else 0.0)
        )

        # P(s' | s, a): preparedness lowers escalation risk; unresolved damage increases it.
        self._prev_severity = self._severity

        p_up = np.clip(0.04 + 0.38 * (damage_after / (damage_raw + 1e-8)), 0.0, 0.85)
        p_down = np.clip(0.06 + 0.30 * (preparedness / 3.0), 0.0, 0.80)
        p = self._rng.random()

        if p < p_up:
            self._severity = min(4, self._severity + 1)
        elif p < p_up + p_down:
            self._severity = max(0, self._severity - 1)

        # Hazard pressure evolves stochastically with persistence.
        self._hazard_pressure = float(
            np.clip(
                0.88 * self._hazard_pressure + 0.12 * hazard_shock + self._rng.normal(0.0, 0.04),
                0.0,
                1.0,
            )
        )

        self._t += 1
        terminated = self._t >= self.episode_length

        self._episode_reward += float(reward)
        self._episode_damage += damage_after
        self._episode_damage_prevented += damage_prevented
        self._episode_violations += int(violation)

        info = {
            "risk_level":   obs_severity / 4.0,
            "violation":    violation,
            "severity":     self._severity,
            "obs_severity": obs_severity,
            "damage_raw": damage_raw,
            "damage_after": damage_after,
            "damage_prevented": damage_prevented,
            "resource_cost": resource_cost,
            "false_alarm_penalty": false_alarm,
            "response_delay_penalty": response_penalty,
            "observation_noise": self.observation_noise,
            "hazard_shock": hazard_shock,
        }
        return self._get_obs(), float(reward), terminated, False, info

    def get_episode_stats(self):
        steps = max(self._t, 1)
        return {
            "episode_reward": float(self._episode_reward),
            "avg_damage_after": float(self._episode_damage / steps),
            "avg_damage_prevented": float(self._episode_damage_prevented / steps),
            "violation_rate": float(self._episode_violations / steps),
            "observation_noise": float(self.observation_noise),
        }

    def get_config(self):
        return {
            "seed": self._init_seed,
            "observation_noise": self.observation_noise,
            "hazard_rate": self.hazard_rate,
        }
