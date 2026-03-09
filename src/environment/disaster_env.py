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
    episode_length = 100

    OPTIMAL_ACTIONS = np.array([
        [0, 1, 2, 3, 3],  # Storm agent
        [0, 0, 1, 2, 3],  # Flood agent
        [0, 0, 1, 1, 2],  # Evacuation agent
    ], dtype=np.int32)

    def __init__(self, seed=None):
        self._init_seed = seed
        self._rng = np.random.default_rng(seed)
        self._t = 0
        self._severity = 1
        self._prev_severity = 1

    def reset(self, seed=None):
        s = seed if seed is not None else self._init_seed
        self._rng = np.random.default_rng(s)
        self._t = 0
        self._severity = int(self._rng.integers(1, 3))
        self._prev_severity = self._severity
        return self._get_obs(), {}

    def _get_obs(self):
        obs = {}
        for i in range(1, 4):
            o = np.zeros(self.OBS_DIM, dtype=np.float32)
            o[0] = self._severity / 4.0
            o[1] = self._t / self.episode_length
            o[2] = (self._severity - self._prev_severity) / 4.0
            noise = self._rng.normal(0, 0.02, self.OBS_DIM - 3).astype(np.float32)
            o[3:] = np.clip(o[0] + noise, 0.0, 1.0)
            if i == 1:
                o[4] = float(np.clip(o[4] + 0.10 * self._severity / 4, 0, 1))
            elif i == 2:
                o[5] = float(np.clip(o[5] + 0.15 * self._severity / 4, 0, 1))
            else:
                o[6] = float(np.clip(o[6] + 0.05 * self._severity / 4, 0, 1))
            obs[i] = o
        return obs

    def step(self, action_dict):
        obs_severity = self._severity   # severity the agent observed

        reward = 0.23
        violation = 0

        for idx, i in enumerate(range(1, 4)):
            act = int(action_dict.get(i, 0))
            opt = int(self.OPTIMAL_ACTIONS[idx, obs_severity])
            diff = abs(act - opt)
            if diff == 0:
                reward += 0.52
            elif diff == 1:
                reward += 0.20
            elif diff == 2:
                reward += 0.0
            else:
                reward -= 0.10
            if obs_severity >= 2 and (act < opt - 1):
                violation = 1

        if violation:
            reward -= 0.25

        # Severity transition with slight downward bias (realistic storm lifecycle)
        self._prev_severity = self._severity
        p = self._rng.random()
        if p < 0.070:
            self._severity = min(4, self._severity + 1)
        elif p < 0.157:    # 0.062 + 0.087
            self._severity = max(0, self._severity - 1)

        self._t += 1
        terminated = self._t >= self.episode_length
        info = {
            "risk_level":   obs_severity / 4.0,
            "violation":    violation,
            "severity":     self._severity,
            "obs_severity": obs_severity,
        }
        return self._get_obs(), float(reward), terminated, False, info
