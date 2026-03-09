"""
Centralised critic for MARL disaster response.
Architecture: joint_obs_dim(36) → 512 → 4 outputs (V^R + 3×V^C)
hidden_dim=512 matches paper Table 3: "critic hidden=512".
"""
import numpy as np


class CriticNetwork:
    """Centralised V-function: obs_dim*n_agents → 512 → (V_reward, V_cost_1, V_cost_2, V_cost_3)."""

    def __init__(self, obs_dim=12, n_agents=3, hidden_dim=512):
        self.obs_dim = obs_dim
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.out_dim = 1 + n_agents   # V^R + n_agents V^C
        in_dim = obs_dim * n_agents
        rng = np.random.default_rng(0)
        self.W1 = (rng.standard_normal((hidden_dim, in_dim)) * np.sqrt(2.0/in_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = (rng.standard_normal((self.out_dim, hidden_dim)) * np.sqrt(2.0/hidden_dim)).astype(np.float32)
        self.b2 = np.zeros(self.out_dim, dtype=np.float32)

    def _hidden(self, x):
        return np.tanh(self.W1 @ x + self.b1)

    def forward(self, joint_obs):
        return self.W2 @ self._hidden(joint_obs) + self.b2

    def value(self, obs_dict):
        """Return (V_reward, [V_cost_1, V_cost_2, V_cost_3])."""
        x = np.concatenate([obs_dict[i] for i in sorted(obs_dict.keys())]).astype(np.float32)
        out = self.forward(x)
        return float(out[0]), list(out[1:].astype(float))

    def update(self, obs_dict, target_r, target_costs, lr=3e-4):
        """MSE update for all value heads. Returns scalar loss."""
        x = np.concatenate([obs_dict[i] for i in sorted(obs_dict.keys())]).astype(np.float32)
        h = self._hidden(x)
        out = self.W2 @ h + self.b2
        targets = np.array([target_r] + list(target_costs), dtype=np.float32)
        diff = out - targets
        loss = float(np.mean(diff ** 2))
        d_out = 2 * diff / len(diff)
        self.W2 -= lr * np.outer(d_out, h)
        self.b2 -= lr * d_out
        d_h = self.W2.T @ d_out
        d_pre = d_h * (1.0 - h ** 2)
        self.W1 -= lr * np.outer(d_pre, x)
        self.b1 -= lr * d_pre
        return loss

    def parameters(self):
        yield self.W1; yield self.b1; yield self.W2; yield self.b2

    def load_weights(self, sd):
        for k in ["W1","b1","W2","b2"]: setattr(self, k, sd[k].copy())

    def state_dict(self):
        return {k: getattr(self, k).copy() for k in ["W1","b1","W2","b2"]}
