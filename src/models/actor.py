"""
Actor network for MARL disaster response.
Pure-numpy MLP with PPO-compatible softmax policy.
hidden_dim=256 matches paper Table 3: "actor hidden=256".
"""
import numpy as np


class ActorNetwork:
    """2-layer MLP actor: obs_dim → 256 → 4 (softmax)."""

    def __init__(self, obs_dim=12, action_dim=4, hidden_dim=256, agent_id=1):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.agent_id = agent_id
        rng = np.random.default_rng(agent_id * 42 + 7)
        self.W1 = (rng.standard_normal((hidden_dim, obs_dim)) * np.sqrt(2.0/obs_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = (rng.standard_normal((action_dim, hidden_dim)) * np.sqrt(2.0/hidden_dim)).astype(np.float32)
        self.b2 = np.zeros(action_dim, dtype=np.float32)

    def _hidden(self, obs):
        return np.tanh(self.W1 @ obs + self.b1)

    def forward(self, obs):
        return self.W2 @ self._hidden(obs) + self.b2

    def get_action_probs(self, obs):
        logits = self.forward(obs)
        logits -= logits.max()
        probs = np.exp(logits)
        return probs / probs.sum()

    def get_action(self, obs, deterministic=True):
        probs = self.get_action_probs(obs)
        if deterministic:
            return int(np.argmax(probs))
        return int(np.random.choice(len(probs), p=probs))

    def log_prob(self, obs, action):
        return float(np.log(self.get_action_probs(obs)[action] + 1e-8))

    def update(self, obs, action, advantage, lr=3e-4, entropy_coef=0.01,
               old_log_prob=None, clip_eps=0.2):
        """PPO-clipped policy gradient update. Returns approx KL."""
        probs = self.get_action_probs(obs)
        log_prob = float(np.log(probs[action] + 1e-8))
        if old_log_prob is None:
            old_log_prob = log_prob
        ratio = np.exp(log_prob - old_log_prob)
        clipped = np.clip(ratio, 1 - clip_eps, 1 + clip_eps)
        pg_loss = -min(float(ratio * advantage), float(clipped * advantage))
        entropy = -float(np.sum(probs * np.log(probs + 1e-8)))
        total_loss = pg_loss - entropy_coef * entropy
        one_hot = np.zeros(self.action_dim, dtype=np.float32)
        one_hot[action] = 1.0
        d_logits = (probs - one_hot) * total_loss
        h = self._hidden(obs)
        self.W2 -= lr * np.outer(d_logits, h)
        self.b2 -= lr * d_logits
        d_h = self.W2.T @ d_logits
        d_pre = d_h * (1.0 - h ** 2)
        self.W1 -= lr * np.outer(d_pre, obs)
        self.b1 -= lr * d_pre
        return float(old_log_prob - log_prob)

    def parameters(self):
        yield self.W1; yield self.b1; yield self.W2; yield self.b2

    def load_weights(self, sd):
        for k in ["W1","b1","W2","b2"]: setattr(self, k, sd[k].copy())

    def state_dict(self):
        return {k: getattr(self, k).copy() for k in ["W1","b1","W2","b2"]}
