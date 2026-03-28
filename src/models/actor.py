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
               old_log_prob=None, clip_eps=0.2,
               grad_clip_norm=0.5,
               kl_loss_scale_threshold=0.005,
               kl_loss_scale_factor=0.5,
               debug=False):
        """PPO-clipped policy gradient update. Returns approx KL."""
        probs = self.get_action_probs(obs)
        log_prob = float(np.log(probs[action] + 1e-8))
        if old_log_prob is None:
            old_log_prob = log_prob
        params_before = self.state_dict()

        # Exact PPO objective:
        # ratio = exp(logp_new - logp_old)
        # surr1 = ratio * A
        # surr2 = clip(ratio, 1-eps, 1+eps) * A
        # policy_loss = -mean(min(surr1, surr2))
        ratio = np.exp(log_prob - old_log_prob)
        clipped = np.clip(ratio, 1 - clip_eps, 1 + clip_eps)
        surr1 = float(ratio * advantage)
        surr2 = float(clipped * advantage)
        pg_obj = min(surr1, surr2)
        pg_loss = -pg_obj

        # Derivative of min(surr1, surr2) wrt log_prob is active only where
        # unclipped term is selected (clipped branch is constant when clipped).
        use_unclipped = (
            (advantage >= 0.0 and ratio <= (1.0 + clip_eps)) or
            (advantage < 0.0 and ratio >= (1.0 - clip_eps))
        )
        coeff = float(ratio * advantage) if use_unclipped else 0.0

        approx_kl = float(old_log_prob - log_prob)
        if abs(approx_kl) > float(kl_loss_scale_threshold):
            coeff *= float(kl_loss_scale_factor)

        entropy = -float(np.sum(probs * np.log(probs + 1e-8)))
        total_loss = pg_loss - entropy_coef * entropy
        one_hot = np.zeros(self.action_dim, dtype=np.float32)
        one_hot[action] = 1.0

        # Gradient descent on policy_loss (equivalent to ascent on PPO objective).
        d_logits = (probs - one_hot) * coeff

        # Entropy regularization: subtract entropy_coef * H in loss.
        if entropy_coef > 0.0:
            d_logits += float(entropy_coef) * (probs * (np.log(probs + 1e-8) + 1.0))

        h = self._hidden(obs)
        d_h = self.W2.T @ d_logits
        d_pre = d_h * (1.0 - h ** 2)

        gW1 = np.outer(d_pre, obs)
        gb1 = d_pre.copy()
        gW2 = np.outer(d_logits, h)
        gb2 = d_logits.copy()

        grads = [gW1, gb1, gW2, gb2]
        global_grad_norm = float(np.sqrt(sum(float(np.sum(g * g)) for g in grads)))
        grad_clip_scale = 1.0
        if global_grad_norm > float(grad_clip_norm) and global_grad_norm > 1e-12:
            grad_clip_scale = float(grad_clip_norm) / global_grad_norm
            grads = [g * grad_clip_scale for g in grads]

        self.W1 -= lr * grads[0].astype(np.float32)
        self.b1 -= lr * grads[1].astype(np.float32)
        self.W2 -= lr * grads[2].astype(np.float32)
        self.b2 -= lr * grads[3].astype(np.float32)

        new_log_prob = float(np.log(self.get_action_probs(obs)[action] + 1e-8))
        approx_kl_post = float(old_log_prob - new_log_prob)

        params_after = self.state_dict()
        delta_param = float(np.mean([
            np.mean(np.abs(params_after[k] - params_before[k]))
            for k in params_before
        ]))

        if not debug:
            return approx_kl_post

        return {
            "logp_old": float(old_log_prob),
            "logp_new": float(new_log_prob),
            "logp_new_requires_grad": False,
            "manual_gradient_flow": bool(global_grad_norm > 0.0),
            "approx_kl": approx_kl_post,
            "policy_loss": float(pg_loss),
            "grad_abs_means": {
                "W1": float(np.mean(np.abs(grads[0]))),
                "b1": float(np.mean(np.abs(grads[1]))),
                "W2": float(np.mean(np.abs(grads[2]))),
                "b2": float(np.mean(np.abs(grads[3]))),
            },
            "global_grad_norm": global_grad_norm,
            "grad_clip_scale": float(grad_clip_scale),
            "delta_param": delta_param,
            "parameter_shapes": {name: list(p.shape) for name, p in self.named_parameters()},
        }

    def parameters(self):
        yield self.W1; yield self.b1; yield self.W2; yield self.b2

    def named_parameters(self):
        return [
            ("W1", self.W1),
            ("b1", self.b1),
            ("W2", self.W2),
            ("b2", self.b2),
        ]

    def load_weights(self, sd):
        for k in ["W1","b1","W2","b2"]: setattr(self, k, sd[k].copy())

    def state_dict(self):
        return {k: getattr(self, k).copy() for k in ["W1","b1","W2","b2"]}
