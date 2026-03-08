#!/usr/bin/env python3
"""
scripts/create_checkpoints.py
==============================
Generate the two pretrained checkpoint files required by the repository:

    checkpoints/marl_policy.pt         — LagrangianCTDE actors + critic
    checkpoints/perception_encoder.pt  — ViT perception encoder

This script produces RANDOMLY INITIALISED (not trained) checkpoints whose
only purpose is to let the repository run end-to-end immediately after
cloning, without requiring a full training run. All weights are seeded
deterministically (torch.manual_seed(0)) so the files are byte-for-byte
reproducible across machines.

Checkpoint formats are **exactly** those expected by the existing loaders:

  marl_policy.pt
  ─────────────
  Loaded by LagrangianCTDE.load_checkpoint() in src/algorithms/lagrangian_ctde.py:
    {
      "episode"                   : int,
      "best_reward"               : float,
      "lambdas"                   : {1: float, 2: float, 3: float},
      "critic_state_dict"         : critic.state_dict(),
      "critic_optimizer_state_dict": optimizer.state_dict(),
      "config"                    : LagrangianCTDEConfig.__dict__,
      "actors"                    : {1: sd, 2: sd, 3: sd},
      "actor_optimizers"          : {1: opt_sd, 2: opt_sd, 3: opt_sd},
    }

  perception_encoder.pt
  ─────────────────────
  Loaded by ViTEncoder and train_encoder() in src/models/vit_encoder.py:
    {
      "model_state_dict" : encoder.state_dict(),
      "feature_dim"      : 128,
      "backbone"         : str,   # architecture tag
      "output_dim"       : 128,
    }

Usage
-----
    python scripts/create_checkpoints.py
    python scripts/create_checkpoints.py --out_dir checkpoints/
    python scripts/create_checkpoints.py --seed 42 --device cpu
    python scripts/create_checkpoints.py --skip_encoder   # skip ViT (no timm)
    python scripts/create_checkpoints.py --force          # overwrite existing

Exit codes
----------
    0 — all requested checkpoints written successfully
    1 — unrecoverable error (missing torch, bad out_dir, etc.)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic placeholder checkpoints for the repository.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--out_dir",
        default="checkpoints",
        help="Directory to write checkpoints (default: checkpoints/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Master random seed for reproducibility (default: 0)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for model init (default: cpu)",
    )
    parser.add_argument(
        "--skip_encoder",
        action="store_true",
        help=(
            "Skip perception_encoder.pt generation. "
            "Use this flag if timm is not installed."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing checkpoint files.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint 1: marl_policy.pt
# ---------------------------------------------------------------------------

def create_marl_policy(
    out_path: Path,
    seed: int,
    device_str: str,
) -> None:
    """
    Build a randomly initialised LagrangianCTDE checkpoint.

    Instantiates the exact same models used at training time and saves
    the full checkpoint dict that LagrangianCTDE.load_checkpoint() expects.
    No environment interaction or training is performed.

    Architecture (from LagrangianCTDEConfig defaults / Table 3):
      3 × ActorNetwork  (obs_dim=12, hidden=256, n_actions=4)
      1 × CriticNetwork (state_dim=24, hidden=512, n_constraint_heads=3)
      3 × Adam actor optimisers
      1 × Adam critic optimiser
    """
    import torch
    import torch.optim as optim

    # Project root must be on sys.path so src.* imports resolve.
    _ensure_src_on_path()

    from src.models.actor import ActorNetwork
    from src.models.critic import CriticNetwork

    # ── Deterministic initialisation ─────────────────────────────────────────
    torch.manual_seed(seed)

    device = torch.device(device_str)

    # Hyperparameters — Table 3 defaults, must match LagrangianCTDEConfig
    OBS_DIM      = 12
    STATE_DIM    = 24
    N_ACTIONS    = 4
    N_AGENTS     = 3
    ACTOR_HIDDEN = 256
    CRITIC_HIDDEN = 512
    LR           = 3e-4

    logger.info("  Building 3 × ActorNetwork(obs_dim=%d, hidden=%d, n_actions=%d) ...",
                OBS_DIM, ACTOR_HIDDEN, N_ACTIONS)

    # Build one actor per agent (independent parameters)
    actors: Dict[int, ActorNetwork] = {
        i: ActorNetwork(
            obs_dim=OBS_DIM,
            n_actions=N_ACTIONS,
            hidden_size=ACTOR_HIDDEN,
            agent_id=i,
        ).to(device)
        for i in range(1, N_AGENTS + 1)
    }

    logger.info(
        "  Building CriticNetwork(state_dim=%d, hidden=%d, n_constraint_heads=%d) ...",
        STATE_DIM, CRITIC_HIDDEN, N_AGENTS,
    )

    critic = CriticNetwork(
        state_dim=STATE_DIM,
        hidden_size=CRITIC_HIDDEN,
        n_agents=N_AGENTS,
        n_constraint_heads=N_AGENTS,   # one cost-value head per agent (Eq. 7)
    ).to(device)

    # ── Optimisers ────────────────────────────────────────────────────────────
    # Adam state_dicts must exist in the checkpoint for load_checkpoint() to
    # restore them. We create real optimisers (even though no training step
    # has been taken) so the state dict structure matches the trained format.
    actor_optimizers = {
        i: optim.Adam(actors[i].parameters(), lr=LR)
        for i in range(1, N_AGENTS + 1)
    }
    critic_optimizer = optim.Adam(critic.parameters(), lr=LR)

    # ── Parameter counts ─────────────────────────────────────────────────────
    total_actor_params = sum(
        sum(p.numel() for p in actors[i].parameters())
        for i in range(1, N_AGENTS + 1)
    )
    critic_params = sum(p.numel() for p in critic.parameters())
    logger.info(
        "  Parameters — actors: %s  critic: %s  total: %s",
        f"{total_actor_params:,}",
        f"{critic_params:,}",
        f"{total_actor_params + critic_params:,}",
    )

    # ── Config snapshot ───────────────────────────────────────────────────────
    # Mirrors LagrangianCTDEConfig.__dict__ so load_checkpoint() can restore it.
    config_snapshot = {
        "obs_dim":        OBS_DIM,
        "state_dim":      STATE_DIM,
        "n_actions":      N_ACTIONS,
        "n_agents":       N_AGENTS,
        "actor_hidden":   ACTOR_HIDDEN,
        "critic_hidden":  CRITIC_HIDDEN,
        "lr_actor":       LR,
        "lr_critic":      LR,
        "lr_lambda":      1e-3,
        "gamma":          0.99,
        "gae_lambda":     0.95,
        "clip_epsilon":   0.2,
        "entropy_coef":   0.01,
        "max_grad_norm":  0.5,
        "mini_batches":   4,
        "constraint_d":   0.10,
        "lambda_init":    0.0,
        "lambda_max":     10.0,
        "n_episodes":     1000,
        "eval_interval":  50,
        "save_interval":  100,
        "checkpoint_dir": "checkpoints/",
        "log_interval":   10,
        "device":         device_str,
        "seed":           seed,
    }

    # ── Build checkpoint dict — must exactly match LagrangianCTDE.save_checkpoint() ──
    checkpoint = {
        # Training state
        "episode":      0,
        "best_reward":  float("-inf"),
        "lambdas":      {i: 0.0 for i in range(1, N_AGENTS + 1)},

        # Model weights
        "critic_state_dict":   critic.state_dict(),
        "actors": {
            i: actors[i].state_dict()
            for i in range(1, N_AGENTS + 1)
        },

        # Optimiser states (structure must match; values are empty at step 0)
        "critic_optimizer_state_dict": critic_optimizer.state_dict(),
        "actor_optimizers": {
            i: actor_optimizers[i].state_dict()
            for i in range(1, N_AGENTS + 1)
        },

        # Configuration snapshot (used for reproducibility logging)
        "config": config_snapshot,

        # Provenance
        "created_by":   "scripts/create_checkpoints.py",
        "trained":      False,   # mark as placeholder — NOT a trained checkpoint
        "seed":         seed,
    }

    torch.save(checkpoint, out_path)
    logger.info("  ✓ Saved marl_policy.pt  (%s)", _file_size(out_path))

    # ── Verify round-trip load ────────────────────────────────────────────────
    logger.info("  Verifying round-trip load ...")
    ckpt_verify = torch.load(out_path, map_location="cpu", weights_only=True)
    assert "actors" in ckpt_verify,               "missing 'actors' key"
    assert "critic_state_dict" in ckpt_verify,    "missing 'critic_state_dict'"
    assert "critic_optimizer_state_dict" in ckpt_verify
    assert "actor_optimizers" in ckpt_verify
    assert "lambdas" in ckpt_verify
    assert set(ckpt_verify["actors"].keys()) == {1, 2, 3}

    # Verify each actor state dict can be loaded back into a fresh ActorNetwork
    for i in range(1, N_AGENTS + 1):
        fresh = ActorNetwork(OBS_DIM, N_ACTIONS, ACTOR_HIDDEN, agent_id=i)
        fresh.load_state_dict(ckpt_verify["actors"][i])

    # Verify critic state dict round-trips
    fresh_critic = CriticNetwork(STATE_DIM, CRITIC_HIDDEN, N_AGENTS, N_AGENTS)
    fresh_critic.load_state_dict(ckpt_verify["critic_state_dict"])

    logger.info("  ✓ Round-trip verification passed")


# ---------------------------------------------------------------------------
# Checkpoint 2: perception_encoder.pt
# ---------------------------------------------------------------------------

def create_perception_encoder(
    out_path: Path,
    seed: int,
    device_str: str,
) -> bool:
    """
    Build a randomly initialised ViTEncoder checkpoint.

    Requires timm>=0.9 (ViT-B/16 backbone). If timm is not installed,
    falls back to a lightweight stub encoder that mirrors the ViTEncoder
    state dict structure for all non-backbone layers.

    Returns True if a real ViTEncoder was used, False if the stub path
    was taken (timm missing).
    """
    import torch

    _ensure_src_on_path()

    torch.manual_seed(seed)
    device = torch.device(device_str)

    OUTPUT_DIM = 128
    MLP_HIDDEN = 256

    # ── Attempt real ViTEncoder ───────────────────────────────────────────────
    timm_available = False
    try:
        import timm  # noqa: F401
        timm_available = True
    except ImportError:
        pass

    if timm_available:
        return _create_real_vit_encoder(out_path, seed, device, OUTPUT_DIM, MLP_HIDDEN)
    else:
        logger.warning(
            "  timm not installed — falling back to LightweightEncoder stub.\n"
            "  The perception_encoder.pt stub is sufficient for demo.sh.\n"
            "  For real perception-coupled training, install timm and re-run:\n"
            "      pip install timm==0.9.12\n"
            "      python scripts/create_checkpoints.py --force"
        )
        return _create_stub_encoder(out_path, seed, device, OUTPUT_DIM, MLP_HIDDEN)


def _create_real_vit_encoder(
    out_path: Path,
    seed: int,
    device: "torch.device",
    output_dim: int,
    mlp_hidden: int,
) -> bool:
    """Create perception_encoder.pt using the real ViTEncoder with pretrained=False."""
    import torch
    from src.models.vit_encoder import ViTEncoder

    logger.info(
        "  Building ViTEncoder(backbone=vit_base_patch16_224, pretrained=False, "
        "output_dim=%d) ...", output_dim,
    )

    # pretrained=False avoids downloading ~350 MB of ImageNet weights —
    # the random init is sufficient for a placeholder checkpoint.
    encoder = ViTEncoder(
        output_dim=output_dim,
        pretrained=False,
        backbone="vit_base_patch16_224",
        mlp_hidden=mlp_hidden,
        freeze_backbone_epochs=10,
    ).to(device)

    n_params = encoder.count_parameters(trainable_only=False)
    logger.info("  Parameters: %s", f"{n_params:,}")

    checkpoint = {
        "model_state_dict": encoder.state_dict(),
        "feature_dim":      output_dim,
        "output_dim":       output_dim,
        "backbone":         "vit_base_patch16_224",
        "mlp_hidden":       mlp_hidden,
        "created_by":       "scripts/create_checkpoints.py",
        "trained":          False,
        "seed":             seed,
        "stub":             False,
    }

    torch.save(checkpoint, out_path)
    logger.info("  ✓ Saved perception_encoder.pt  (%s)", _file_size(out_path))

    # Round-trip verify
    logger.info("  Verifying round-trip load ...")
    ckpt_v = torch.load(out_path, map_location="cpu", weights_only=True)
    fresh = ViTEncoder(output_dim=output_dim, pretrained=False, mlp_hidden=mlp_hidden)
    fresh.load_state_dict(ckpt_v["model_state_dict"])
    logger.info("  ✓ Round-trip verification passed")

    return True


def _create_stub_encoder(
    out_path: Path,
    seed: int,
    device: "torch.device",
    output_dim: int,
    mlp_hidden: int,
) -> bool:
    """
    Create a minimal stub encoder checkpoint when timm is unavailable.

    Uses LightweightEncoder — a pure-PyTorch replacement for ViTEncoder
    that has the same projector and input-projection layers but replaces
    the ViT-B/16 backbone with a simple CNN that produces 768-dim output,
    matching the [CLS] token dimension.

    The stub IS loadable by LightweightEncoder but NOT by ViTEncoder (since
    the backbone layers have different names). Its sole purpose is to allow
    demo.sh to succeed before timm is installed. Replace with the real
    encoder by running with timm installed.
    """
    import torch
    import torch.nn as nn

    VIT_EMBED_DIM = 768   # must match ViTEncoder.VIT_EMBED_DIM

    class _LightweightBackbone(nn.Module):
        """
        CPU-friendly CNN backbone producing a 768-dim 'CLS-equivalent' vector.
        Input: (B, 3, 224, 224).  Output: (B, 768).
        Architecture deliberately mimics the ViT-B/16 output dimension so that
        the downstream projector has the same shape as in the real encoder.
        """
        def __init__(self) -> None:
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64,  kernel_size=7, stride=4, padding=3),   # →56×56
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # →28×28
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # →14×14
                nn.ReLU(),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), # →7×7
                nn.ReLU(),
            )
            self.pool = nn.AdaptiveAvgPool2d((1, 1))  # →(B, 512, 1, 1)
            self.fc   = nn.Linear(512, VIT_EMBED_DIM) # →(B, 768)

        def forward_features(self, x: torch.Tensor) -> torch.Tensor:
            # Returns shape (B, seq_len, VIT_EMBED_DIM) matching timm convention.
            # For stub use: seq_len=1, index 0 is the "CLS" token.
            feat = self.stem(x)                      # (B, 512, 7, 7)
            feat = self.pool(feat).flatten(1)         # (B, 512)
            cls  = self.fc(feat).unsqueeze(1)         # (B, 1, 768)
            return cls

    class LightweightEncoder(nn.Module):
        """
        Stub encoder with identical interface to ViTEncoder but no timm dep.
        Produces φ_t ∈ ℝ^{output_dim} from (x_radar, x_satellite) inputs.
        """
        VIT_EMBED_DIM: int = VIT_EMBED_DIM

        def __init__(self, output_dim: int = 128, mlp_hidden: int = 256) -> None:
            super().__init__()
            self.output_dim = output_dim

            self.backbone       = _LightweightBackbone()
            self.radar_proj     = nn.Conv2d(1, 3, kernel_size=1, bias=False)
            self.satellite_proj = nn.Conv2d(3, 3, kernel_size=1, bias=False)

            fused_dim = 2 * self.VIT_EMBED_DIM
            self.projector = nn.Sequential(
                nn.Linear(fused_dim, mlp_hidden),
                nn.LayerNorm(mlp_hidden),
                nn.GELU(),
                nn.Linear(mlp_hidden, output_dim),
                nn.LayerNorm(output_dim),
            )
            self._init_weights()

        def _extract_cls(self, x: torch.Tensor) -> torch.Tensor:
            features = self.backbone.forward_features(x)  # (B, 1, 768)
            return features[:, 0, :]                       # (B, 768)

        def forward(
            self,
            x_radar: torch.Tensor,
            x_satellite: torch.Tensor,
        ) -> torch.Tensor:
            r   = self._extract_cls(self.radar_proj(x_radar))
            s   = self._extract_cls(self.satellite_proj(x_satellite))
            phi = self.projector(torch.cat([r, s], dim=-1))
            return phi

        def encode(self, x_radar: torch.Tensor, x_satellite: torch.Tensor) -> torch.Tensor:
            return self.forward(x_radar, x_satellite)

        def _init_weights(self) -> None:
            import torch.nn.init as init
            init.kaiming_normal_(self.radar_proj.weight,     mode="fan_out")
            init.kaiming_normal_(self.satellite_proj.weight, mode="fan_out")
            for layer in self.projector:
                if isinstance(layer, nn.Linear):
                    init.trunc_normal_(layer.weight, std=0.02)
                    if layer.bias is not None:
                        init.zeros_(layer.bias)

    logger.info(
        "  Building LightweightEncoder stub (output_dim=%d, mlp_hidden=%d) ...",
        output_dim, mlp_hidden,
    )

    encoder = LightweightEncoder(output_dim=output_dim, mlp_hidden=mlp_hidden).to(device)
    n_params = sum(p.numel() for p in encoder.parameters())
    logger.info("  Parameters: %s", f"{n_params:,}")

    checkpoint = {
        "model_state_dict": encoder.state_dict(),
        "feature_dim":      output_dim,
        "output_dim":       output_dim,
        "backbone":         "lightweight_cnn_stub",
        "mlp_hidden":       mlp_hidden,
        "created_by":       "scripts/create_checkpoints.py",
        "trained":          False,
        "seed":             seed,
        "stub":             True,   # sentinel — replace with real encoder when timm is available
        "stub_note": (
            "This checkpoint was generated without timm. "
            "Install timm==0.9.12 and re-run create_checkpoints.py --force "
            "to replace with a real ViTEncoder checkpoint."
        ),
    }

    torch.save(checkpoint, out_path)
    logger.info("  ✓ Saved perception_encoder.pt (stub)  (%s)", _file_size(out_path))

    # Verify stub loads cleanly
    ckpt_v = torch.load(out_path, map_location="cpu", weights_only=False)
    assert ckpt_v["stub"] is True
    assert "model_state_dict" in ckpt_v
    assert ckpt_v["feature_dim"] == output_dim
    logger.info("  ✓ Stub round-trip verification passed")

    return False


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _ensure_src_on_path() -> None:
    """Add the project root to sys.path so `from src.models.*` imports work."""
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def _file_size(path: Path) -> str:
    size = path.stat().st_size
    if size >= 1_000_000:
        return f"{size / 1_000_000:.1f} MB"
    return f"{size / 1_000:.1f} KB"


def _should_skip(path: Path, force: bool) -> bool:
    """Return True if the file exists and --force was not set."""
    if path.exists() and not force:
        logger.info(
            "  Skipping (already exists): %s  "
            "Pass --force to regenerate.", path,
        )
        return True
    return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    # ── Sanity-check torch ─────────────────────────────────────────────────────
    try:
        import torch
        logger.info("torch %s  |  CUDA: %s", torch.__version__,
                    torch.cuda.is_available())
    except ImportError:
        logger.error(
            "PyTorch is not installed.\n"
            "Install it with: pip install torch\n"
            "then re-run this script."
        )
        return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", out_dir.resolve())

    # ── Checkpoint 1: marl_policy.pt ──────────────────────────────────────────
    marl_path = out_dir / "marl_policy.pt"
    logger.info("")
    logger.info("[ 1 / 2 ]  marl_policy.pt")
    logger.info("─" * 55)

    if not _should_skip(marl_path, args.force):
        try:
            create_marl_policy(marl_path, seed=args.seed, device_str=args.device)
        except Exception as exc:
            logger.error("Failed to create marl_policy.pt: %s", exc)
            raise

    # ── Checkpoint 2: perception_encoder.pt ───────────────────────────────────
    enc_path = out_dir / "perception_encoder.pt"
    logger.info("")
    logger.info("[ 2 / 2 ]  perception_encoder.pt")
    logger.info("─" * 55)

    if args.skip_encoder:
        logger.info("  Skipping (--skip_encoder flag set).")
    elif not _should_skip(enc_path, args.force):
        try:
            real = create_perception_encoder(enc_path, seed=args.seed, device_str=args.device)
            if not real:
                logger.info("  (stub encoder saved — install timm for real ViT weights)")
        except Exception as exc:
            logger.error("Failed to create perception_encoder.pt: %s", exc)
            logger.error("Use --skip_encoder to skip this checkpoint if timm is unavailable.")
            raise

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 55)
    logger.info("  Checkpoints ready in: %s", out_dir.resolve())
    for fname in ["marl_policy.pt", "perception_encoder.pt"]:
        p = out_dir / fname
        if p.exists():
            logger.info("    ✓  %-30s  %s", fname, _file_size(p))
        else:
            logger.info("    –  %-30s  (not generated)", fname)
    logger.info("=" * 55)
    logger.info("")
    logger.info("Run the demo with:")
    logger.info("  bash scripts/demo.sh")

    return 0


if __name__ == "__main__":
    sys.exit(main())
