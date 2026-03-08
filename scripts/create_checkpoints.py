#!/usr/bin/env python3
"""
scripts/create_checkpoints.py
==============================
Prepare the two checkpoint files required by the repository:

    checkpoints/marl_policy.pt         — LagrangianCTDE actors + critic
    checkpoints/perception_encoder.pt  — downloaded pretrained ViT perception encoder

This script produces a RANDOMLY INITIALISED (not trained) marl_policy
checkpoint so the repository can run end-to-end immediately after cloning,
without requiring a full training run.

The perception encoder checkpoint is no longer generated locally because
the file is large (300MB+). It is downloaded from Hugging Face instead.

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
    Downloaded from Hugging Face and loaded by ViTEncoder/train_encoder().

Usage
-----
    python scripts/create_checkpoints.py
    python scripts/create_checkpoints.py --out_dir checkpoints/
    python scripts/create_checkpoints.py --seed 42 --device cpu
    python scripts/create_checkpoints.py --skip_encoder   # skip encoder download
    python scripts/create_checkpoints.py --force          # overwrite existing

Exit codes
----------
    0 — all requested checkpoints written successfully
    1 — unrecoverable error (missing torch, bad out_dir, etc.)
"""

from __future__ import annotations

import argparse
import logging
import sys
import urllib.request
from urllib.error import HTTPError, URLError
from pathlib import Path
from typing import Dict


DEFAULT_PERCEPTION_URL = (
    "https://huggingface.co/datasets/aliakarma/perception_encoder/resolve/main/"
    "perception_encoder.pt"
)

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
        description="Prepare required checkpoints for the repository.",
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
        help="Skip perception_encoder.pt download.",
    )
    parser.add_argument(
        "--perception_url",
        default=DEFAULT_PERCEPTION_URL,
        help=(
            "Download URL for perception_encoder.pt "
            f"(default: {DEFAULT_PERCEPTION_URL})"
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
# Checkpoint 2: perception_encoder.pt (download)
# ---------------------------------------------------------------------------

def download_perception_encoder(out_path: Path, url: str) -> None:
    """Download pretrained perception encoder checkpoint from Hugging Face."""
    logger.info("  Downloading pretrained perception encoder from Hugging Face ...")
    logger.info("  URL: %s", url)

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    try:
        with urllib.request.urlopen(url) as response, open(tmp_path, "wb") as f:
            total = response.headers.get("Content-Length")
            total_bytes = int(total) if total else None
            downloaded = 0

            while True:
                chunk = response.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)

                if total_bytes:
                    pct = (downloaded / total_bytes) * 100
                    logger.info(
                        "  Downloaded: %.1f%% (%s / %s)",
                        pct,
                        _human_size(downloaded),
                        _human_size(total_bytes),
                    )
                else:
                    logger.info("  Downloaded: %s", _human_size(downloaded))

        tmp_path.replace(out_path)
        logger.info("  ✓ Saved perception_encoder.pt  (%s)", _file_size(out_path))
    except (HTTPError, URLError) as exc:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"failed to download checkpoint from {url}: {exc}") from exc
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


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


def _human_size(num_bytes: int) -> str:
    if num_bytes >= 1_000_000_000:
        return f"{num_bytes / 1_000_000_000:.2f} GB"
    if num_bytes >= 1_000_000:
        return f"{num_bytes / 1_000_000:.1f} MB"
    if num_bytes >= 1_000:
        return f"{num_bytes / 1_000:.1f} KB"
    return f"{num_bytes} B"


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
            download_perception_encoder(enc_path, args.perception_url)
        except Exception as exc:
            logger.error("Failed to create perception_encoder.pt: %s", exc)
            logger.error("Use --skip_encoder to skip downloading this checkpoint.")
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
