"""
Compatibility training entry point.

This module preserves the `python -m src.train` interface used by scripts
and documentation while delegating implementation to `src.orchestration.train`.
"""

from src.orchestration.train import train, _parse_args


if __name__ == "__main__":
    args = _parse_args()
    result = train(
        algo=args.algo,
        config_path=args.config,
        n_episodes=args.episodes,
        device=args.device,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )

    print("\n" + "=" * 50)
    print(f"  Algorithm        : {result['algo']}")
    print(f"  Episodes trained : {result['episodes']}")
    print(f"  Final reward     : {result['final_reward']:.2f}")
    print(f"  Final VR         : {result['final_violation_rate']:.4f}")
    print(f"  Training time    : {result['training_time_sec']:.1f}s")
    print(f"  Checkpoint       : {result['checkpoint_path']}")
    print("=" * 50)
