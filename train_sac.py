from __future__ import annotations

import os
from pathlib import Path

from higher_lower_rl import SACPolicyStrategy, save_policy_weights, train_self_play_sac


# Mirrors the PPO walkthrough setup in walkthrough.ipynb, with SAC-specific settings added.

# Your training budget is measured in complete self-play games.
TOTAL_GAMES = int(os.environ.get("TOTAL_GAMES", "2000000"))
GAMES_PER_BATCH = int(os.environ.get("GAMES_PER_BATCH", "128"))
CHECKPOINT_PATH = Path(os.environ.get("CHECKPOINT_PATH", "checkpoints/jax_sac_self_play.pkl"))
RESUME_FROM_CHECKPOINT = os.environ.get("RESUME_FROM_CHECKPOINT") or None
POLICY_EXPORT_PATH = Path(os.environ.get("POLICY_EXPORT_PATH", "artifacts/submission_policy_sac.pkl"))

TRAINING_CONFIG = dict(
    seed=0,
    total_games=TOTAL_GAMES,
    episodes_per_batch=GAMES_PER_BATCH,
    replay_buffer_size=50_000,
    warmup_games=32,
    gradient_steps_per_update=128,
    batch_size=256,
    learning_rate=3e-4,
    alpha=0.1,
    target_update_tau=0.01,
    use_wandb=False,
    checkpoint_path=CHECKPOINT_PATH,
    resume_from_checkpoint=RESUME_FROM_CHECKPOINT,
    checkpoint_every_games=2_000,
)


def main() -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    POLICY_EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("training budget (games):", TOTAL_GAMES)
    print("checkpoint:", CHECKPOINT_PATH)
    if RESUME_FROM_CHECKPOINT is not None:
        print("resume from checkpoint:", RESUME_FROM_CHECKPOINT)

    trained_params, training_history = train_self_play_sac(**TRAINING_CONFIG)
    strategy = SACPolicyStrategy(trained_params, name="MyStrategy(SAC)")
    export_info = save_policy_weights(
        POLICY_EXPORT_PATH,
        trained_params,
        metadata={
            "name": str(strategy),
            "checkpoint_path": str(CHECKPOINT_PATH),
            "total_games": TOTAL_GAMES,
        },
    )

    print("exported policy:", export_info["path"])
    print("export sha256:", export_info["sha256"])
    print("training updates:", len(training_history.get("games_played", [])))


if __name__ == "__main__":
    main()
