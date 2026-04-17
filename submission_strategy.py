from __future__ import annotations

import os
from pathlib import Path
from typing import List

import numpy as np

import game
from higher_lower_rl import (
    PolicyStrategy,
    RemotePolicyStrategy,
    download_policy_weights,
    load_training_checkpoint,
    train_self_play,
)


# Set this before submission, or provide HIGHER_LOWER_POLICY_SOURCE at runtime.
# This default points at the uploaded checkpoint in fracapuano/sqif26.
# If you prefer inference-only loading, switch to:
# https://raw.githubusercontent.com/fracapuano/sqif26/main/artifacts/submission_policy.pkl
POLICY_SOURCE = os.environ.get(
    "HIGHER_LOWER_POLICY_SOURCE",
    "https://raw.githubusercontent.com/fracapuano/sqif26/main/artifacts/jax_ppo_self_play.pkl",
)
POLICY_SHA256 = os.environ.get(
    "HIGHER_LOWER_POLICY_SHA256",
    "f7e816a830bfe02555bbc358291eec1d202684d608865786ad677b3526823cf5",
) or None
POLICY_CACHE_DIR = os.environ.get("HIGHER_LOWER_POLICY_CACHE_DIR") or None
POLICY_NAME = os.environ.get("HIGHER_LOWER_POLICY_NAME", "SubmissionPolicy")
POLICY_SOURCE_KIND = os.environ.get("HIGHER_LOWER_POLICY_SOURCE_KIND", "checkpoint").strip().lower()
POLICY_DOWNLOAD_TIMEOUT_SECONDS = float(os.environ.get("HIGHER_LOWER_POLICY_TIMEOUT_SECONDS", "60"))
POLICY_FORCE_DOWNLOAD = os.environ.get("HIGHER_LOWER_POLICY_FORCE_DOWNLOAD", "0") == "1"

# If POLICY_SOURCE_KIND == "checkpoint", MyStrategy can keep training after download.
TRAIN_EXTRA_GAMES = int(os.environ.get("HIGHER_LOWER_TRAIN_EXTRA_GAMES", "0"))
TRAIN_EPISODES_PER_BATCH = int(os.environ.get("HIGHER_LOWER_TRAIN_EPISODES_PER_BATCH", "128"))
TRAIN_PPO_EPOCHS = int(os.environ.get("HIGHER_LOWER_TRAIN_PPO_EPOCHS", "4"))
TRAIN_MINIBATCH_SIZE = int(os.environ.get("HIGHER_LOWER_TRAIN_MINIBATCH_SIZE", "256"))
TRAIN_LEARNING_RATE = float(os.environ.get("HIGHER_LOWER_TRAIN_LEARNING_RATE", "3e-4"))
TRAIN_GAMMA = float(os.environ.get("HIGHER_LOWER_TRAIN_GAMMA", "0.99"))
TRAIN_GAE_LAMBDA = float(os.environ.get("HIGHER_LOWER_TRAIN_GAE_LAMBDA", "0.95"))
TRAIN_CLIP_EPSILON = float(os.environ.get("HIGHER_LOWER_TRAIN_CLIP_EPSILON", "0.2"))
TRAIN_VALUE_COEF = float(os.environ.get("HIGHER_LOWER_TRAIN_VALUE_COEF", "0.5"))
TRAIN_ENTROPY_COEF = float(os.environ.get("HIGHER_LOWER_TRAIN_ENTROPY_COEF", "5e-3"))
TRAIN_SAC_REPLAY_BUFFER_SIZE = int(os.environ.get("HIGHER_LOWER_TRAIN_SAC_REPLAY_BUFFER_SIZE", "50000"))
TRAIN_SAC_WARMUP_GAMES = int(os.environ.get("HIGHER_LOWER_TRAIN_SAC_WARMUP_GAMES", "32"))
TRAIN_SAC_GRADIENT_STEPS = int(os.environ.get("HIGHER_LOWER_TRAIN_SAC_GRADIENT_STEPS", "128"))
TRAIN_SAC_BATCH_SIZE = int(os.environ.get("HIGHER_LOWER_TRAIN_SAC_BATCH_SIZE", "256"))
TRAIN_SAC_ALPHA = float(os.environ.get("HIGHER_LOWER_TRAIN_SAC_ALPHA", "0.1"))
TRAIN_SAC_TARGET_UPDATE_TAU = float(os.environ.get("HIGHER_LOWER_TRAIN_SAC_TARGET_UPDATE_TAU", "0.01"))

_CACHED_STRATEGY: game.Strategy | None = None


def _resolve_policy_source() -> str | Path:
    if POLICY_SOURCE:
        return POLICY_SOURCE
    default_path = Path("artifacts") / "submission_policy.pkl"
    if default_path.exists():
        return default_path
    raise RuntimeError(
        "No policy weights configured. Set POLICY_SOURCE in submission_strategy.py or "
        "provide HIGHER_LOWER_POLICY_SOURCE at runtime."
    )


def _materialize_source(path_or_url: str | Path) -> Path:
    if isinstance(path_or_url, Path):
        return path_or_url
    if path_or_url.startswith(("http://", "https://")):
        return download_policy_weights(
            path_or_url,
            cache_dir=POLICY_CACHE_DIR,
            expected_sha256=POLICY_SHA256,
            timeout_seconds=POLICY_DOWNLOAD_TIMEOUT_SECONDS,
            force_download=POLICY_FORCE_DOWNLOAD,
        )
    return Path(path_or_url)


def _build_strategy() -> game.Strategy:
    policy_source = _resolve_policy_source()

    if POLICY_SOURCE_KIND == "policy":
        return RemotePolicyStrategy(
            policy_source,
            expected_sha256=POLICY_SHA256,
            cache_dir=POLICY_CACHE_DIR,
            timeout_seconds=POLICY_DOWNLOAD_TIMEOUT_SECONDS,
            force_download=POLICY_FORCE_DOWNLOAD,
            name=POLICY_NAME,
        )

    if POLICY_SOURCE_KIND != "checkpoint":
        raise ValueError(f"Unsupported HIGHER_LOWER_POLICY_SOURCE_KIND={POLICY_SOURCE_KIND!r}.")

    checkpoint_path = _materialize_source(policy_source)
    checkpoint_state = load_training_checkpoint(checkpoint_path)
    algorithm = str(checkpoint_state["training_config"].get("algorithm", "ppo")).lower()

    if TRAIN_EXTRA_GAMES > 0:
        params, _ = train_self_play(
            algorithm=algorithm,
            seed=int(checkpoint_state["training_config"].get("seed", 0)),
            total_games=int(checkpoint_state["games_played"]) + TRAIN_EXTRA_GAMES,
            episodes_per_batch=TRAIN_EPISODES_PER_BATCH,
            learning_rate=TRAIN_LEARNING_RATE,
            gamma=TRAIN_GAMMA,
            hidden_sizes=tuple(checkpoint_state["training_config"].get("hidden_sizes", [128, 128])),
            use_wandb=False,
            resume_from_checkpoint=checkpoint_path,
            ppo_epochs=TRAIN_PPO_EPOCHS,
            minibatch_size=TRAIN_MINIBATCH_SIZE,
            gae_lambda=TRAIN_GAE_LAMBDA,
            clip_epsilon=TRAIN_CLIP_EPSILON,
            value_coef=TRAIN_VALUE_COEF,
            entropy_coef=TRAIN_ENTROPY_COEF,
            replay_buffer_size=TRAIN_SAC_REPLAY_BUFFER_SIZE,
            warmup_games=TRAIN_SAC_WARMUP_GAMES,
            gradient_steps_per_update=TRAIN_SAC_GRADIENT_STEPS,
            batch_size=TRAIN_SAC_BATCH_SIZE,
            alpha=TRAIN_SAC_ALPHA,
            target_update_tau=TRAIN_SAC_TARGET_UPDATE_TAU,
        )
        return PolicyStrategy(params, name=POLICY_NAME)

    return PolicyStrategy(checkpoint_state["params"], name=POLICY_NAME)


def _get_strategy() -> game.Strategy:
    global _CACHED_STRATEGY
    if _CACHED_STRATEGY is None:
        _CACHED_STRATEGY = _build_strategy()
    return _CACHED_STRATEGY


class MyStrategy(game.Strategy):
    def evaluate(
        self,
        my_points: int,
        op_points: int,
        prev_cards_drawn: List[int],
        num_remaining_draws: int,
        rng: np.random.RandomState,
    ) -> int:
        return _get_strategy().evaluate(
            my_points=my_points,
            op_points=op_points,
            prev_cards_drawn=prev_cards_drawn,
            num_remaining_draws=num_remaining_draws,
            rng=rng,
        )

    def __str__(self) -> str:
        return POLICY_NAME


SubmissionStrategy = MyStrategy
