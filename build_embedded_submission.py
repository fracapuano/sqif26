from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path
from typing import Any

import numpy as np

from higher_lower_rl import load_policy_weights, load_training_checkpoint, save_policy_weights


SUBMISSION_TEMPLATE = """from __future__ import annotations

import base64
import io
from typing import List, Sequence, Tuple

import numpy as np

import deck
import game


MAX_POINTS = game.STARTING_POINTS * (2 ** (game.NUM_CARDS_TO_PLAY - 1))
ACTION_FRACTIONS = np.linspace(-1.0, 1.0, 81, dtype=np.float32)
_EMBEDDED_POLICY_NPZ_BASE64 = \"\"\"{embedded_npz}\"\"\"
_CACHED_PARAMS = None


def _load_embedded_params():
    global _CACHED_PARAMS
    if _CACHED_PARAMS is not None:
        return _CACHED_PARAMS

    archive = np.load(io.BytesIO(base64.b64decode(_EMBEDDED_POLICY_NPZ_BASE64.encode("ascii"))))
    num_layers = int(archive["num_layers"][0])
    params = {{
        "layers": [
            {{
                "w": archive[f"layers_{{layer_idx}}_w"].astype(np.float32),
                "b": archive[f"layers_{{layer_idx}}_b"].astype(np.float32),
            }}
            for layer_idx in range(num_layers)
        ],
        "policy_head": {{
            "w": archive["policy_head_w"].astype(np.float32),
            "b": archive["policy_head_b"].astype(np.float32),
        }},
    }}
    _CACHED_PARAMS = params
    return params


def _drawn_mask(cards_drawn: Sequence[int]) -> np.ndarray:
    mask = np.zeros(deck.NUM_CARDS, dtype=np.float32)
    for card in cards_drawn:
        mask[int(card)] = 1.0
    return mask


def _unseen_direction_probabilities(cards_drawn: Sequence[int]) -> Tuple[float, float]:
    last_card = int(cards_drawn[-1])
    drawn = set(int(card) for card in cards_drawn)
    unseen_higher = 0
    unseen_lower = 0

    for card in range(deck.NUM_CARDS):
        if card in drawn:
            continue
        if card > last_card:
            unseen_higher += 1
        elif card < last_card:
            unseen_lower += 1

    total_unseen = unseen_higher + unseen_lower
    if total_unseen == 0:
        return 0.5, 0.5

    return unseen_lower / total_unseen, unseen_higher / total_unseen


def _encode_observation(
    my_points: int,
    op_points: int,
    prev_cards_drawn: Sequence[int],
    num_remaining_draws: int,
) -> np.ndarray:
    last_card = int(prev_cards_drawn[-1])
    lower_prob, higher_prob = _unseen_direction_probabilities(prev_cards_drawn)
    features = np.array(
        [
            my_points / MAX_POINTS,
            op_points / MAX_POINTS,
            (my_points - op_points) / MAX_POINTS,
            num_remaining_draws / max(1, game.NUM_CARDS_TO_PLAY - 1),
            len(prev_cards_drawn) / game.NUM_CARDS_TO_PLAY,
            last_card / max(1, deck.NUM_CARDS - 1),
            lower_prob,
            higher_prob,
        ],
        dtype=np.float32,
    )
    return np.concatenate([features, _drawn_mask(prev_cards_drawn)], axis=0)


def _greedy_action_index(params, obs: np.ndarray) -> int:
    x = obs[None, :].astype(np.float32)
    for layer in params["layers"]:
        x = np.tanh(x @ layer["w"] + layer["b"])

    logits = x @ params["policy_head"]["w"] + params["policy_head"]["b"]
    return int(np.argmax(logits, axis=-1)[0])


def _action_index_to_stake(action_index: int, my_points: int) -> int:
    fraction = float(ACTION_FRACTIONS[int(action_index)])
    stake = int(np.round(fraction * my_points))
    return int(np.clip(stake, -my_points, my_points))


class MyStrategy(game.Strategy):
    def evaluate(
        self,
        my_points: int,
        op_points: int,
        prev_cards_drawn: List[int],
        num_remaining_draws: int,
        rng: np.random.RandomState,
    ) -> int:
        del rng
        params = _load_embedded_params()
        obs = _encode_observation(
            my_points=my_points,
            op_points=op_points,
            prev_cards_drawn=prev_cards_drawn,
            num_remaining_draws=num_remaining_draws,
        )
        action_index = _greedy_action_index(params, obs)
        return _action_index_to_stake(action_index, my_points)

    def __str__(self) -> str:
        return "{strategy_name}"
"""


def _to_numpy_params(source: str | Path):
    source = Path(source)
    if source.suffix == ".pkl":
        try:
            checkpoint_state = load_training_checkpoint(source)
        except Exception:
            return load_policy_weights(source)["params"]
        return checkpoint_state["params"]
    return load_policy_weights(source)["params"]


def _params_to_base64_npz(params: Any) -> str:
    arrays = {"num_layers": np.asarray([len(params["layers"])], dtype=np.int32)}
    for layer_idx, layer in enumerate(params["layers"]):
        arrays[f"layers_{layer_idx}_w"] = np.asarray(layer["w"], dtype=np.float32)
        arrays[f"layers_{layer_idx}_b"] = np.asarray(layer["b"], dtype=np.float32)
    arrays["policy_head_w"] = np.asarray(params["policy_head"]["w"], dtype=np.float32)
    arrays["policy_head_b"] = np.asarray(params["policy_head"]["b"], dtype=np.float32)

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def build_submission(source: str | Path, destination: str | Path, strategy_name: str) -> Path:
    params = _to_numpy_params(source)
    embedded_npz = _params_to_base64_npz(params)
    destination = Path(destination)
    destination.write_text(
        SUBMISSION_TEMPLATE.format(
            embedded_npz=embedded_npz,
            strategy_name=strategy_name,
        )
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a self-contained SQIF submission file with embedded policy weights.")
    parser.add_argument(
        "--source",
        default="artifacts/submission_policy.pkl",
        help="Path to an exported policy weights file or training checkpoint.",
    )
    parser.add_argument(
        "--dest",
        default="submission_embedded.py",
        help="Where to write the self-contained submission file.",
    )
    parser.add_argument(
        "--name",
        default="MyStrategy",
        help="Display name returned by MyStrategy.__str__.",
    )
    args = parser.parse_args()

    destination = build_submission(args.source, args.dest, args.name)
    print(destination)


if __name__ == "__main__":
    main()
