import hashlib
import pickle
import shutil
import tempfile
from pathlib import Path
from typing import List, Sequence, Tuple
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np

import deck
import game


POLICY_SOURCE = "https://raw.githubusercontent.com/fracapuano/sqif26/main/artifacts/submission_policy.pkl"
# Leave unset to always consume the latest weights published at POLICY_SOURCE.
# Set a concrete SHA256 string only if you want reproducible pinned inference.
POLICY_SHA256 = None
POLICY_FORMAT = "higher-lower-v2-policy"
POLICY_VERSION = 1
POLICY_CACHE_PATH = Path(tempfile.gettempdir()) / "sqif26_submission_policy.pkl"

MAX_POINTS = game.STARTING_POINTS * (2 ** (game.NUM_CARDS_TO_PLAY - 1))
ACTION_FRACTIONS = np.linspace(-1.0, 1.0, 81, dtype=np.float32)

_CACHED_PARAMS = None


def _sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_remote_source(path_or_url: str) -> bool:
    return urlparse(path_or_url).scheme in {"http", "https"}


def _materialize_policy(source: str, destination: Path, expected_sha256: str | None) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if not _is_remote_source(source):
        local_path = Path(source)
        if expected_sha256 is not None and _sha256sum(local_path) != expected_sha256:
            raise ValueError("Local policy weights hash mismatch.")
        return local_path

    # In unpinned mode we always refresh from the remote on each new process so
    # updated weights published to the same URL are picked up automatically.
    if destination.exists() and expected_sha256 is not None:
        if _sha256sum(destination) == expected_sha256:
            return destination

    request = Request(source, method="GET")
    with urlopen(request, timeout=60.0) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)

    if expected_sha256 is not None:
        actual_sha256 = _sha256sum(destination)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Downloaded policy weights hash mismatch: expected {expected_sha256}, got {actual_sha256}."
            )
    return destination


def _load_policy_params():
    global _CACHED_PARAMS
    if _CACHED_PARAMS is not None:
        return _CACHED_PARAMS

    local_path = _materialize_policy(POLICY_SOURCE, POLICY_CACHE_PATH, POLICY_SHA256)
    with local_path.open("rb") as handle:
        payload = pickle.load(handle)

    if payload.get("format") != POLICY_FORMAT:
        raise ValueError(f"Unsupported policy format: {payload.get('format')!r}.")
    if int(payload.get("version", -1)) != POLICY_VERSION:
        raise ValueError(f"Unsupported policy version: {payload.get('version')!r}.")

    params = payload["params"]
    for layer in params["layers"]:
        layer["w"] = np.asarray(layer["w"], dtype=np.float32)
        layer["b"] = np.asarray(layer["b"], dtype=np.float32)
    params["policy_head"]["w"] = np.asarray(params["policy_head"]["w"], dtype=np.float32)
    params["policy_head"]["b"] = np.asarray(params["policy_head"]["b"], dtype=np.float32)

    _CACHED_PARAMS = params
    return _CACHED_PARAMS


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
        params = _load_policy_params()
        obs = _encode_observation(
            my_points=my_points,
            op_points=op_points,
            prev_cards_drawn=prev_cards_drawn,
            num_remaining_draws=num_remaining_draws,
        )
        action_index = _greedy_action_index(params, obs)
        return _action_index_to_stake(action_index, my_points)

    def __str__(self) -> str:
        return "RL-ForTheWin"
