from __future__ import annotations

import hashlib
import importlib
import inspect
import pickle
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Sequence, Tuple
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import gymnasium as gym
from gymnasium import spaces
import numpy as np
try:
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError:
    jax = None
    jnp = None

import deck
import game


MAX_POINTS = game.STARTING_POINTS * (2 ** (game.NUM_CARDS_TO_PLAY - 1))
ACTION_FRACTIONS = np.linspace(-1.0, 1.0, 81, dtype=np.float32)
OBSERVATION_DIM = 8 + deck.NUM_CARDS
POLICY_WEIGHTS_FORMAT = "higher-lower-v2-policy"
POLICY_WEIGHTS_VERSION = 1
DEFAULT_POLICY_FILENAME = "policy_weights.pkl"
DEFAULT_POLICY_CACHE_DIR = Path.home() / ".cache" / "higher-lower-v2"


def _require_jax(feature: str) -> None:
    if jax is None or jnp is None:
        raise ImportError(
            f"jax is required for {feature}. Install jax to train PPO policies, "
            "or load exported policy weights for NumPy-only inference."
        )


def sha256sum(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_remote_source(path_or_url: str | Path) -> bool:
    return isinstance(path_or_url, str) and urlparse(path_or_url).scheme in {"http", "https"}


def _policy_download_path(weights_url: str, cache_dir: str | Path | None) -> Path:
    cache_root = Path(cache_dir) if cache_dir is not None else DEFAULT_POLICY_CACHE_DIR
    parsed = urlparse(weights_url)
    filename = Path(parsed.path).name or DEFAULT_POLICY_FILENAME
    url_hash = hashlib.sha256(weights_url.encode("utf-8")).hexdigest()[:16]
    return cache_root / f"{url_hash}-{filename}"


def download_policy_weights(
    weights_url: str,
    *,
    destination_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    expected_sha256: str | None = None,
    timeout_seconds: float = 60.0,
    force_download: bool = False,
) -> Path:
    destination = (
        Path(destination_path)
        if destination_path is not None
        else _policy_download_path(weights_url, cache_dir=cache_dir)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and not force_download:
        if expected_sha256 is None or sha256sum(destination) == expected_sha256:
            return destination

    request = Request(weights_url, method="GET")
    with urlopen(request, timeout=timeout_seconds) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)

    if expected_sha256 is not None:
        actual_sha256 = sha256sum(destination)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Downloaded policy weights hash mismatch for {weights_url}: "
                f"expected {expected_sha256}, got {actual_sha256}."
            )

    return destination


def upload_policy_weights(
    weights_path: str | Path,
    destination_url: str,
    *,
    method: str = "PUT",
    headers: Dict[str, str] | None = None,
    timeout_seconds: float = 60.0,
) -> Dict[str, Any]:
    weights_path = Path(weights_path)
    payload = weights_path.read_bytes()

    request = Request(destination_url, data=payload, method=method.upper())
    request.add_header("Content-Type", "application/octet-stream")
    request.add_header("Content-Length", str(len(payload)))
    for header_name, header_value in (headers or {}).items():
        request.add_header(header_name, header_value)

    with urlopen(request, timeout=timeout_seconds) as response:
        return {
            "url": destination_url,
            "method": method.upper(),
            "status": getattr(response, "status", None),
            "reason": getattr(response, "reason", ""),
        }


def _tree_to_numpy(tree):
    if jax is not None:
        return jax.tree_util.tree_map(lambda x: np.asarray(x), tree)
    return _tree_map_numpy(tree)


def _tree_map_numpy(tree):
    if isinstance(tree, dict):
        return {key: _tree_map_numpy(value) for key, value in tree.items()}
    if isinstance(tree, (list, tuple)):
        values = [_tree_map_numpy(value) for value in tree]
        return type(tree)(values)
    return np.asarray(tree)


def _tree_to_jax(tree):
    _require_jax("loading JAX policy weights")
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x), tree)


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


def encode_observation(
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


def action_index_to_stake(action_index: int, my_points: int) -> int:
    fraction = float(ACTION_FRACTIONS[int(action_index)])
    stake = int(np.round(fraction * my_points))
    return int(np.clip(stake, -my_points, my_points))


def _init_linear_layer(key: jax.Array, in_dim: int, out_dim: int) -> Dict[str, jax.Array]:
    _require_jax("initializing network layers")
    return {
        "w": jax.random.normal(key, (in_dim, out_dim), dtype=jnp.float32) / jnp.sqrt(float(in_dim)),
        "b": jnp.zeros((out_dim,), dtype=jnp.float32),
    }


def _mlp_features(params, obs: jax.Array) -> jax.Array:
    x = obs
    for layer in params["layers"]:
        x = jnp.tanh(x @ layer["w"] + layer["b"])
    return x


def _mlp_features_numpy(params, obs: np.ndarray) -> np.ndarray:
    x = np.asarray(obs, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    for layer in params["layers"]:
        x = np.tanh(x @ np.asarray(layer["w"], dtype=np.float32) + np.asarray(layer["b"], dtype=np.float32))
    return x


def _policy_head_key(params: Dict[str, Any]) -> str:
    if "policy_head" in params:
        return "policy_head"
    if "actor_head" in params:
        return "actor_head"
    raise ValueError("Unsupported policy parameter structure: expected `policy_head` or `actor_head`.")


def extract_policy_params(params):
    if isinstance(params, dict) and "actor" in params:
        return params["actor"]
    return params


def infer_policy_algorithm(params) -> str | None:
    if isinstance(params, dict) and "actor" in params:
        return "sac"
    policy_params = extract_policy_params(params)
    if isinstance(policy_params, dict) and "value_head" in policy_params:
        return "ppo"
    if isinstance(policy_params, dict) and "actor_head" in policy_params:
        return "sac"
    return None


def policy_logits(params, obs: jax.Array) -> jax.Array:
    _require_jax("running policy inference")
    policy_params = extract_policy_params(params)
    x = _mlp_features(policy_params, obs)
    head = policy_params[_policy_head_key(policy_params)]
    return x @ head["w"] + head["b"]


def policy_logits_numpy(params, obs: np.ndarray) -> np.ndarray:
    policy_params = extract_policy_params(params)
    x = _mlp_features_numpy(policy_params, obs)
    head = policy_params[_policy_head_key(policy_params)]
    return x @ np.asarray(head["w"], dtype=np.float32) + np.asarray(head["b"], dtype=np.float32)


def discrete_policy_distribution(params, obs: jax.Array) -> Tuple[jax.Array, jax.Array]:
    logits = policy_logits(params, obs)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probs = jnp.exp(log_probs)
    return probs, log_probs


def init_actor_params(
    key: jax.Array,
    input_dim: int = OBSERVATION_DIM,
    hidden_sizes: Sequence[int] = (128, 128),
    num_actions: int = len(ACTION_FRACTIONS),
):
    _require_jax("initializing actor parameters")
    keys = iter(jax.random.split(key, len(hidden_sizes) + 1))
    layers = []
    in_dim = input_dim
    for out_dim in hidden_sizes:
        layers.append(_init_linear_layer(next(keys), in_dim, out_dim))
        in_dim = out_dim
    return {"layers": layers, "actor_head": _init_linear_layer(next(keys), in_dim, num_actions)}


def init_q_params(
    key: jax.Array,
    input_dim: int = OBSERVATION_DIM,
    hidden_sizes: Sequence[int] = (128, 128),
    num_actions: int = len(ACTION_FRACTIONS),
):
    _require_jax("initializing critic parameters")
    keys = iter(jax.random.split(key, len(hidden_sizes) + 1))
    layers = []
    in_dim = input_dim
    for out_dim in hidden_sizes:
        layers.append(_init_linear_layer(next(keys), in_dim, out_dim))
        in_dim = out_dim
    return {"layers": layers, "q_head": _init_linear_layer(next(keys), in_dim, num_actions)}


class HigherLowerSelfPlayEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None):
        super().__init__()
        self.action_space = spaces.MultiDiscrete([len(ACTION_FRACTIONS), len(ACTION_FRACTIONS)])
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2, OBSERVATION_DIM),
            dtype=np.float32,
        )
        self._rng = np.random.RandomState(seed)
        self.points = np.array([game.STARTING_POINTS, game.STARTING_POINTS], dtype=np.int32)
        self.cards_drawn: List[int] = []
        self.num_remaining_draws = 0

    def _get_obs(self) -> np.ndarray:
        return np.stack(
            [
                encode_observation(
                    my_points=int(self.points[0]),
                    op_points=int(self.points[1]),
                    prev_cards_drawn=self.cards_drawn,
                    num_remaining_draws=self.num_remaining_draws,
                ),
                encode_observation(
                    my_points=int(self.points[1]),
                    op_points=int(self.points[0]),
                    prev_cards_drawn=self.cards_drawn,
                    num_remaining_draws=self.num_remaining_draws,
                ),
            ],
            axis=0,
        ).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.RandomState(seed)

        self.points = np.array([game.STARTING_POINTS, game.STARTING_POINTS], dtype=np.int32)
        self.cards_drawn = [deck.random_card(self._rng)]
        self.num_remaining_draws = game.NUM_CARDS_TO_PLAY - 1
        return self._get_obs(), {"first_card": int(self.cards_drawn[0])}

    def step(self, action: Iterable[int]):
        action = np.asarray(action, dtype=np.int32)
        if action.shape != (2,):
            raise ValueError(f"Expected joint action with shape (2,), got {action.shape}")

        old_points = self.points.copy()
        prev_card = int(self.cards_drawn[-1])
        stakes = np.array(
            [action_index_to_stake(action[0], int(self.points[0])), action_index_to_stake(action[1], int(self.points[1]))],
            dtype=np.int32,
        )
        next_card = deck.draw_card(self.cards_drawn, self._rng)
        higher = next_card > prev_card

        if higher:
            self.points = self.points + stakes
        else:
            self.points = self.points - stakes

        self.cards_drawn.append(int(next_card))
        self.num_remaining_draws -= 1

        terminated = False
        match_result = 0
        if self.points[1] == 0:
            terminated = True
            match_result = 0 if self.points[0] == 0 else 1
        elif self.points[0] == 0:
            terminated = True
            match_result = -1
        elif self.num_remaining_draws == 0:
            terminated = True
            if self.points[0] > self.points[1]:
                match_result = 1
            elif self.points[0] < self.points[1]:
                match_result = -1

        rewards = np.log1p(self.points.astype(np.float32)) - np.log1p(old_points.astype(np.float32))
        if terminated:
            rewards = rewards + 0.1 * np.array([match_result, -match_result], dtype=np.float32)

        info = {
            "higher": bool(higher),
            "next_card": int(next_card),
            "stakes": stakes.copy(),
            "points": self.points.copy(),
            "match_result": int(match_result),
        }
        return self._get_obs(), rewards.astype(np.float32), terminated, False, info


def init_policy_params(
    key: jax.Array,
    input_dim: int = OBSERVATION_DIM,
    hidden_sizes: Sequence[int] = (128, 128),
    num_actions: int = len(ACTION_FRACTIONS),
):
    _require_jax("initializing PPO policy parameters")
    keys = iter(jax.random.split(key, len(hidden_sizes) + 2))
    layers = []
    in_dim = input_dim
    for out_dim in hidden_sizes:
        layers.append(_init_linear_layer(next(keys), in_dim, out_dim))
        in_dim = out_dim
    return {
        "layers": layers,
        "policy_head": _init_linear_layer(next(keys), in_dim, num_actions),
        "value_head": _init_linear_layer(next(keys), in_dim, 1),
    }


def policy_value(params, obs: jax.Array) -> Tuple[jax.Array, jax.Array]:
    _require_jax("running JAX PPO policy inference")
    x = _mlp_features(params, obs)
    logits = x @ params["policy_head"]["w"] + params["policy_head"]["b"]
    values = jnp.squeeze(x @ params["value_head"]["w"] + params["value_head"]["b"], axis=-1)
    return logits, values


def policy_value_numpy(params, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = _mlp_features_numpy(params, obs)
    logits = x @ np.asarray(params["policy_head"]["w"], dtype=np.float32) + np.asarray(
        params["policy_head"]["b"], dtype=np.float32
    )
    values = np.squeeze(
        x @ np.asarray(params["value_head"]["w"], dtype=np.float32) + np.asarray(params["value_head"]["b"], dtype=np.float32),
        axis=-1,
    )
    return logits, values


def q_values(params, obs: jax.Array) -> jax.Array:
    _require_jax("running critic inference")
    x = _mlp_features(params, obs)
    return x @ params["q_head"]["w"] + params["q_head"]["b"]


def categorical_log_probs(logits: jax.Array, actions: jax.Array) -> jax.Array:
    _require_jax("computing PPO log probabilities")
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(log_probs, actions[..., None], axis=-1).squeeze(axis=-1)


def categorical_entropy(logits: jax.Array) -> jax.Array:
    _require_jax("computing PPO entropy")
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probs = jnp.exp(log_probs)
    return -jnp.sum(probs * log_probs, axis=-1)


class AdamState(NamedTuple):
    step: int
    m: object
    v: object


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int = OBSERVATION_DIM):
        if capacity <= 0:
            raise ValueError(f"Replay buffer capacity must be positive, got {capacity}.")
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity,), dtype=np.int32)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity,), dtype=np.float32)
        self.size = 0
        self.position = 0

    def add(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, next_obs: np.ndarray, dones: np.ndarray) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.int32)
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.float32)

        if obs.ndim == 1:
            obs = obs[None, :]
            next_obs = next_obs[None, :]
            actions = actions[None]
            rewards = rewards[None]
            dones = dones[None]

        batch_size = int(obs.shape[0])
        for idx in range(batch_size):
            self.obs[self.position] = obs[idx]
            self.actions[self.position] = actions[idx]
            self.rewards[self.position] = rewards[idx]
            self.next_obs[self.position] = next_obs[idx]
            self.dones[self.position] = dones[idx]
            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)
        return batch_size

    def sample(self, batch_size: int, rng: np.random.RandomState) -> Dict[str, np.ndarray]:
        if self.size < batch_size:
            raise ValueError(f"Cannot sample {batch_size} transitions from replay buffer size {self.size}.")
        indices = rng.randint(0, self.size, size=batch_size)
        return {
            "obs": self.obs[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_obs": self.next_obs[indices],
            "dones": self.dones[indices],
        }


def adam_init(params) -> AdamState:
    _require_jax("initializing the PPO optimizer")
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    return AdamState(step=0, m=zeros, v=zeros)


def adam_update(params, grads, state: AdamState, learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
    _require_jax("updating the PPO optimizer")
    step = state.step + 1
    m = jax.tree_util.tree_map(lambda m_prev, g: beta1 * m_prev + (1.0 - beta1) * g, state.m, grads)
    v = jax.tree_util.tree_map(lambda v_prev, g: beta2 * v_prev + (1.0 - beta2) * (g * g), state.v, grads)
    m_hat = jax.tree_util.tree_map(lambda x: x / (1.0 - beta1**step), m)
    v_hat = jax.tree_util.tree_map(lambda x: x / (1.0 - beta2**step), v)
    updated_params = jax.tree_util.tree_map(
        lambda p, m_term, v_term: p - learning_rate * m_term / (jnp.sqrt(v_term) + eps),
        params,
        m_hat,
        v_hat,
    )
    return updated_params, AdamState(step=step, m=m, v=v)


def _serialize_optimizer_state(state):
    if isinstance(state, AdamState):
        return {
            "__type__": "adam",
            "step": int(state.step),
            "m": _tree_to_numpy(state.m),
            "v": _tree_to_numpy(state.v),
        }
    if isinstance(state, dict):
        return {key: _serialize_optimizer_state(value) for key, value in state.items()}
    raise TypeError(f"Unsupported optimizer state type: {type(state)!r}")


def _restore_optimizer_state(state):
    if isinstance(state, dict) and state.get("__type__") == "adam":
        return AdamState(
            step=int(state["step"]),
            m=_tree_to_jax(state["m"]),
            v=_tree_to_jax(state["v"]),
        )
    if isinstance(state, dict):
        return {key: _restore_optimizer_state(value) for key, value in state.items()}
    raise TypeError(f"Unsupported serialized optimizer state type: {type(state)!r}")


def soft_update_params(target_params, source_params, tau: float):
    _require_jax("updating target network parameters")
    return jax.tree_util.tree_map(lambda target, source: (1.0 - tau) * target + tau * source, target_params, source_params)


def compute_gae(rewards: np.ndarray, values: np.ndarray, gamma: float, gae_lambda: float) -> Tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = 0.0
    next_value = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        delta = rewards[idx] + gamma * next_value - values[idx]
        last_advantage = delta + gamma * gae_lambda * last_advantage
        advantages[idx] = last_advantage
        next_value = values[idx]
    returns = advantages + values
    return advantages, returns.astype(np.float32)


def sample_actions(params, obs: np.ndarray, key: jax.Array):
    _require_jax("sampling PPO actions")
    obs_jax = jnp.asarray(obs, dtype=jnp.float32)
    logits, values = policy_value(params, obs_jax)
    key, sample_key = jax.random.split(key)
    actions = jax.random.categorical(sample_key, logits, axis=-1)
    log_probs = categorical_log_probs(logits, actions)
    return (
        np.asarray(actions, dtype=np.int32),
        np.asarray(log_probs, dtype=np.float32),
        np.asarray(values, dtype=np.float32),
        key,
    )


def greedy_actions(params, obs: np.ndarray) -> np.ndarray:
    _require_jax("running greedy policy actions in JAX")
    logits = policy_logits(params, jnp.asarray(obs, dtype=jnp.float32))
    return np.asarray(jnp.argmax(logits, axis=-1), dtype=np.int32)


def greedy_actions_numpy(params, obs: np.ndarray) -> np.ndarray:
    logits = policy_logits_numpy(params, obs)
    return np.asarray(np.argmax(logits, axis=-1), dtype=np.int32)


def collect_self_play_batch(
    params,
    key: jax.Array,
    episodes_per_batch: int,
    seed: int,
    gamma: float,
    gae_lambda: float,
) -> Tuple[Dict[str, np.ndarray], jax.Array, Dict[str, float]]:
    all_obs: List[np.ndarray] = []
    all_actions: List[int] = []
    all_log_probs: List[float] = []
    all_values: List[float] = []
    all_advantages: List[float] = []
    all_returns: List[float] = []

    match_results = []
    final_points = []
    episode_lengths = []

    for episode_idx in range(episodes_per_batch):
        env = HigherLowerSelfPlayEnv(seed=seed + episode_idx)
        obs, _ = env.reset()

        episode_buffers = [
            {"obs": [], "actions": [], "log_probs": [], "values": [], "rewards": []},
            {"obs": [], "actions": [], "log_probs": [], "values": [], "rewards": []},
        ]

        terminated = False
        final_info = None
        while not terminated:
            actions, log_probs, values, key = sample_actions(params, obs, key)
            next_obs, rewards, terminated, _, info = env.step(actions)

            for player_idx in range(2):
                episode_buffers[player_idx]["obs"].append(obs[player_idx])
                episode_buffers[player_idx]["actions"].append(int(actions[player_idx]))
                episode_buffers[player_idx]["log_probs"].append(float(log_probs[player_idx]))
                episode_buffers[player_idx]["values"].append(float(values[player_idx]))
                episode_buffers[player_idx]["rewards"].append(float(rewards[player_idx]))

            obs = next_obs
            final_info = info

        match_results.append(int(final_info["match_result"]))
        final_points.append(np.asarray(final_info["points"], dtype=np.float32))
        episode_lengths.append(len(episode_buffers[0]["rewards"]))

        for player_idx in range(2):
            rewards = np.asarray(episode_buffers[player_idx]["rewards"], dtype=np.float32)
            values = np.asarray(episode_buffers[player_idx]["values"], dtype=np.float32)
            advantages, returns = compute_gae(rewards, values, gamma=gamma, gae_lambda=gae_lambda)

            all_obs.extend(episode_buffers[player_idx]["obs"])
            all_actions.extend(episode_buffers[player_idx]["actions"])
            all_log_probs.extend(episode_buffers[player_idx]["log_probs"])
            all_values.extend(values.tolist())
            all_advantages.extend(advantages.tolist())
            all_returns.extend(returns.tolist())

    batch = {
        "obs": np.asarray(all_obs, dtype=np.float32),
        "actions": np.asarray(all_actions, dtype=np.int32),
        "log_probs": np.asarray(all_log_probs, dtype=np.float32),
        "values": np.asarray(all_values, dtype=np.float32),
        "advantages": np.asarray(all_advantages, dtype=np.float32),
        "returns": np.asarray(all_returns, dtype=np.float32),
    }
    stats = {
        "mean_match_result": float(np.mean(match_results)),
        "mean_final_points": float(np.mean(np.asarray(final_points, dtype=np.float32))),
        "mean_episode_length": float(np.mean(episode_lengths)),
    }
    return batch, key, stats


def save_training_checkpoint(
    checkpoint_path: str | Path,
    *,
    params,
    optimizer_state,
    rng: np.random.RandomState,
    key: jax.Array,
    history: Dict[str, List[float]],
    completed_updates: int,
    games_played: int,
    cumulative_agent_steps: int,
    next_env_seed: int,
    training_config: Dict[str, Any],
    extra_state: Dict[str, Any] | None = None,
):
    _require_jax("saving a training checkpoint")
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "params": _tree_to_numpy(params),
        "optimizer_state": _serialize_optimizer_state(optimizer_state),
        "rng_state": rng.get_state(),
        "jax_key": np.asarray(key),
        "history": history,
        "completed_updates": int(completed_updates),
        "games_played": int(games_played),
        "cumulative_agent_steps": int(cumulative_agent_steps),
        "next_env_seed": int(next_env_seed),
        "training_config": training_config,
        "extra_state": _tree_to_numpy(extra_state) if extra_state is not None else None,
    }

    with checkpoint_path.open("wb") as handle:
        pickle.dump(payload, handle)


def load_training_checkpoint(checkpoint_path: str | Path) -> Dict[str, Any]:
    _require_jax("loading a training checkpoint")
    checkpoint_path = Path(checkpoint_path)
    with checkpoint_path.open("rb") as handle:
        payload = pickle.load(handle)

    rng = np.random.RandomState()
    rng.set_state(payload["rng_state"])

    return {
        "params": _tree_to_jax(payload["params"]),
        "optimizer_state": _restore_optimizer_state(payload["optimizer_state"]),
        "rng": rng,
        "key": jnp.asarray(payload["jax_key"]),
        "history": payload["history"],
        "completed_updates": int(payload.get("completed_updates", 0)),
        "games_played": int(payload.get("games_played", 0)),
        "cumulative_agent_steps": int(payload.get("cumulative_agent_steps", 0)),
        "next_env_seed": int(payload.get("next_env_seed", 0)),
        "training_config": payload.get("training_config", {}),
        "extra_state": _tree_to_jax(payload["extra_state"]) if payload.get("extra_state") is not None else {},
    }


def save_policy_weights(
    weights_path: str | Path,
    params,
    *,
    algorithm: str | None = None,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    weights_path = Path(weights_path)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    policy_params = extract_policy_params(params)
    resolved_algorithm = algorithm or infer_policy_algorithm(params)
    metadata_payload = dict(metadata or {})
    if resolved_algorithm is not None:
        metadata_payload.setdefault("algorithm", resolved_algorithm)

    payload = {
        "format": POLICY_WEIGHTS_FORMAT,
        "version": POLICY_WEIGHTS_VERSION,
        "params": _tree_to_numpy(policy_params),
        "metadata": metadata_payload,
    }
    with weights_path.open("wb") as handle:
        pickle.dump(payload, handle)

    return {
        "path": weights_path,
        "sha256": sha256sum(weights_path),
        "metadata": payload["metadata"],
    }


def load_policy_weights(
    weights_source: str | Path,
    *,
    expected_sha256: str | None = None,
    cache_dir: str | Path | None = None,
    timeout_seconds: float = 60.0,
    force_download: bool = False,
    as_jax: bool = False,
) -> Dict[str, Any]:
    local_path = (
        download_policy_weights(
            weights_source,
            cache_dir=cache_dir,
            expected_sha256=expected_sha256,
            timeout_seconds=timeout_seconds,
            force_download=force_download,
        )
        if _is_remote_source(weights_source)
        else Path(weights_source)
    )

    with local_path.open("rb") as handle:
        payload = pickle.load(handle)

    if payload.get("format") != POLICY_WEIGHTS_FORMAT:
        raise ValueError(
            f"Unsupported policy weights format in {local_path}: "
            f"expected {POLICY_WEIGHTS_FORMAT}, got {payload.get('format')}."
        )
    if int(payload.get("version", -1)) != POLICY_WEIGHTS_VERSION:
        raise ValueError(
            f"Unsupported policy weights version in {local_path}: "
            f"expected {POLICY_WEIGHTS_VERSION}, got {payload.get('version')}."
        )

    raw_params = payload.get("policy_params", payload["params"])
    params = _tree_to_jax(raw_params) if as_jax else _tree_to_numpy(raw_params)
    return {
        "params": params,
        "metadata": dict(payload.get("metadata", {})),
        "path": local_path,
        "sha256": sha256sum(local_path),
    }


def _ppo_loss(
    params,
    obs: jax.Array,
    actions: jax.Array,
    old_log_probs: jax.Array,
    advantages: jax.Array,
    returns: jax.Array,
    clip_epsilon: float,
    value_coef: float,
    entropy_coef: float,
):
    _require_jax("computing the PPO loss")
    logits, values = policy_value(params, obs)
    log_probs = categorical_log_probs(logits, actions)
    ratios = jnp.exp(log_probs - old_log_probs)

    normalized_advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
    unclipped = ratios * normalized_advantages
    clipped = jnp.clip(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * normalized_advantages
    policy_loss = -jnp.mean(jnp.minimum(unclipped, clipped))
    value_loss = 0.5 * jnp.mean((returns - values) ** 2)
    entropy = jnp.mean(categorical_entropy(logits))
    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    metrics = {
        "loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
    }
    return total_loss, metrics


def _ppo_update_minibatch_impl(
    params,
    optimizer_state: AdamState,
    obs: jax.Array,
    actions: jax.Array,
    old_log_probs: jax.Array,
    advantages: jax.Array,
    returns: jax.Array,
    learning_rate: float,
    clip_epsilon: float,
    value_coef: float,
    entropy_coef: float,
):
    (_, metrics), grads = jax.value_and_grad(_ppo_loss, has_aux=True)(
        params,
        obs,
        actions,
        old_log_probs,
        advantages,
        returns,
        clip_epsilon,
        value_coef,
        entropy_coef,
    )
    updated_params, updated_optimizer_state = adam_update(params, grads, optimizer_state, learning_rate)
    return updated_params, updated_optimizer_state, metrics


if jax is not None:
    ppo_update_minibatch = jax.jit(_ppo_update_minibatch_impl)
else:
    def ppo_update_minibatch(*args, **kwargs):
        del args, kwargs
        _require_jax("running PPO minibatch updates")


def sample_policy_actions(params, obs: np.ndarray, key: jax.Array):
    _require_jax("sampling policy actions")
    logits = policy_logits(params, jnp.asarray(obs, dtype=jnp.float32))
    key, sample_key = jax.random.split(key)
    actions = jax.random.categorical(sample_key, logits, axis=-1)
    log_probs = categorical_log_probs(logits, actions)
    return np.asarray(actions, dtype=np.int32), np.asarray(log_probs, dtype=np.float32), key


def collect_self_play_replay(
    actor_params,
    key: jax.Array,
    episodes_per_batch: int,
    seed: int,
    replay_buffer: ReplayBuffer,
) -> Tuple[jax.Array, Dict[str, float]]:
    match_results = []
    final_points = []
    episode_lengths = []
    sampled_log_probs = []
    transitions_collected = 0

    for episode_idx in range(episodes_per_batch):
        env = HigherLowerSelfPlayEnv(seed=seed + episode_idx)
        obs, _ = env.reset()

        terminated = False
        final_info = None
        episode_length = 0
        while not terminated:
            actions, log_probs, key = sample_policy_actions(actor_params, obs, key)
            next_obs, rewards, terminated, _, info = env.step(actions)
            transitions_collected += replay_buffer.add(
                obs,
                actions,
                rewards,
                next_obs,
                np.full((2,), float(terminated), dtype=np.float32),
            )
            sampled_log_probs.extend(log_probs.tolist())
            obs = next_obs
            final_info = info
            episode_length += 1

        match_results.append(int(final_info["match_result"]))
        final_points.append(np.asarray(final_info["points"], dtype=np.float32))
        episode_lengths.append(episode_length)

    stats = {
        "mean_match_result": float(np.mean(match_results)),
        "mean_final_points": float(np.mean(np.asarray(final_points, dtype=np.float32))),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_action_log_prob": float(np.mean(sampled_log_probs)) if sampled_log_probs else 0.0,
        "transitions_collected": float(transitions_collected),
    }
    return key, stats


def _sac_critic_loss(
    critic_1_params,
    critic_2_params,
    actor_params,
    target_critic_1_params,
    target_critic_2_params,
    obs: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    next_obs: jax.Array,
    dones: jax.Array,
    gamma: float,
    alpha: float,
):
    next_probs, next_log_probs = discrete_policy_distribution(actor_params, next_obs)
    next_q1 = q_values(target_critic_1_params, next_obs)
    next_q2 = q_values(target_critic_2_params, next_obs)
    next_min_q = jnp.minimum(next_q1, next_q2)
    next_v = jnp.sum(next_probs * (next_min_q - alpha * next_log_probs), axis=-1)
    targets = rewards + gamma * (1.0 - dones) * next_v

    q1 = q_values(critic_1_params, obs)
    q2 = q_values(critic_2_params, obs)
    q1_selected = jnp.take_along_axis(q1, actions[..., None], axis=-1).squeeze(axis=-1)
    q2_selected = jnp.take_along_axis(q2, actions[..., None], axis=-1).squeeze(axis=-1)
    td_error_1 = q1_selected - targets
    td_error_2 = q2_selected - targets
    critic_loss = 0.5 * (jnp.mean(td_error_1**2) + jnp.mean(td_error_2**2))
    metrics = {
        "critic_loss": critic_loss,
        "target_q": jnp.mean(targets),
        "mean_q": 0.5 * (jnp.mean(q1_selected) + jnp.mean(q2_selected)),
    }
    return critic_loss, metrics


def _sac_actor_loss(actor_params, critic_1_params, critic_2_params, obs: jax.Array, alpha: float):
    probs, log_probs = discrete_policy_distribution(actor_params, obs)
    q1 = q_values(critic_1_params, obs)
    q2 = q_values(critic_2_params, obs)
    min_q = jnp.minimum(q1, q2)
    entropy = -jnp.sum(probs * log_probs, axis=-1)
    actor_loss = jnp.mean(jnp.sum(probs * (alpha * log_probs - min_q), axis=-1))
    metrics = {
        "actor_loss": actor_loss,
        "entropy": jnp.mean(entropy),
    }
    return actor_loss, metrics


def _sac_update_step_impl(
    params,
    target_params,
    optimizer_state,
    obs: jax.Array,
    actions: jax.Array,
    rewards: jax.Array,
    next_obs: jax.Array,
    dones: jax.Array,
    actor_learning_rate: float,
    critic_learning_rate: float,
    gamma: float,
    alpha: float,
    target_update_tau: float,
):
    (_, critic_metrics), critic_grads = jax.value_and_grad(_sac_critic_loss, argnums=(0, 1), has_aux=True)(
        params["critic_1"],
        params["critic_2"],
        params["actor"],
        target_params["critic_1"],
        target_params["critic_2"],
        obs,
        actions,
        rewards,
        next_obs,
        dones,
        gamma,
        alpha,
    )

    critic_1_params, critic_1_optimizer_state = adam_update(
        params["critic_1"],
        critic_grads[0],
        optimizer_state["critic_1"],
        critic_learning_rate,
    )
    critic_2_params, critic_2_optimizer_state = adam_update(
        params["critic_2"],
        critic_grads[1],
        optimizer_state["critic_2"],
        critic_learning_rate,
    )

    (_, actor_metrics), actor_grads = jax.value_and_grad(_sac_actor_loss, has_aux=True)(
        params["actor"],
        critic_1_params,
        critic_2_params,
        obs,
        alpha,
    )
    actor_params, actor_optimizer_state = adam_update(
        params["actor"],
        actor_grads,
        optimizer_state["actor"],
        actor_learning_rate,
    )

    updated_params = {
        "actor": actor_params,
        "critic_1": critic_1_params,
        "critic_2": critic_2_params,
    }
    updated_target_params = {
        "critic_1": soft_update_params(target_params["critic_1"], critic_1_params, target_update_tau),
        "critic_2": soft_update_params(target_params["critic_2"], critic_2_params, target_update_tau),
    }
    updated_optimizer_state = {
        "actor": actor_optimizer_state,
        "critic_1": critic_1_optimizer_state,
        "critic_2": critic_2_optimizer_state,
    }
    metrics = {
        "critic_loss": critic_metrics["critic_loss"],
        "actor_loss": actor_metrics["actor_loss"],
        "entropy": actor_metrics["entropy"],
        "target_q": critic_metrics["target_q"],
        "mean_q": critic_metrics["mean_q"],
    }
    return updated_params, updated_target_params, updated_optimizer_state, metrics


if jax is not None:
    sac_update_step = jax.jit(_sac_update_step_impl)
else:
    def sac_update_step(*args, **kwargs):
        del args, kwargs
        _require_jax("running SAC updates")


def train_self_play_ppo(
    seed: int = 0,
    num_updates: int | None = 120,
    total_games: int | None = None,
    episodes_per_batch: int = 512,
    ppo_epochs: int = 6,
    minibatch_size: int = 512,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_epsilon: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    hidden_sizes: Sequence[int] = (128, 128),
    use_wandb: bool = False,
    wandb_project: str = "higher-lower-v2-ppo",
    wandb_run_name: str | None = None,
    wandb_mode: str | None = None,
    wandb_config: Dict[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
    resume_from_checkpoint: str | Path | None = None,
    checkpoint_every_updates: int | None = None,
    checkpoint_every_games: int | None = None,
    show_progress: bool = True,
):
    _require_jax("training a PPO strategy")
    if total_games is None and num_updates is None:
        raise ValueError("Provide either total_games or num_updates.")
    if total_games is not None and total_games < 0:
        raise ValueError(f"total_games must be non-negative, got {total_games}.")
    if num_updates is not None and num_updates < 0:
        raise ValueError(f"num_updates must be non-negative, got {num_updates}.")
    if episodes_per_batch <= 0:
        raise ValueError(f"episodes_per_batch must be positive, got {episodes_per_batch}.")
    if checkpoint_every_updates is not None and checkpoint_every_updates <= 0:
        raise ValueError(f"checkpoint_every_updates must be positive, got {checkpoint_every_updates}.")
    if checkpoint_every_games is not None and checkpoint_every_games <= 0:
        raise ValueError(f"checkpoint_every_games must be positive, got {checkpoint_every_games}.")
    if (checkpoint_every_updates is not None or checkpoint_every_games is not None) and checkpoint_path is None:
        raise ValueError("Set checkpoint_path when requesting periodic checkpoint saves.")

    if resume_from_checkpoint is not None:
        resumed_state = load_training_checkpoint(resume_from_checkpoint)
        params = resumed_state["params"]
        optimizer_state = resumed_state["optimizer_state"]
        rng = resumed_state["rng"]
        key = resumed_state["key"]
        completed_updates = resumed_state["completed_updates"]
        games_played = resumed_state["games_played"]
        cumulative_agent_steps = resumed_state["cumulative_agent_steps"]
        next_env_seed = resumed_state["next_env_seed"]
        history = resumed_state["history"]
    else:
        rng = np.random.RandomState(seed)
        key = jax.random.PRNGKey(seed)
        key, init_key = jax.random.split(key)
        params = init_policy_params(init_key, hidden_sizes=hidden_sizes)
        optimizer_state = adam_init(params)
        completed_updates = 0
        games_played = 0
        cumulative_agent_steps = 0
        next_env_seed = seed * 10000
        history = {
            "loss": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "mean_match_result": [],
            "mean_final_points": [],
            "mean_episode_length": [],
            "games_played": [],
            "agent_steps": [],
        }

    for key_name in (
        "loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "mean_match_result",
        "mean_final_points",
        "mean_episode_length",
        "games_played",
        "agent_steps",
    ):
        history.setdefault(key_name, [])

    training_config = {
        "algorithm": "ppo",
        "seed": seed,
        "num_updates": num_updates,
        "total_games": total_games,
        "episodes_per_batch": episodes_per_batch,
        "ppo_epochs": ppo_epochs,
        "minibatch_size": minibatch_size,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_epsilon": clip_epsilon,
        "value_coef": value_coef,
        "entropy_coef": entropy_coef,
        "hidden_sizes": list(hidden_sizes),
        "use_wandb": use_wandb,
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
        "wandb_mode": wandb_mode,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "resume_from_checkpoint": str(resume_from_checkpoint) if resume_from_checkpoint is not None else None,
        "checkpoint_every_updates": checkpoint_every_updates,
        "checkpoint_every_games": checkpoint_every_games,
        "show_progress": show_progress,
    }

    wandb_run = None
    if use_wandb:
        try:
            wandb = importlib.import_module("wandb")
            if not hasattr(wandb, "init"):
                raise AttributeError(
                    f"Imported `wandb` from {getattr(wandb, '__file__', 'unknown location')}, "
                    "but it does not expose `wandb.init`. Restart the kernel and re-import the real Weights & Biases package."
                )

            run_dir = Path.cwd() / "wandb"
            run_dir.mkdir(parents=True, exist_ok=True)

            config = {
                "algorithm": "ppo",
                "seed": seed,
                "num_updates": num_updates,
                "total_games": total_games,
                "episodes_per_batch": episodes_per_batch,
                "ppo_epochs": ppo_epochs,
                "minibatch_size": minibatch_size,
                "learning_rate": learning_rate,
                "gamma": gamma,
                "gae_lambda": gae_lambda,
                "clip_epsilon": clip_epsilon,
                "value_coef": value_coef,
                "entropy_coef": entropy_coef,
                "hidden_sizes": list(hidden_sizes),
                "num_actions": len(ACTION_FRACTIONS),
                "observation_dim": OBSERVATION_DIM,
                "resumed_updates": completed_updates,
                "resumed_games": games_played,
            }
            if wandb_config:
                config.update(wandb_config)
            wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                mode=wandb_mode,
                config=config,
                dir=str(run_dir),
                reinit=True,
            )
        except Exception as exc:
            print(f"wandb init failed, continuing without experiment tracking: {exc}")

    progress = None
    progress_task_id = None
    if show_progress:
        try:
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                TaskProgressColumn,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )

            progress_total = total_games if total_games is not None else num_updates
            progress_units = "games" if total_games is not None else "updates"
            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TextColumn(f"[cyan]{progress_units}[/cyan]"),
                TextColumn("loss={task.fields[loss]:.4f}"),
                TextColumn("points={task.fields[points]:.1f}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=False,
            )
            progress.start()
            progress_task_id = progress.add_task(
                "PPO self-play",
                total=progress_total,
                completed=games_played if total_games is not None else completed_updates,
                loss=history["loss"][-1] if history["loss"] else 0.0,
                points=history["mean_final_points"][-1] if history["mean_final_points"] else 0.0,
            )
        except Exception as exc:
            print(f"rich progress unavailable, continuing without progress bar: {exc}")

    try:
        last_saved_update_bucket = (
            completed_updates // checkpoint_every_updates if checkpoint_every_updates is not None else 0
        )
        last_saved_games_bucket = games_played // checkpoint_every_games if checkpoint_every_games is not None else 0

        while True:
            if total_games is not None:
                remaining_games = total_games - games_played
                if remaining_games <= 0:
                    break
                games_this_batch = min(episodes_per_batch, remaining_games)
            else:
                if completed_updates >= int(num_updates):
                    break
                games_this_batch = episodes_per_batch

            batch, key, rollout_stats = collect_self_play_batch(
                params=params,
                key=key,
                episodes_per_batch=games_this_batch,
                seed=next_env_seed,
                gamma=gamma,
                gae_lambda=gae_lambda,
            )
            next_env_seed += games_this_batch

            batch_size = batch["obs"].shape[0]
            cumulative_agent_steps += batch_size
            metrics_accumulator = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
            num_minibatches = 0

            for _ in range(ppo_epochs):
                permutation = rng.permutation(batch_size)
                for start in range(0, batch_size, minibatch_size):
                    idx = permutation[start : start + minibatch_size]
                    if idx.size == 0:
                        continue

                    params, optimizer_state, metrics = ppo_update_minibatch(
                        params,
                        optimizer_state,
                        jnp.asarray(batch["obs"][idx]),
                        jnp.asarray(batch["actions"][idx]),
                        jnp.asarray(batch["log_probs"][idx]),
                        jnp.asarray(batch["advantages"][idx]),
                        jnp.asarray(batch["returns"][idx]),
                        learning_rate,
                        clip_epsilon,
                        value_coef,
                        entropy_coef,
                    )
                    metrics_accumulator = {
                        key_name: metrics_accumulator[key_name] + float(metrics[key_name]) for key_name in metrics_accumulator
                    }
                    num_minibatches += 1

            averaged_metrics = {
                key_name: metrics_accumulator[key_name] / max(1, num_minibatches) for key_name in metrics_accumulator
            }
            completed_updates += 1
            games_played += games_this_batch
            for key_name, value in averaged_metrics.items():
                history[key_name].append(value)
            for key_name, value in rollout_stats.items():
                history[key_name].append(value)
            history["games_played"].append(float(games_played))
            history["agent_steps"].append(float(cumulative_agent_steps))

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "update": completed_updates - 1,
                        "games_played": games_played,
                        "agent_steps": cumulative_agent_steps,
                        "rollout/mean_match_result": rollout_stats["mean_match_result"],
                        "rollout/mean_final_points": rollout_stats["mean_final_points"],
                        "rollout/mean_episode_length": rollout_stats["mean_episode_length"],
                        "train/loss": averaged_metrics["loss"],
                        "train/policy_loss": averaged_metrics["policy_loss"],
                        "train/value_loss": averaged_metrics["value_loss"],
                        "train/entropy": averaged_metrics["entropy"],
                    },
                    step=completed_updates - 1,
                )

            if progress is not None and progress_task_id is not None:
                progress.update(
                    progress_task_id,
                    completed=games_played if total_games is not None else completed_updates,
                    loss=averaged_metrics["loss"],
                    points=rollout_stats["mean_final_points"],
                )

            should_save = False
            if checkpoint_path is not None and checkpoint_every_updates is not None:
                current_update_bucket = completed_updates // checkpoint_every_updates
                if current_update_bucket > last_saved_update_bucket:
                    last_saved_update_bucket = current_update_bucket
                    should_save = True
            if checkpoint_path is not None and checkpoint_every_games is not None:
                current_games_bucket = games_played // checkpoint_every_games
                if current_games_bucket > last_saved_games_bucket:
                    last_saved_games_bucket = current_games_bucket
                    should_save = True
            if should_save:
                save_training_checkpoint(
                    checkpoint_path,
                    params=params,
                    optimizer_state=optimizer_state,
                    rng=rng,
                    key=key,
                    history=history,
                    completed_updates=completed_updates,
                    games_played=games_played,
                    cumulative_agent_steps=cumulative_agent_steps,
                    next_env_seed=next_env_seed,
                    training_config=training_config,
                )
    finally:
        if progress is not None:
            progress.stop()
        if checkpoint_path is not None:
            save_training_checkpoint(
                checkpoint_path,
                params=params,
                optimizer_state=optimizer_state,
                rng=rng,
                key=key,
                history=history,
                completed_updates=completed_updates,
                games_played=games_played,
                cumulative_agent_steps=cumulative_agent_steps,
                next_env_seed=next_env_seed,
                training_config=training_config,
            )
        if wandb_run is not None:
            wandb_run.finish()

    return params, history


def train_self_play_sac(
    seed: int = 0,
    num_updates: int | None = 120,
    total_games: int | None = None,
    episodes_per_batch: int = 128,
    replay_buffer_size: int = 200_000,
    warmup_games: int = 64,
    gradient_steps_per_update: int = 256,
    batch_size: int = 512,
    learning_rate: float = 3e-4,
    actor_learning_rate: float | None = None,
    critic_learning_rate: float | None = None,
    gamma: float = 0.99,
    alpha: float = 0.1,
    target_update_tau: float = 0.01,
    hidden_sizes: Sequence[int] = (128, 128),
    use_wandb: bool = False,
    wandb_project: str = "higher-lower-v2-sac",
    wandb_run_name: str | None = None,
    wandb_mode: str | None = None,
    wandb_config: Dict[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
    resume_from_checkpoint: str | Path | None = None,
    checkpoint_every_updates: int | None = None,
    checkpoint_every_games: int | None = None,
    show_progress: bool = True,
):
    _require_jax("training a SAC strategy")
    if total_games is None and num_updates is None:
        raise ValueError("Provide either total_games or num_updates.")
    if total_games is not None and total_games < 0:
        raise ValueError(f"total_games must be non-negative, got {total_games}.")
    if num_updates is not None and num_updates < 0:
        raise ValueError(f"num_updates must be non-negative, got {num_updates}.")
    if episodes_per_batch <= 0:
        raise ValueError(f"episodes_per_batch must be positive, got {episodes_per_batch}.")
    if replay_buffer_size <= 0:
        raise ValueError(f"replay_buffer_size must be positive, got {replay_buffer_size}.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if gradient_steps_per_update <= 0:
        raise ValueError(f"gradient_steps_per_update must be positive, got {gradient_steps_per_update}.")
    if checkpoint_every_updates is not None and checkpoint_every_updates <= 0:
        raise ValueError(f"checkpoint_every_updates must be positive, got {checkpoint_every_updates}.")
    if checkpoint_every_games is not None and checkpoint_every_games <= 0:
        raise ValueError(f"checkpoint_every_games must be positive, got {checkpoint_every_games}.")
    if (checkpoint_every_updates is not None or checkpoint_every_games is not None) and checkpoint_path is None:
        raise ValueError("Set checkpoint_path when requesting periodic checkpoint saves.")

    actor_learning_rate = learning_rate if actor_learning_rate is None else actor_learning_rate
    critic_learning_rate = learning_rate if critic_learning_rate is None else critic_learning_rate
    warmup_transitions = warmup_games * 2 * max(1, game.NUM_CARDS_TO_PLAY - 1)
    replay_buffer = ReplayBuffer(replay_buffer_size)

    if resume_from_checkpoint is not None:
        resumed_state = load_training_checkpoint(resume_from_checkpoint)
        params = resumed_state["params"]
        target_params = resumed_state["extra_state"]["target_params"]
        optimizer_state = resumed_state["optimizer_state"]
        rng = resumed_state["rng"]
        key = resumed_state["key"]
        completed_updates = resumed_state["completed_updates"]
        games_played = resumed_state["games_played"]
        cumulative_agent_steps = resumed_state["cumulative_agent_steps"]
        next_env_seed = resumed_state["next_env_seed"]
        history = resumed_state["history"]
    else:
        rng = np.random.RandomState(seed)
        key = jax.random.PRNGKey(seed)
        key, actor_key, critic_1_key, critic_2_key = jax.random.split(key, 4)
        params = {
            "actor": init_actor_params(actor_key, hidden_sizes=hidden_sizes),
            "critic_1": init_q_params(critic_1_key, hidden_sizes=hidden_sizes),
            "critic_2": init_q_params(critic_2_key, hidden_sizes=hidden_sizes),
        }
        target_params = {
            "critic_1": params["critic_1"],
            "critic_2": params["critic_2"],
        }
        optimizer_state = {
            "actor": adam_init(params["actor"]),
            "critic_1": adam_init(params["critic_1"]),
            "critic_2": adam_init(params["critic_2"]),
        }
        completed_updates = 0
        games_played = 0
        cumulative_agent_steps = 0
        next_env_seed = seed * 10000
        history = {
            "critic_loss": [],
            "actor_loss": [],
            "entropy": [],
            "target_q": [],
            "mean_q": [],
            "mean_match_result": [],
            "mean_final_points": [],
            "mean_episode_length": [],
            "mean_action_log_prob": [],
            "games_played": [],
            "agent_steps": [],
            "replay_buffer_size": [],
        }

    for key_name in (
        "critic_loss",
        "actor_loss",
        "entropy",
        "target_q",
        "mean_q",
        "mean_match_result",
        "mean_final_points",
        "mean_episode_length",
        "mean_action_log_prob",
        "games_played",
        "agent_steps",
        "replay_buffer_size",
    ):
        history.setdefault(key_name, [])

    training_config = {
        "algorithm": "sac",
        "seed": seed,
        "num_updates": num_updates,
        "total_games": total_games,
        "episodes_per_batch": episodes_per_batch,
        "replay_buffer_size": replay_buffer_size,
        "warmup_games": warmup_games,
        "gradient_steps_per_update": gradient_steps_per_update,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "actor_learning_rate": actor_learning_rate,
        "critic_learning_rate": critic_learning_rate,
        "gamma": gamma,
        "alpha": alpha,
        "target_update_tau": target_update_tau,
        "hidden_sizes": list(hidden_sizes),
        "use_wandb": use_wandb,
        "wandb_project": wandb_project,
        "wandb_run_name": wandb_run_name,
        "wandb_mode": wandb_mode,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "resume_from_checkpoint": str(resume_from_checkpoint) if resume_from_checkpoint is not None else None,
        "checkpoint_every_updates": checkpoint_every_updates,
        "checkpoint_every_games": checkpoint_every_games,
        "show_progress": show_progress,
    }

    wandb_run = None
    if use_wandb:
        try:
            wandb = importlib.import_module("wandb")
            if not hasattr(wandb, "init"):
                raise AttributeError(
                    f"Imported `wandb` from {getattr(wandb, '__file__', 'unknown location')}, "
                    "but it does not expose `wandb.init`. Restart the kernel and re-import the real Weights & Biases package."
                )

            run_dir = Path.cwd() / "wandb"
            run_dir.mkdir(parents=True, exist_ok=True)

            config = {
                "algorithm": "sac",
                "seed": seed,
                "num_updates": num_updates,
                "total_games": total_games,
                "episodes_per_batch": episodes_per_batch,
                "replay_buffer_size": replay_buffer_size,
                "warmup_games": warmup_games,
                "gradient_steps_per_update": gradient_steps_per_update,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "actor_learning_rate": actor_learning_rate,
                "critic_learning_rate": critic_learning_rate,
                "gamma": gamma,
                "alpha": alpha,
                "target_update_tau": target_update_tau,
                "hidden_sizes": list(hidden_sizes),
                "num_actions": len(ACTION_FRACTIONS),
                "observation_dim": OBSERVATION_DIM,
                "resumed_updates": completed_updates,
                "resumed_games": games_played,
            }
            if wandb_config:
                config.update(wandb_config)
            wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                mode=wandb_mode,
                config=config,
                dir=str(run_dir),
                reinit=True,
            )
        except Exception as exc:
            print(f"wandb init failed, continuing without experiment tracking: {exc}")

    progress = None
    progress_task_id = None
    if show_progress:
        try:
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                TaskProgressColumn,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )

            progress_total = total_games if total_games is not None else num_updates
            progress_units = "games" if total_games is not None else "updates"
            progress = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TextColumn(f"[cyan]{progress_units}[/cyan]"),
                TextColumn("actor={task.fields[actor_loss]:.4f}"),
                TextColumn("points={task.fields[points]:.1f}"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                transient=False,
            )
            progress.start()
            progress_task_id = progress.add_task(
                "SAC self-play",
                total=progress_total,
                completed=games_played if total_games is not None else completed_updates,
                actor_loss=history["actor_loss"][-1] if history["actor_loss"] else 0.0,
                points=history["mean_final_points"][-1] if history["mean_final_points"] else 0.0,
            )
        except Exception as exc:
            print(f"rich progress unavailable, continuing without progress bar: {exc}")

    try:
        last_saved_update_bucket = (
            completed_updates // checkpoint_every_updates if checkpoint_every_updates is not None else 0
        )
        last_saved_games_bucket = games_played // checkpoint_every_games if checkpoint_every_games is not None else 0

        while True:
            if total_games is not None:
                remaining_games = total_games - games_played
                if remaining_games <= 0:
                    break
                games_this_batch = min(episodes_per_batch, remaining_games)
            else:
                if completed_updates >= int(num_updates):
                    break
                games_this_batch = episodes_per_batch

            key, rollout_stats = collect_self_play_replay(
                params["actor"],
                key=key,
                episodes_per_batch=games_this_batch,
                seed=next_env_seed,
                replay_buffer=replay_buffer,
            )
            next_env_seed += games_this_batch
            games_played += games_this_batch
            cumulative_agent_steps += int(rollout_stats["transitions_collected"])

            metrics_accumulator = {
                "critic_loss": 0.0,
                "actor_loss": 0.0,
                "entropy": 0.0,
                "target_q": 0.0,
                "mean_q": 0.0,
            }
            num_gradient_updates = 0

            if replay_buffer.size >= max(batch_size, warmup_transitions):
                for _ in range(gradient_steps_per_update):
                    batch = replay_buffer.sample(batch_size, rng)
                    params, target_params, optimizer_state, metrics = sac_update_step(
                        params,
                        target_params,
                        optimizer_state,
                        jnp.asarray(batch["obs"]),
                        jnp.asarray(batch["actions"]),
                        jnp.asarray(batch["rewards"]),
                        jnp.asarray(batch["next_obs"]),
                        jnp.asarray(batch["dones"]),
                        actor_learning_rate,
                        critic_learning_rate,
                        gamma,
                        alpha,
                        target_update_tau,
                    )
                    metrics_accumulator = {
                        metric_name: metrics_accumulator[metric_name] + float(metrics[metric_name])
                        for metric_name in metrics_accumulator
                    }
                    num_gradient_updates += 1

            averaged_metrics = {
                metric_name: metrics_accumulator[metric_name] / max(1, num_gradient_updates)
                for metric_name in metrics_accumulator
            }
            completed_updates += 1

            for key_name, value in averaged_metrics.items():
                history[key_name].append(value)
            for key_name in (
                "mean_match_result",
                "mean_final_points",
                "mean_episode_length",
                "mean_action_log_prob",
            ):
                history[key_name].append(rollout_stats[key_name])
            history["games_played"].append(float(games_played))
            history["agent_steps"].append(float(cumulative_agent_steps))
            history["replay_buffer_size"].append(float(replay_buffer.size))

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "update": completed_updates - 1,
                        "games_played": games_played,
                        "agent_steps": cumulative_agent_steps,
                        "replay_buffer/size": replay_buffer.size,
                        "rollout/mean_match_result": rollout_stats["mean_match_result"],
                        "rollout/mean_final_points": rollout_stats["mean_final_points"],
                        "rollout/mean_episode_length": rollout_stats["mean_episode_length"],
                        "rollout/mean_action_log_prob": rollout_stats["mean_action_log_prob"],
                        "train/critic_loss": averaged_metrics["critic_loss"],
                        "train/actor_loss": averaged_metrics["actor_loss"],
                        "train/entropy": averaged_metrics["entropy"],
                        "train/target_q": averaged_metrics["target_q"],
                        "train/mean_q": averaged_metrics["mean_q"],
                    },
                    step=completed_updates - 1,
                )

            if progress is not None and progress_task_id is not None:
                progress.update(
                    progress_task_id,
                    completed=games_played if total_games is not None else completed_updates,
                    actor_loss=averaged_metrics["actor_loss"],
                    points=rollout_stats["mean_final_points"],
                )

            should_save = False
            if checkpoint_path is not None and checkpoint_every_updates is not None:
                current_update_bucket = completed_updates // checkpoint_every_updates
                if current_update_bucket > last_saved_update_bucket:
                    last_saved_update_bucket = current_update_bucket
                    should_save = True
            if checkpoint_path is not None and checkpoint_every_games is not None:
                current_games_bucket = games_played // checkpoint_every_games
                if current_games_bucket > last_saved_games_bucket:
                    last_saved_games_bucket = current_games_bucket
                    should_save = True
            if should_save:
                save_training_checkpoint(
                    checkpoint_path,
                    params=params,
                    optimizer_state=optimizer_state,
                    rng=rng,
                    key=key,
                    history=history,
                    completed_updates=completed_updates,
                    games_played=games_played,
                    cumulative_agent_steps=cumulative_agent_steps,
                    next_env_seed=next_env_seed,
                    training_config=training_config,
                    extra_state={"target_params": target_params},
                )
    finally:
        if progress is not None:
            progress.stop()
        if checkpoint_path is not None:
            save_training_checkpoint(
                checkpoint_path,
                params=params,
                optimizer_state=optimizer_state,
                rng=rng,
                key=key,
                history=history,
                completed_updates=completed_updates,
                games_played=games_played,
                cumulative_agent_steps=cumulative_agent_steps,
                next_env_seed=next_env_seed,
                training_config=training_config,
                extra_state={"target_params": target_params},
            )
        if wandb_run is not None:
            wandb_run.finish()

    return params, history


def train_self_play(algorithm: str = "ppo", **kwargs):
    algorithm_name = algorithm.lower()
    def _filter_kwargs(fn, values):
        signature = inspect.signature(fn)
        return {key: value for key, value in values.items() if key in signature.parameters}

    if algorithm_name == "ppo":
        return train_self_play_ppo(**_filter_kwargs(train_self_play_ppo, kwargs))
    if algorithm_name == "sac":
        return train_self_play_sac(**_filter_kwargs(train_self_play_sac, kwargs))
    raise ValueError(f"Unsupported algorithm {algorithm!r}. Expected 'ppo' or 'sac'.")


class PolicyStrategy(game.Strategy):
    def __init__(self, params, name: str = "Policy"):
        self.params = _tree_to_numpy(extract_policy_params(params))
        self.name = name

    @classmethod
    def from_weights_file(
        cls,
        weights_source: str | Path,
        *,
        expected_sha256: str | None = None,
        cache_dir: str | Path | None = None,
        timeout_seconds: float = 60.0,
        force_download: bool = False,
        name: str | None = None,
    ) -> "PolicyStrategy":
        loaded = load_policy_weights(
            weights_source,
            expected_sha256=expected_sha256,
            cache_dir=cache_dir,
            timeout_seconds=timeout_seconds,
            force_download=force_download,
        )
        strategy_name = name or loaded["metadata"].get("name") or "Policy"
        return cls(loaded["params"], name=strategy_name)

    def evaluate(
        self,
        my_points: int,
        op_points: int,
        prev_cards_drawn: List[int],
        num_remaining_draws: int,
        rng: np.random.RandomState,
    ) -> int:
        del rng
        obs = encode_observation(
            my_points=my_points,
            op_points=op_points,
            prev_cards_drawn=prev_cards_drawn,
            num_remaining_draws=num_remaining_draws,
        )[None, :]
        action_index = int(greedy_actions_numpy(self.params, obs)[0])
        return action_index_to_stake(action_index, my_points)

    def __str__(self):
        return self.name


class RemotePolicyStrategy(game.Strategy):
    def __init__(
        self,
        weights_source: str | Path,
        *,
        expected_sha256: str | None = None,
        cache_dir: str | Path | None = None,
        timeout_seconds: float = 60.0,
        force_download: bool = False,
        name: str = "RemotePolicy",
    ):
        self.weights_source = weights_source
        self.expected_sha256 = expected_sha256
        self.cache_dir = cache_dir
        self.timeout_seconds = timeout_seconds
        self.force_download = force_download
        self.name = name
        self._delegate: PolicyStrategy | None = None

    def _get_delegate(self) -> PolicyStrategy:
        if self._delegate is None:
            self._delegate = PolicyStrategy.from_weights_file(
                self.weights_source,
                expected_sha256=self.expected_sha256,
                cache_dir=self.cache_dir,
                timeout_seconds=self.timeout_seconds,
                force_download=self.force_download,
                name=self.name,
            )
        return self._delegate

    def evaluate(
        self,
        my_points: int,
        op_points: int,
        prev_cards_drawn: List[int],
        num_remaining_draws: int,
        rng: np.random.RandomState,
    ) -> int:
        return self._get_delegate().evaluate(
            my_points=my_points,
            op_points=op_points,
            prev_cards_drawn=prev_cards_drawn,
            num_remaining_draws=num_remaining_draws,
            rng=rng,
        )

    def __str__(self):
        return self.name


class PPOPolicyStrategy(PolicyStrategy):
    pass


class RemotePPOPolicyStrategy(RemotePolicyStrategy):
    pass


class SACPolicyStrategy(PolicyStrategy):
    pass


class RemoteSACPolicyStrategy(RemotePolicyStrategy):
    pass
