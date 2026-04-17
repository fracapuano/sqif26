# sqif26

Higher or Lower V2 experiments, training code, checkpoints, and submission files
for SQIF 2026.

The repo contains three distinct things:
- the game and notebook walkthrough
- training code for PPO and SAC self-play
- shareable submission files, including a standalone `submission.py`

## Repository layout

- `walkthrough.ipynb`: notebook walkthrough of the game, baseline strategies, and PPO training flow
- `deck.py`, `game.py`: game mechanics and the `game.Strategy` interface
- `higher_lower_rl.py`: PPO and SAC training code, checkpointing, export helpers, and policy wrappers
- `train_sac.py`: SAC training entrypoint
- `submission.py`: standalone submission file that only depends on `game.py`, `deck.py`, `numpy`, and stdlib
- `submission_strategy.py`: configurable submission wrapper that reuses helper code
- `submission_embedded.py`: generated fully embedded policy submission
- `build_embedded_submission.py`: helper to generate embedded submissions from checkpoints or policy files
- `artifacts/submission_policy.pkl`: exported NumPy inference weights
- `artifacts/jax_ppo_self_play.pkl`: PPO training checkpoint
- `artifacts/jax_sac_self_play.pkl`: SAC training checkpoint

## Install

Full training environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Minimal submission/runtime environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-submission.txt
```

If you want to open the notebook, also install Jupyter:

```bash
pip install notebook
```

## Quick start

Open the walkthrough notebook:

```bash
jupyter notebook walkthrough.ipynb
```

Run the standalone submission file locally:

```python
import numpy as np
from submission import MyStrategy

strategy = MyStrategy()
stake = strategy.evaluate(100, 100, [10], 4, np.random.RandomState(0))
print(stake)
```

`submission.py` is the clean review target for other people:
- it does not import `higher_lower_rl.py`
- it downloads the latest exported policy from this repo by default
- it runs pure NumPy inference

Current remote policy URL used by default:

```text
https://raw.githubusercontent.com/fracapuano/sqif26/main/artifacts/submission_policy.pkl
```

## Training

PPO training is driven from `walkthrough.ipynb` and the helpers in `higher_lower_rl.py`.

SAC training can be launched directly:

```bash
python train_sac.py
```

Useful environment overrides:

```bash
TOTAL_GAMES=2000000 GAMES_PER_BATCH=128 python train_sac.py
RESUME_FROM_CHECKPOINT=artifacts/jax_sac_self_play.pkl python train_sac.py
```

## Exporting a policy

The committed deployable policy is `artifacts/submission_policy.pkl`.

If you have a fresh checkpoint and want to regenerate the exported inference
policy, use the helpers in `higher_lower_rl.py`. Example:

```python
import pickle
from pathlib import Path
from higher_lower_rl import save_policy_weights

checkpoint_path = Path("artifacts/jax_ppo_self_play.pkl")
with checkpoint_path.open("rb") as handle:
    payload = pickle.load(handle)

save_policy_weights(
    "artifacts/submission_policy.pkl",
    payload["params"],
    metadata={
        "name": "SubmissionPPO",
        "source_checkpoint": str(checkpoint_path),
        "completed_updates": int(payload.get("completed_updates", 0)),
        "games_played": int(payload.get("games_played", 0)),
    },
)
```

## Submission modes

Remote-weight submission:
- use `submission.py`
- smallest code that still allows rolling forward the weights by updating the repo artifact

Helper-based submission:
- use `submission_strategy.py`
- easier to experiment with locally, but not self-contained enough for strict submission environments

Embedded submission:
- use `submission_embedded.py`
- no remote fetch required, but the file is much larger because the policy weights are inlined

Regenerate the embedded submission from the current exported policy:

```bash
python build_embedded_submission.py \
  --source artifacts/submission_policy.pkl \
  --dest submission_embedded.py \
  --name SubmissionPPO
```

## Sanity checks

Basic syntax validation:

```bash
python -m py_compile deck.py game.py higher_lower_rl.py train_sac.py submission.py
```

Basic submission smoke test:

```python
import numpy as np
from submission import MyStrategy

strategy = MyStrategy()
print(strategy.evaluate(100, 100, [10], 4, np.random.RandomState(0)))
```

## Notes

- `submission.py` intentionally leaves policy SHA pinning off by default so that updating
  `artifacts/submission_policy.pkl` in this repo updates the remotely fetched policy automatically.
- `game.py` only imports `scipy` and `matplotlib` inside the ELO/plotting helpers, so the
  minimal submission path does not need them unless you are plotting or computing tournament ELO.
