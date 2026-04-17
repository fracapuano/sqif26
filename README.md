# sqif26

Higher or Lower V2 experiments, training code, checkpoints, and submission files
for SQIF 2026.

Contents:
- `walkthrough.ipynb`: notebook walkthrough of the game and RL setup
- `deck.py`, `game.py`: game mechanics and strategy interface
- `higher_lower_rl.py`: PPO training code and policy export helpers
- `train_sac.py`: SAC training entrypoint
- `submission.py`: standalone remote-weight submission file
- `submission_strategy.py`: configurable submission wrapper
- `submission_embedded.py`: generated fully embedded policy submission
- `build_embedded_submission.py`: helper to generate embedded submissions
- `artifacts/submission_policy.pkl`: exported NumPy inference weights
- `artifacts/jax_ppo_self_play.pkl`: PPO training checkpoint
- `artifacts/jax_sac_self_play.pkl`: SAC training checkpoint

Quick start:
- Open `walkthrough.ipynb` to inspect the environment and baseline workflow.
- Use `submission.py` if you want a single-file strategy that fetches the latest
  exported weights from GitHub.
- Use `artifacts/submission_policy.pkl` with the training helpers if you want the
  deployable policy weights directly.
