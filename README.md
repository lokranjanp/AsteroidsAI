# AsteroidsAI v2

AsteroidsAI is a deterministic vertical dodger-shooter and reinforcement-learning benchmark. Human play, scripted policies, training, and evaluation now use the same environment, so gameplay mechanics cannot silently diverge between experiments.

The repository retains the original undergraduate-project implementation and checkpoints as legacy artifacts. Version 2 introduces a clean observation/action contract; old checkpoints are intentionally not compatible.

## What changed

- A Gymnasium-style environment with seeded, fixed-step physics and optional rendering.
- Six discrete actions: idle, left, right, fire, left-and-fire, and right-and-fire.
- Structured observations containing ship state plus masked asteroid, bullet, and pickup entities.
- Auditable per-transition rewards rather than cumulative episode reward.
- Double/Dueling DQN with a target network, Huber loss, gradient clipping, n-step returns, prioritized replay, and resumable checkpoints.
- Comparable MLP and entity-transformer encoders.
- Random and rule-based baselines, fixed-seed evaluation, JSONL metrics, confidence intervals, and TensorBoard logging.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The compatibility dependency file can also be used:

```bash
python -m pip install -r requirements.txt
```

## Human play

```bash
python main.py
```

Use A/D or the arrow keys to move and Space to fire. Movement and firing can be combined. Press R to restart an episode.

## Training

Run the reference MLP or transformer experiment:

```bash
python -m asteroids_ai.train --config configs/mlp.json
python -m asteroids_ai.train --config configs/transformer.json
```

Useful quick-run overrides:

```bash
python -m asteroids_ai.train --model mlp --steps 10000 --seed 1 --device auto --output runs/smoke
```

Device selection prefers CUDA, then Apple MPS, then CPU. The default reference budget is 500,000 environment steps. Checkpoints, episode JSONL, TensorBoard events, and a run summary are written beneath the selected output directory.

Resume a run with:

```bash
python -m asteroids_ai.train --resume runs/mlp_seed_0/checkpoint_25000.pth --steps 500000
```

## Evaluation and comparison

Evaluation always uses greedy actions for trained agents and defaults to 100 held-out seeds:

```bash
python -m asteroids_ai.evaluate --policy random
python -m asteroids_ai.evaluate --policy rule
python -m asteroids_ai.evaluate --policy checkpoint --checkpoint runs/mlp_seed_0/final.pth
```

Compare baselines and multiple trained models on identical seeds:

```bash
python -m asteroids_ai.benchmark \
  --checkpoint mlp=runs/mlp_seed_0/final.pth \
  --checkpoint transformer=runs/transformer_seed_0/final.pth
```

The benchmark reports mean, median, standard deviation, bootstrap 95% confidence intervals, and paired differences from the random policy.

## Environment contract

`AsteroidsEnv.reset(seed=...)` returns `(observation, info)`. `step(action)` returns `(observation, reward, terminated, truncated, info)`.

The observation is a dictionary:

- `global` — seven normalized ship and episode features.
- `entities` — up to 22 rows with normalized relative position, velocity, size, and a one-hot entity type.
- `entity_mask` — active/padded entity rows.

The default reward is the sum of independently logged components:

- +1.0 for destroying an asteroid.
- -1.0 for a collision.
- -2.0 once on terminal death.
- -0.02 for a successfully fired shot.
- Up to +0.25 for a pickup, scaled by the amount actually restored.
- +0.001 for each non-terminal survival step.

Display score remains `100 × hits - 2 × shots` and is deliberately separate from training return.

## Tests

The suite uses the standard library and runs without opening a game window:

```bash
python -m unittest discover -s tests -v
```

It covers determinism, reset isolation, observation contracts, reward accounting, truncation, masked transformer behavior, batch independence, replay/n-step logic, and detached target-network learning.

## Legacy material

`gameAI.py`, `DQN.py`, `TQN.py`, `model/`, `TQN_model/`, historical reports, and the gameplay CSV are retained for comparison. New development should use the `asteroids_ai` package and v2 checkpoints.

Graphics originate from [Kenney's Space Shooter Redux](https://www.kenney.nl/assets/space-shooter-redux).
