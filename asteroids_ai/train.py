from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .dqn import DQNAgent, DQNConfig
from .env import AsteroidsEnv, EnvConfig
from .evaluate import run_evaluation, summarize
from .replay import NStepAccumulator, ReplayBuffer

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - TensorBoard is an optional runtime integration.
    class SummaryWriter:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def add_scalar(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def close(self) -> None:
            pass


def _load_config(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(argv: Optional[List[str]] = None) -> Path:
    parser = argparse.ArgumentParser(description="Train a Double/Dueling DQN agent")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", choices=("mlp", "transformer"), default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--no-prioritized-replay", action="store_true")
    args = parser.parse_args(argv)

    file_config = _load_config(args.config)
    resume_state: Optional[Dict[str, Any]] = None
    if args.resume is not None:
        resume_state = torch.load(args.resume, map_location="cpu", weights_only=False)
    checkpoint_env = resume_state.get("env_config", {}) if resume_state else {}
    checkpoint_dqn = resume_state.get("dqn_config", {}) if resume_state else {}
    model_type = args.model or file_config.get("model") or (
        resume_state.get("model_type", "mlp") if resume_state else "mlp"
    )
    total_steps = args.steps or int(file_config.get("steps", 500_000))
    seed = args.seed if args.seed is not None else int(file_config.get("seed", 0))
    device = args.device or file_config.get("device", "auto")
    output = args.output or Path(file_config.get("output", f"runs/{model_type}_seed_{seed}"))
    dqn_config = DQNConfig(**file_config.get("dqn", checkpoint_dqn))
    if args.no_prioritized_replay:
        dqn_config = replace(dqn_config, prioritized_replay=False)
    env_config = EnvConfig(**file_config.get("environment", checkpoint_env.get("environment", {})))

    _seed_everything(seed)
    env = AsteroidsEnv(env_config)
    observation, _ = env.reset(seed=seed)
    agent = DQNAgent(env, model_type, dqn_config, device=device, seed=seed)
    replay = ReplayBuffer(
        dqn_config.replay_capacity,
        env.GLOBAL_FEATURES,
        env.config.max_entities,
        env.ENTITY_FEATURES,
        prioritized=dqn_config.prioritized_replay,
        alpha=dqn_config.priority_alpha,
        seed=seed,
    )
    accumulator = NStepAccumulator(dqn_config.n_step, dqn_config.gamma)
    global_step = 0
    if resume_state is not None:
        global_step = agent.load_checkpoint(resume_state, replay)
        if "env_rng_state" in resume_state:
            env.set_rng_state(resume_state["env_rng_state"])
        observation, _ = env.reset()

    output.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(output / "tensorboard"))
    episode_log = (output / "episodes.jsonl").open("a", encoding="utf-8")
    episode_return = 0.0
    episode = 0
    start_time = time.monotonic()
    last_metrics: Dict[str, float] = {}

    try:
        while global_step < total_steps:
            epsilon = agent.epsilon(global_step)
            action = agent.act(observation, epsilon)
            next_observation, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            for transition in accumulator.append(
                observation, action, reward, next_observation, terminated, truncated
            ):
                replay.add(*transition)
            observation = next_observation
            global_step += 1

            if (
                len(replay) >= max(dqn_config.warmup_steps, dqn_config.batch_size)
                and global_step % dqn_config.train_frequency == 0
            ):
                sample = replay.sample(dqn_config.batch_size, agent.priority_beta(global_step))
                last_metrics = agent.train_batch(sample, replay)
                for key, value in last_metrics.items():
                    writer.add_scalar(f"train/{key}", value, global_step)
                writer.add_scalar("train/epsilon", epsilon, global_step)

            if terminated or truncated:
                episode += 1
                record = {
                    "episode": episode,
                    "global_step": global_step,
                    "return": episode_return,
                    **{key: value for key, value in info.items() if key != "reward_components"},
                }
                episode_log.write(json.dumps(record) + "\n")
                episode_log.flush()
                writer.add_scalar("episode/return", episode_return, global_step)
                writer.add_scalar("episode/score", info["score"], global_step)
                writer.add_scalar("episode/hits", info["hits"], global_step)
                observation, _ = env.reset()
                episode_return = 0.0

            if global_step % 25_000 == 0:
                evaluation_env = AsteroidsEnv(env_config)
                records = run_evaluation(
                    evaluation_env,
                    agent,
                    range(10_000, 10_020),
                )
                evaluation_env.close()
                evaluation_summary = summarize(records)
                writer.add_scalar("evaluation/mean_return", evaluation_summary["return"]["mean"], global_step)
                writer.add_scalar("evaluation/mean_hits", evaluation_summary["hits"]["mean"], global_step)
                agent.save(output / f"checkpoint_{global_step}.pth", global_step, env, replay)

        checkpoint = output / "final.pth"
        agent.save(checkpoint, global_step, env, replay)
        elapsed = max(time.monotonic() - start_time, 1e-9)
        metadata = {
            "model": model_type,
            "seed": seed,
            "global_step": global_step,
            "steps_per_second": global_step / elapsed,
            "environment": asdict(env_config),
            "dqn": asdict(dqn_config),
            "last_train_metrics": last_metrics,
        }
        with (output / "run_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)
        return checkpoint
    finally:
        episode_log.close()
        writer.close()
        env.close()


def main() -> None:
    checkpoint = train()
    print(f"Saved checkpoint to {checkpoint}")


if __name__ == "__main__":
    main()
