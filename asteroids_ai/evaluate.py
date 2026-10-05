from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

from .dqn import DQNAgent, DQNConfig
from .env import AsteroidsEnv, EnvConfig, Observation, RewardConfig
from .policies import RandomPolicy, RuleBasedPolicy


def run_evaluation(
    env: AsteroidsEnv,
    policy: Any,
    seeds: Iterable[int],
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for seed in seeds:
        observation, _ = env.reset(seed=int(seed))
        terminated = truncated = False
        info: Dict[str, Any] = {}
        episode_return = 0.0
        action_counts = np.zeros((int(env.action_space.n),), dtype=np.int64)
        while not (terminated or truncated):
            action = int(policy.act(observation))
            action_counts[action] += 1
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
        records.append(
            {
                "seed": int(seed),
                "return": episode_return,
                "score": info["score"],
                "survival_steps": info["survival_steps"],
                "hits": info["hits"],
                "shots": info["shots"],
                "accuracy": info["accuracy"],
                "collisions": info["collisions"],
                "pickups": info["pickups"],
                "death_reason": info["death_reason"],
                "action_counts": action_counts.tolist(),
            }
        )
    return records


def bootstrap_mean_ci(values: np.ndarray, seed: int = 0, samples: int = 2_000) -> List[float]:
    if not len(values):
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"episodes": len(records)}
    for key in ("return", "score", "survival_steps", "hits", "shots", "accuracy", "collisions", "pickups"):
        values = np.asarray([row[key] for row in records], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()) if len(values) else 0.0,
            "median": float(np.median(values)) if len(values) else 0.0,
            "std": float(values.std()) if len(values) else 0.0,
            "mean_95_ci": bootstrap_mean_ci(values),
        }
    return summary


class _GreedyAgentPolicy:
    def __init__(self, agent: DQNAgent) -> None:
        self.agent = agent

    def act(self, observation: Observation) -> int:
        return self.agent.act(observation, epsilon=0.0)


def load_checkpoint_policy(path: Path, device: str = "auto") -> tuple[AsteroidsEnv, _GreedyAgentPolicy]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    env_data = state.get("env_config", {})
    env = AsteroidsEnv(
        EnvConfig(**env_data.get("environment", {})),
        RewardConfig(**env_data.get("reward", {})),
    )
    agent = DQNAgent(
        env,
        model_type=state["model_type"],
        config=DQNConfig(**state.get("dqn_config", {})),
        device=device,
    )
    agent.online.load_state_dict(state["online"])
    agent.target.load_state_dict(state.get("target", state["online"]))
    return env, _GreedyAgentPolicy(agent)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate AsteroidsAI policies on shared fixed seeds")
    parser.add_argument("--policy", choices=("random", "rule", "checkpoint"), default="rule")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation"))
    args = parser.parse_args(argv)

    if args.policy == "checkpoint":
        if args.checkpoint is None:
            parser.error("--checkpoint is required when --policy=checkpoint")
        env, policy = load_checkpoint_policy(args.checkpoint, args.device)
    else:
        env = AsteroidsEnv()
        policy = RandomPolicy(seed=args.seed_start) if args.policy == "random" else RuleBasedPolicy()

    records = run_evaluation(env, policy, range(args.seed_start, args.seed_start + args.episodes))
    summary = summarize(records)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / f"{args.policy}_episodes.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    with (args.output / f"{args.policy}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    env.close()


if __name__ == "__main__":
    main()
