from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .env import AsteroidsEnv
from .evaluate import load_checkpoint_policy, run_evaluation, summarize
from .policies import RandomPolicy, RuleBasedPolicy


def _difference_ci(first: List[Dict[str, Any]], baseline: List[Dict[str, Any]], key: str) -> List[float]:
    differences = np.asarray([row[key] for row in first], dtype=np.float64) - np.asarray(
        [row[key] for row in baseline], dtype=np.float64
    )
    rng = np.random.default_rng(0)
    draws = rng.choice(differences, size=(2_000, len(differences)), replace=True).mean(axis=1)
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Compare AsteroidsAI policies on identical held-out seeds")
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Checkpoint to include; may be repeated",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=Path("runs/benchmark.json"))
    args = parser.parse_args(argv)
    seeds = range(args.seed_start, args.seed_start + args.episodes)

    baseline_env = AsteroidsEnv()
    random_records = run_evaluation(baseline_env, RandomPolicy(seed=args.seed_start), seeds)
    results: Dict[str, Any] = {"random": summarize(random_records)}
    rule_records = run_evaluation(baseline_env, RuleBasedPolicy(), seeds)
    results["rule"] = summarize(rule_records)
    results["rule"]["difference_from_random"] = {
        key: _difference_ci(rule_records, random_records, key) for key in ("return", "hits", "score")
    }
    baseline_env.close()

    for item in args.checkpoint:
        if "=" not in item:
            parser.error("--checkpoint values must use LABEL=PATH")
        label, raw_path = item.split("=", 1)
        env, policy = load_checkpoint_policy(Path(raw_path), args.device)
        records = run_evaluation(env, policy, seeds)
        results[label] = summarize(records)
        results[label]["difference_from_random"] = {
            key: _difference_ci(records, random_records, key) for key in ("return", "hits", "score")
        }
        env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

