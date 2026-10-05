from __future__ import annotations

from typing import Protocol

import numpy as np

from .env import Action, Observation


class Policy(Protocol):
    def act(self, observation: Observation) -> int:
        ...


class RandomPolicy:
    def __init__(self, action_count: int = len(Action), seed: int = 0) -> None:
        self.action_count = action_count
        self.rng = np.random.default_rng(seed)

    def act(self, observation: Observation) -> int:
        del observation
        return int(self.rng.integers(0, self.action_count))


class RuleBasedPolicy:
    """Small deterministic baseline using only the public observation."""

    def act(self, observation: Observation) -> int:
        entities = observation["entities"]
        mask = observation["entity_mask"] > 0.0
        asteroids = entities[mask & (entities[:, 5] > 0.5)]
        cooldown_ready = observation["global"][4] <= 0.0
        if not len(asteroids):
            return int(Action.IDLE)

        nearest = min(asteroids, key=lambda row: abs(float(row[0])) + 0.35 * abs(float(row[1])))
        relative_x, relative_y = float(nearest[0]), float(nearest[1])
        is_near = relative_y > -0.35
        in_firing_lane = abs(relative_x) < 0.07

        if is_near and abs(relative_x) < 0.13:
            move = Action.LEFT if relative_x >= 0.0 else Action.RIGHT
        elif relative_x < -0.03:
            move = Action.LEFT
        elif relative_x > 0.03:
            move = Action.RIGHT
        else:
            move = Action.IDLE

        if not cooldown_ready or not in_firing_lane:
            return int(move)
        if move == Action.LEFT:
            return int(Action.LEFT_FIRE)
        if move == Action.RIGHT:
            return int(Action.RIGHT_FIRE)
        return int(Action.FIRE)

