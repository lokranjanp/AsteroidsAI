from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple, cast

import numpy as np

from .env import Observation


@dataclass
class ReplaySample:
    observations: Dict[str, np.ndarray]
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: Dict[str, np.ndarray]
    terminated: np.ndarray
    discounts: np.ndarray
    weights: np.ndarray
    indices: np.ndarray


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        global_features: int,
        max_entities: int,
        entity_features: int,
        prioritized: bool = False,
        alpha: float = 0.6,
        seed: int = 0,
    ) -> None:
        self.capacity = int(capacity)
        self.prioritized = prioritized
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self.position = 0
        self.size = 0
        self.globals = np.zeros((capacity, global_features), dtype=np.float32)
        self.entities = np.zeros((capacity, max_entities, entity_features), dtype=np.float32)
        self.masks = np.zeros((capacity, max_entities), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_globals = np.zeros_like(self.globals)
        self.next_entities = np.zeros_like(self.entities)
        self.next_masks = np.zeros_like(self.masks)
        self.terminated = np.zeros((capacity,), dtype=np.float32)
        self.discounts = np.zeros((capacity,), dtype=np.float32)
        self.priorities = np.ones((capacity,), dtype=np.float32)

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        observation: Observation,
        action: int,
        reward: float,
        next_observation: Observation,
        terminated: bool,
        discount: float,
    ) -> None:
        index = self.position
        self.globals[index] = observation["global"]
        self.entities[index] = observation["entities"]
        self.masks[index] = observation["entity_mask"]
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_globals[index] = next_observation["global"]
        self.next_entities[index] = next_observation["entities"]
        self.next_masks[index] = next_observation["entity_mask"]
        self.terminated[index] = float(terminated)
        self.discounts[index] = discount
        self.priorities[index] = float(self.priorities[: self.size].max()) if self.size else 1.0
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 1.0) -> ReplaySample:
        if self.size < batch_size:
            raise ValueError(f"Need {batch_size} transitions, only {self.size} available")
        if self.prioritized:
            scaled = np.maximum(self.priorities[: self.size], 1e-6) ** self.alpha
            probabilities = scaled / scaled.sum()
            indices = self.rng.choice(self.size, size=batch_size, replace=False, p=probabilities)
            weights = (self.size * probabilities[indices]) ** (-beta)
            weights = (weights / weights.max()).astype(np.float32)
        else:
            indices = self.rng.choice(self.size, size=batch_size, replace=False)
            weights = np.ones((batch_size,), dtype=np.float32)
        return ReplaySample(
            observations={
                "global": self.globals[indices].copy(),
                "entities": self.entities[indices].copy(),
                "entity_mask": self.masks[indices].copy(),
            },
            actions=self.actions[indices].copy(),
            rewards=self.rewards[indices].copy(),
            next_observations={
                "global": self.next_globals[indices].copy(),
                "entities": self.next_entities[indices].copy(),
                "entity_mask": self.next_masks[indices].copy(),
            },
            terminated=self.terminated[indices].copy(),
            discounts=self.discounts[indices].copy(),
            weights=weights,
            indices=indices,
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        if self.prioritized:
            self.priorities[indices] = np.maximum(np.asarray(priorities, dtype=np.float32), 1e-6)

    def state_dict(self, include_data: bool = True) -> Dict[str, object]:
        state: Dict[str, object] = {
            "capacity": self.capacity,
            "prioritized": self.prioritized,
            "alpha": self.alpha,
            "position": self.position,
            "size": self.size,
            "data_included": include_data,
            "rng_state": self.rng.bit_generator.state,
        }
        if include_data:
            names = (
                "globals",
                "entities",
                "masks",
                "actions",
                "rewards",
                "next_globals",
                "next_entities",
                "next_masks",
                "terminated",
                "discounts",
                "priorities",
            )
            state["data"] = {name: getattr(self, name)[: self.size].copy() for name in names}
        return state

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.rng.bit_generator.state = state["rng_state"]  # type: ignore[assignment]
        data = state.get("data")
        if isinstance(data, dict):
            self.position = cast(int, state["position"])
            self.size = cast(int, state["size"])
            for name, values in data.items():
                getattr(self, name)[: self.size] = values
        else:
            # Metadata-only checkpoints resume with an empty buffer and warm it up again.
            self.position = 0
            self.size = 0


@dataclass
class _PendingTransition:
    observation: Observation
    action: int
    reward: float
    next_observation: Observation
    terminated: bool
    episode_end: bool


NStepTransition = Tuple[Observation, int, float, Observation, bool, float]


class NStepAccumulator:
    def __init__(self, steps: int, gamma: float) -> None:
        if steps < 1:
            raise ValueError("steps must be at least one")
        self.steps = steps
        self.gamma = gamma
        self.pending: Deque[_PendingTransition] = deque()

    def append(
        self,
        observation: Observation,
        action: int,
        reward: float,
        next_observation: Observation,
        terminated: bool,
        truncated: bool,
    ) -> List[NStepTransition]:
        self.pending.append(
            _PendingTransition(observation, action, reward, next_observation, terminated, terminated or truncated)
        )
        ready: List[NStepTransition] = []
        if terminated or truncated:
            while self.pending:
                ready.append(self._emit())
        elif len(self.pending) >= self.steps:
            ready.append(self._emit())
        return ready

    def _emit(self) -> NStepTransition:
        first = self.pending[0]
        total_reward = 0.0
        discount = 1.0
        next_observation = first.next_observation
        terminated = False
        for index, transition in enumerate(self.pending):
            if index >= self.steps:
                break
            total_reward += discount * transition.reward
            discount *= self.gamma
            next_observation = transition.next_observation
            terminated = transition.terminated
            if transition.episode_end:
                break
        self.pending.popleft()
        return first.observation, first.action, total_reward, next_observation, terminated, discount
