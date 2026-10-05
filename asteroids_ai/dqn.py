from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn

from .env import AsteroidsEnv, Observation
from .models import build_q_network
from .replay import ReplayBuffer, ReplaySample


@dataclass(frozen=True)
class DQNConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 128
    replay_capacity: int = 100_000
    warmup_steps: int = 5_000
    train_frequency: int = 4
    target_sync_updates: int = 2_000
    gradient_clip: float = 10.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000
    n_step: int = 3
    prioritized_replay: bool = True
    priority_alpha: float = 0.6
    priority_beta_start: float = 0.4
    priority_beta_steps: int = 500_000


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def observation_to_tensors(observation: Observation, device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(value, dtype=torch.float32, device=device).unsqueeze(0)
        for key, value in observation.items()
    }


def batch_to_tensors(observation: Dict[str, np.ndarray], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value, dtype=torch.float32, device=device) for key, value in observation.items()}


class DQNAgent:
    def __init__(
        self,
        env: AsteroidsEnv,
        model_type: str = "mlp",
        config: Optional[DQNConfig] = None,
        device: str = "auto",
        seed: int = 0,
    ) -> None:
        self.config = config or DQNConfig()
        self.model_type = model_type
        self.device = resolve_device(device)
        self.rng = np.random.default_rng(seed)
        self.action_count = int(env.action_space.n)
        torch.manual_seed(seed)
        dimensions = (
            env.GLOBAL_FEATURES,
            env.config.max_entities,
            env.ENTITY_FEATURES,
            self.action_count,
        )
        self.online = build_q_network(model_type, *dimensions).to(self.device)
        self.target = build_q_network(model_type, *dimensions).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.AdamW(self.online.parameters(), lr=self.config.learning_rate)
        self.learner_updates = 0

    def epsilon(self, global_step: int) -> float:
        fraction = min(max(global_step, 0) / self.config.epsilon_decay_steps, 1.0)
        return self.config.epsilon_start + fraction * (self.config.epsilon_end - self.config.epsilon_start)

    def priority_beta(self, global_step: int) -> float:
        fraction = min(max(global_step, 0) / self.config.priority_beta_steps, 1.0)
        return self.config.priority_beta_start + fraction * (1.0 - self.config.priority_beta_start)

    def act(self, observation: Observation, epsilon: float = 0.0) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.action_count))
        self.online.eval()
        with torch.no_grad():
            q_values = self.online(observation_to_tensors(observation, self.device))
        self.online.train()
        return int(q_values.argmax(dim=1).item())

    def train_batch(self, sample: ReplaySample, replay: Optional[ReplayBuffer] = None) -> Dict[str, float]:
        observations = batch_to_tensors(sample.observations, self.device)
        next_observations = batch_to_tensors(sample.next_observations, self.device)
        actions = torch.as_tensor(sample.actions, dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor(sample.rewards, dtype=torch.float32, device=self.device)
        terminated = torch.as_tensor(sample.terminated, dtype=torch.float32, device=self.device)
        discounts = torch.as_tensor(sample.discounts, dtype=torch.float32, device=self.device)
        weights = torch.as_tensor(sample.weights, dtype=torch.float32, device=self.device)

        predicted = self.online(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.online(next_observations).argmax(dim=1, keepdim=True)
            next_values = self.target(next_observations).gather(1, next_actions).squeeze(1)
            targets = rewards + discounts * (1.0 - terminated) * next_values

        td_error = targets - predicted
        per_item_loss = functional.smooth_l1_loss(predicted, targets, reduction="none")
        loss = (per_item_loss * weights).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(nn.utils.clip_grad_norm_(self.online.parameters(), self.config.gradient_clip))
        self.optimizer.step()
        self.learner_updates += 1
        if self.learner_updates % self.config.target_sync_updates == 0:
            self.target.load_state_dict(self.online.state_dict())
        if replay is not None:
            replay.update_priorities(sample.indices, td_error.detach().abs().cpu().numpy() + 1e-6)
        return {
            "loss": float(loss.detach().cpu()),
            "mean_q": float(predicted.detach().mean().cpu()),
            "mean_target": float(targets.detach().mean().cpu()),
            "gradient_norm": gradient_norm,
        }

    def checkpoint(
        self,
        global_step: int,
        env: AsteroidsEnv,
        replay: Optional[ReplayBuffer] = None,
        include_replay_data: bool = False,
    ) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "version": 2,
            "model_type": self.model_type,
            "dqn_config": asdict(self.config),
            "env_config": env.config_dict(),
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "learner_updates": self.learner_updates,
            "global_step": global_step,
            "agent_rng_state": self.rng.bit_generator.state,
            "env_rng_state": env.rng_state(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        if replay is not None:
            state["replay"] = replay.state_dict(include_data=include_replay_data)
        return state

    def save(
        self,
        path: Path,
        global_step: int,
        env: AsteroidsEnv,
        replay: Optional[ReplayBuffer] = None,
        include_replay_data: bool = False,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint(global_step, env, replay, include_replay_data), path)

    def load_checkpoint(self, state: Dict[str, Any], replay: Optional[ReplayBuffer] = None) -> int:
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.learner_updates = int(state.get("learner_updates", 0))
        self.rng.bit_generator.state = state["agent_rng_state"]
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available() and "cuda_rng_state" in state:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        if replay is not None and "replay" in state:
            replay.load_state_dict(state["replay"])
        return int(state.get("global_step", 0))
