"""Compatibility launcher for AsteroidsAI v2 training."""

from asteroids_ai.dqn import DQNAgent, DQNConfig
from asteroids_ai.train import main, train

__all__ = ["DQNAgent", "DQNConfig", "train"]


if __name__ == "__main__":
    main()
