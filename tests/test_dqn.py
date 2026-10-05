import unittest

import numpy as np
import torch

from asteroids_ai.dqn import DQNAgent, DQNConfig
from asteroids_ai.env import AsteroidsEnv
from asteroids_ai.replay import ReplayBuffer


class DQNTests(unittest.TestCase):
    def test_training_step_is_finite_updates_weights_and_detaches_target(self) -> None:
        env = AsteroidsEnv()
        observation, _ = env.reset(seed=9)
        config = DQNConfig(batch_size=4, warmup_steps=4, target_sync_updates=100)
        agent = DQNAgent(env, "mlp", config, device="cpu", seed=9)
        replay = ReplayBuffer(8, 7, 22, 9, prioritized=True, seed=9)
        for index in range(4):
            replay.add(observation, index, float(index), observation, False, config.gamma)
        before = [parameter.detach().clone() for parameter in agent.online.parameters()]
        metrics = agent.train_batch(replay.sample(4, beta=0.4), replay)
        self.assertTrue(np.isfinite(metrics["loss"]))
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, agent.online.parameters())))
        self.assertTrue(all(parameter.grad is None for parameter in agent.target.parameters()))

    def test_epsilon_schedule(self) -> None:
        env = AsteroidsEnv()
        config = DQNConfig(epsilon_decay_steps=100)
        agent = DQNAgent(env, config=config, device="cpu")
        self.assertAlmostEqual(agent.epsilon(0), 1.0)
        self.assertAlmostEqual(agent.epsilon(50), 0.525)
        self.assertAlmostEqual(agent.epsilon(100), 0.05)
        self.assertAlmostEqual(agent.epsilon(1_000), 0.05)

    def test_checkpoint_round_trip_preserves_policy(self) -> None:
        env = AsteroidsEnv()
        observation, _ = env.reset(seed=12)
        agent = DQNAgent(env, "mlp", device="cpu", seed=12)
        state = agent.checkpoint(123, env)
        restored = DQNAgent(env, "mlp", device="cpu", seed=99)
        step = restored.load_checkpoint(state)
        self.assertEqual(step, 123)
        with torch.no_grad():
            original = agent.online(
                {key: torch.as_tensor(value).unsqueeze(0) for key, value in observation.items()}
            )
            reloaded = restored.online(
                {key: torch.as_tensor(value).unsqueeze(0) for key, value in observation.items()}
            )
        torch.testing.assert_close(original, reloaded)

    def test_transformer_uses_the_shared_learner(self) -> None:
        env = AsteroidsEnv()
        observation, _ = env.reset(seed=21)
        config = DQNConfig(batch_size=2, target_sync_updates=10)
        agent = DQNAgent(env, "transformer", config, device="cpu", seed=21)
        replay = ReplayBuffer(4, 7, 22, 9, seed=21)
        replay.add(observation, 0, 1.0, observation, False, config.gamma)
        replay.add(observation, 1, -1.0, observation, False, config.gamma)
        metrics = agent.train_batch(replay.sample(2), replay)
        self.assertTrue(np.isfinite(metrics["loss"]))


if __name__ == "__main__":
    unittest.main()
