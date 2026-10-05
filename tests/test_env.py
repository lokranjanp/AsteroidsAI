import unittest

import numpy as np

from asteroids_ai.env import Action, AsteroidsEnv, EnvConfig, _Entity, _Pickup


class AsteroidsEnvTests(unittest.TestCase):
    def test_reset_starts_clean_and_returns_valid_observation(self) -> None:
        env = AsteroidsEnv()
        observation, info = env.reset(seed=7)
        self.assertEqual(info["score"], 0)
        self.assertEqual(info["hits"], 0)
        self.assertEqual(observation["global"].shape, (7,))
        self.assertEqual(observation["entities"].shape, (22, 9))
        self.assertEqual(observation["entity_mask"].sum(), 0)
        self.assertAlmostEqual(float(observation["global"][2]), 1.0)

        env.bullets.append(_Entity(10, 10, 0, -1, 3))
        env.asteroids.append(_Entity(20, 20, 0, 1, 20))
        env.pickups.append(_Pickup(30, 30, 0, 1, 10, "fuel"))
        env.reset(seed=7)
        self.assertEqual(env.bullets, [])
        self.assertEqual(env.asteroids, [])
        self.assertEqual(env.pickups, [])

    def test_seeded_trajectories_are_identical(self) -> None:
        first, second = AsteroidsEnv(), AsteroidsEnv()
        first_observation, _ = first.reset(seed=42)
        second_observation, _ = second.reset(seed=42)
        for key in first_observation:
            np.testing.assert_array_equal(first_observation[key], second_observation[key])
        for step in range(120):
            action = step % len(Action)
            first_result = first.step(action)
            second_result = second.step(action)
            for key in first_result[0]:
                np.testing.assert_array_equal(first_result[0][key], second_result[0][key])
            self.assertEqual(first_result[1:4], second_result[1:4])
            self.assertEqual(first_result[4], second_result[4])

    def test_reward_is_per_transition_not_cumulative(self) -> None:
        env = AsteroidsEnv()
        env.reset(seed=1)
        env.bullets.append(_Entity(100.0, 100.0, 0.0, -5.5, 3.0))
        env.asteroids.append(_Entity(100.0, 92.5, 0.0, 2.0, 20.0))
        _, first_reward, _, _, first_info = env.step(Action.IDLE)
        _, second_reward, _, _, second_info = env.step(Action.IDLE)
        self.assertAlmostEqual(first_reward, 1.001)
        self.assertAlmostEqual(first_info["reward_components"]["asteroid_destroyed"], 1.0)
        self.assertAlmostEqual(second_reward, 0.001)
        self.assertAlmostEqual(second_info["reward_components"]["asteroid_destroyed"], 0.0)

    def test_combined_action_moves_and_fires(self) -> None:
        env = AsteroidsEnv()
        env.reset(seed=2)
        start_x = env.ship.x
        env.step(Action.LEFT_FIRE)
        self.assertLess(env.ship.x, start_x)
        self.assertEqual(env.shots, 1)
        self.assertEqual(len(env.bullets), 1)

    def test_time_limit_is_truncation(self) -> None:
        env = AsteroidsEnv(EnvConfig(max_steps=2))
        env.reset(seed=3)
        _, _, terminated, truncated, _ = env.step(Action.IDLE)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        _, _, terminated, truncated, info = env.step(Action.IDLE)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["death_reason"], "time_limit")
        with self.assertRaises(RuntimeError):
            env.step(Action.IDLE)

    def test_headless_steps_do_not_initialize_pygame(self) -> None:
        env = AsteroidsEnv(render_mode=None)
        env.reset(seed=4)
        env.step(Action.IDLE)
        self.assertIsNone(env._pygame)

    def test_rgb_render_returns_frame(self) -> None:
        env = AsteroidsEnv(render_mode="rgb_array")
        env.reset(seed=5)
        frame = env.render()
        self.assertEqual(frame.shape, (600, 600, 3))
        env.close()

    def test_terminal_penalty_is_applied_once(self) -> None:
        env = AsteroidsEnv()
        env.reset(seed=6)
        env.ship.health = 0.0
        _, reward, terminated, truncated, info = env.step(Action.IDLE)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertAlmostEqual(reward, -2.0)
        self.assertAlmostEqual(info["reward_components"]["death"], -2.0)


if __name__ == "__main__":
    unittest.main()
