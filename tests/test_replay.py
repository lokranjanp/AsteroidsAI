import unittest

import numpy as np

from asteroids_ai.env import AsteroidsEnv
from asteroids_ai.replay import NStepAccumulator, ReplayBuffer


class ReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = AsteroidsEnv()
        self.observation, _ = self.env.reset(seed=1)

    def test_three_step_return(self) -> None:
        accumulator = NStepAccumulator(3, 0.9)
        ready = []
        for reward in (1.0, 2.0, 3.0):
            ready.extend(accumulator.append(self.observation, 0, reward, self.observation, False, False))
        self.assertEqual(len(ready), 1)
        self.assertAlmostEqual(ready[0][2], 1.0 + 0.9 * 2.0 + 0.9**2 * 3.0)
        self.assertAlmostEqual(ready[0][5], 0.9**3)

    def test_episode_end_flushes_pending_transitions(self) -> None:
        accumulator = NStepAccumulator(3, 0.9)
        self.assertEqual(accumulator.append(self.observation, 0, 1.0, self.observation, False, False), [])
        ready = accumulator.append(self.observation, 1, -2.0, self.observation, True, False)
        self.assertEqual(len(ready), 2)
        self.assertTrue(ready[0][4])
        self.assertTrue(ready[1][4])

    def test_prioritized_buffer_shapes_and_updates(self) -> None:
        buffer = ReplayBuffer(16, 7, 22, 9, prioritized=True, seed=5)
        for index in range(8):
            buffer.add(self.observation, index % 6, float(index), self.observation, False, 0.99)
        sample = buffer.sample(4, beta=0.4)
        self.assertEqual(sample.observations["entities"].shape, (4, 22, 9))
        self.assertEqual(sample.actions.shape, (4,))
        buffer.update_priorities(sample.indices, np.full((4,), 3.0, dtype=np.float32))
        np.testing.assert_array_equal(buffer.priorities[sample.indices], 3.0)

    def test_metadata_only_resume_rewarms_an_empty_buffer(self) -> None:
        source = ReplayBuffer(16, 7, 22, 9, seed=5)
        source.add(self.observation, 0, 1.0, self.observation, False, 0.99)
        restored = ReplayBuffer(16, 7, 22, 9, seed=7)
        restored.load_state_dict(source.state_dict(include_data=False))
        self.assertEqual(len(restored), 0)
        self.assertEqual(restored.position, 0)


if __name__ == "__main__":
    unittest.main()
