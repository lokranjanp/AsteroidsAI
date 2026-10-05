import unittest

import torch

from asteroids_ai.models import EntityTransformerQNetwork, MLPQNetwork


def observation(batch: int = 2):
    return {
        "global": torch.randn(batch, 7),
        "entities": torch.randn(batch, 22, 9),
        "entity_mask": torch.ones(batch, 22),
    }


class ModelTests(unittest.TestCase):
    def test_model_output_shapes(self) -> None:
        sample = observation()
        self.assertEqual(MLPQNetwork(7, 22, 9, 6)(sample).shape, (2, 6))
        transformer = EntityTransformerQNetwork(7, 9, 6)
        transformer.eval()
        self.assertEqual(transformer(sample).shape, (2, 6))

    def test_transformer_does_not_mix_batch_samples(self) -> None:
        torch.manual_seed(2)
        model = EntityTransformerQNetwork(7, 9, 6)
        model.eval()
        sample = observation()
        with torch.no_grad():
            first = model(sample)[0]
            sample["global"][1] *= 100.0
            sample["entities"][1] *= -50.0
            second = model(sample)[0]
        torch.testing.assert_close(first, second)

    def test_masked_entities_do_not_change_prediction(self) -> None:
        torch.manual_seed(3)
        model = EntityTransformerQNetwork(7, 9, 6)
        model.eval()
        sample = observation(batch=1)
        sample["entity_mask"][:, 5:] = 0
        with torch.no_grad():
            first = model(sample)
            sample["entities"][:, 5:] = torch.randn_like(sample["entities"][:, 5:]) * 1_000
            second = model(sample)
        torch.testing.assert_close(first, second)


if __name__ == "__main__":
    unittest.main()

