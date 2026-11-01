"""Check the network's input/output interface without training it."""

import unittest

import torch

from src.model import TerrainCNN


class TerrainCNNTests(unittest.TestCase):
    def test_forward_pass_preserves_image_size_and_outputs_four_classes(self):
        model = TerrainCNN()
        model.eval()
        image = torch.zeros(1, 1, 256, 256)

        with torch.inference_mode():
            scores = model(image)

        self.assertEqual(tuple(scores.shape), (1, 4, 256, 256))
        self.assertTrue(torch.isfinite(scores).all().item())


if __name__ == "__main__":
    unittest.main()
