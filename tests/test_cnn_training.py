"""Check loss masking and weight updates using tiny synthetic batches."""

import unittest

import torch
from torch import nn

from src.train_cnn import run_epoch


class CNNTrainingTests(unittest.TestCase):
    def test_validation_ignores_unlabeled_pixels_and_does_not_update_weights(self):
        model = nn.Conv2d(1, 4, kernel_size=1)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()
        batches = [
            {"image": torch.ones(1, 1, 2, 2),
             "mask": torch.tensor([[[0, 1], [255, 255]]])},
            {"image": torch.ones(1, 1, 2, 2),
             "mask": torch.full((1, 2, 2), 255, dtype=torch.long)},
        ]
        before = model.weight.detach().clone()
        result = run_epoch(model, batches, torch.device("cpu"))
        self.assertEqual(result["labeled_pixels"], 2)
        self.assertEqual(result["correct_pixels"], 1)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["all_ignore_batches_skipped"], 1)
        self.assertAlmostEqual(result["loss"], 1.38629436, places=5)
        self.assertTrue(torch.equal(before, model.weight))

    def test_training_updates_weights(self):
        model = nn.Conv2d(1, 4, kernel_size=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        batches = [{"image": torch.ones(1, 1, 2, 2),
                    "mask": torch.zeros(1, 2, 2, dtype=torch.long)}]
        before = model.weight.detach().clone()
        result = run_epoch(model, batches, torch.device("cpu"), optimizer)
        self.assertEqual(result["labeled_pixels"], 4)
        self.assertFalse(torch.equal(before, model.weight))


if __name__ == "__main__":
    unittest.main()
