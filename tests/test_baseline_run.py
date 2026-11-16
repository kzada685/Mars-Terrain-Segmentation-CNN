"""Check sampling and validation counting without fitting a forest."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from src.prepare import pixel_fingerprint
from src.run_baseline import accuracy_counts, sample_training_pixels, validate


def save_pair(root, number, mask, split):
    image = np.arange(mask.size, dtype=np.uint8).reshape(mask.shape)
    image_name = f"image{number}.png"
    mask_name = f"mask{number}.png"
    Image.fromarray(image).save(root / image_name)
    Image.fromarray(mask).save(root / mask_name)
    return {"image": image_name, "mask": mask_name, "split": split,
            "image_sha256": pixel_fingerprint(image),
            "mask_sha256": pixel_fingerprint(mask)}


class PredictSoil:
    def predict(self, features):
        return np.zeros(len(features), dtype=np.int64)


class BaselineRunTests(unittest.TestCase):
    def test_training_sample_is_repeatable_and_ignores_unlabeled_pixels(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            mask = np.tile(np.array([0, 1, 2, 3], dtype=np.uint8), (10, 10))
            mask[0, 0] = 255
            record = save_pair(root, 0, mask, "train")
            features, labels = sample_training_pixels(root, [record], 100, 360)
            repeated_features, repeated_labels = sample_training_pixels(root, [record], 100, 360)
            self.assertEqual(features.shape, (100, 2))
            self.assertEqual(set(labels), {0, 1, 2, 3})
            np.testing.assert_array_equal(features, repeated_features)
            np.testing.assert_array_equal(labels, repeated_labels)
            _, all_labels = sample_training_pixels(root, [record], 1000, 360)
            self.assertEqual(len(all_labels), mask.size - 1)
            record["split"] = "val"
            with self.assertRaises(ValueError):
                sample_training_pixels(root, [record])

    def test_validation_pools_pixel_counts_and_saves_class_masks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = save_pair(root, 0, np.array([[0, 0, 255]], dtype=np.uint8), "val")
            second = save_pair(root, 1, np.array([[1, 255]], dtype=np.uint8), "val")
            result = validate(PredictSoil(), root, [first, second], root)
            self.assertEqual(result["correct_pixels"], 2)
            self.assertEqual(result["labeled_pixels"], 3)
            self.assertAlmostEqual(result["accuracy"], 2 / 3)
            resumed = validate(PredictSoil(), root, [first, second], root)
            self.assertEqual(resumed, result)
            with Image.open(root / "examples/example_1_true.png") as source:
                self.assertEqual(np.array(source)[0, 2], 255)
            prediction = np.array([[0, 1]])
            self.assertEqual(accuracy_counts(prediction, np.array([[255, 255]])), (0, 0))


if __name__ == "__main__":
    unittest.main()
