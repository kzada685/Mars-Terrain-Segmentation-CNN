"""Data invariants checked on small synthetic images, without training."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from src.dataset import TerrainDataset, crop_origin, load_manifest
from src.prepare import grouped_splits


class DataTests(unittest.TestCase):
    def test_groups_are_reproducible_and_disjoint(self):
        records = [{"sequence": str(i // 3), "mask": str(i)} for i in range(90)]
        first = grouped_splits(records)
        self.assertEqual(first, grouped_splits(records))
        groups = {}
        for record in first:
            groups.setdefault(record["sequence"], set()).add(record["split"])
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))
        self.assertEqual({r["split"] for r in first}, {"train", "val", "test"})

    def test_sparse_crop_and_all_ignore(self):
        mask = np.full((300, 300), 255, dtype=np.uint8)
        with self.assertRaises(ValueError):
            crop_origin(mask, 256, np.random.default_rng(2))
        mask[299, 299] = 3
        top, left = crop_origin(mask, 256, np.random.default_rng(2))
        self.assertTrue(np.any(mask[top:top + 256, left:left + 256] == 3))

    def test_tiles_augmentation_and_leakage(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            image = np.zeros((260, 270), dtype=np.uint8)
            image[:, 135:] = 255
            mask = (image > 0).astype(np.uint8)
            records = []
            for i, split in enumerate(("train", "val", "test")):
                Image.fromarray(image).save(root / f"image{i}.png")
                Image.fromarray(mask).save(root / f"mask{i}.png")
                fingerprint = lambda a: hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()
                records.append(dict(image=f"image{i}.png", mask=f"mask{i}.png",
                                    sequence=str(i), image_sha256=fingerprint(image),
                                    mask_sha256=fingerprint(mask), split=split,
                                    shape=[260, 270], review_required=False))
            # Distinct source images, with the same synthetic segmentation pattern.
            for i, record in enumerate(records):
                variant = image.copy()
                variant[0, 0] = i
                Image.fromarray(variant).save(root / record["image"])
                record["image_sha256"] = fingerprint(variant)
            path = root / "manifest.json"
            path.write_text(json.dumps(dict(dataset_root=".", records=records)))
            train = TerrainDataset(path, "train", augment=True)
            x, y, _ = train.arrays(0)
            self.assertEqual(x.shape, (1, 256, 256))
            self.assertEqual(y.dtype, np.int64)
            np.testing.assert_array_equal(x[0] > 0, y > 0)
            np.testing.assert_array_equal(x, train.arrays(0)[0])
            train.set_epoch(1)
            self.assertFalse(np.array_equal(x, train.arrays(0)[0]))
            val = TerrainDataset(path, "val")
            self.assertEqual(len(val), 4)
            reconstructed = np.zeros_like(mask)
            for index in range(len(val)):
                x, y, info = val.arrays(index)
                h, w, t, l = (info[k] for k in ("height", "width", "top", "left"))
                reconstructed[t:t+h, l:l+w] = y[:h, :w]
                self.assertTrue(np.all(y[h:] == 255))
                self.assertTrue(np.all(y[:, w:] == 255))
            np.testing.assert_array_equal(reconstructed, mask)
            with self.assertRaises(ValueError):
                TerrainDataset(path, "val", augment=True)
            changed = mask.copy()
            changed[0, 0] = 3
            Image.fromarray(changed).save(root / records[1]["mask"])
            with self.assertRaises(ValueError):
                val.arrays(0)
            records[1]["sequence"] = "0"
            path.write_text(json.dumps(dict(dataset_root=".", records=records)))
            with self.assertRaises(ValueError):
                load_manifest(path)


if __name__ == "__main__":
    unittest.main()
