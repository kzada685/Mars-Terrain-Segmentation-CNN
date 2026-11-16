"""On-demand 256-pixel crops from frozen source-image membership.

No crops are saved, no masks are interpolated, and only training is augmented.
The NumPy interface can also supply the Random Forest's pixel feature code.
"""

import json
from pathlib import Path

import numpy as np

from .data import IGNORE_LABEL
from .prepare import pixel_fingerprint, read_pair


def load_manifest(path):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    sequences = {}
    images = {}
    masks = set()
    for record in manifest["records"]:
        split = record["split"]
        if split not in {"train", "val", "test"} or record["review_required"]:
            raise ValueError("Invalid split or unresolved annotation")
        if record["mask"] in masks:
            raise ValueError("Repeated mask in manifest")
        masks.add(record["mask"])
        for registry, key in ((sequences, record["sequence"]), (images, record["image_sha256"])):
            if key in registry and registry[key] != split:
                raise ValueError("Image or command sequence crosses splits")
            registry[key] = split
    if set(sequences.values()) != {"train", "val", "test"}:
        raise ValueError("Expected three nonempty splits")
    return path.parent / manifest["dataset_root"], manifest


def crop_origin(mask, size, rng):
    height, width = mask.shape
    if min(height, width) < size:
        raise ValueError("Image is smaller than crop")
    for _ in range(32):
        top = int(rng.integers(height - size + 1))
        left = int(rng.integers(width - size + 1))
        if np.any(mask[top:top + size, left:left + size] != IGNORE_LABEL):
            return top, left
    valid = np.argwhere(mask != IGNORE_LABEL)
    if not len(valid):
        raise ValueError("All-ignore training sample")
    row, column = valid[int(rng.integers(len(valid)))]
    top = int(row) - size // 2
    left = int(column) - size // 2
    # Keep the crop inside the image when the labeled pixel is near an edge.
    top = min(max(top, 0), height - size)
    left = min(max(left, 0), width - size)
    return top, left


class TerrainDataset:
    """PyTorch-compatible map-style dataset, with a separately usable array API.

    Training: one random labeled crop per image per epoch, horizontal flip
    probability 0.5, brightness multiplier uniform in [0.9, 1.1]. These are
    initial preparation defaults, not tuned settings. Call set_epoch each epoch.
    Validation/test: complete nonoverlapping tile grid, including all-ignore
    tiles. Edge padding is zero for images and 255 for targets.
    """

    def __init__(self, manifest, split, *, augment=False, crop_size=256, seed=360):
        if split not in {"train", "val", "test"}:
            raise ValueError("Unknown split")
        if augment and split != "train":
            raise ValueError("Augmentation is allowed only for training")
        if not isinstance(crop_size, int) or crop_size <= 0:
            raise ValueError("Crop size must be a positive integer")
        self.root, source = load_manifest(manifest)
        self.records = [r for r in source["records"] if r["split"] == split]
        self.split = split
        self.augment = augment
        self.crop_size = crop_size
        self.seed = seed
        self.epoch = 0
        self.items = []
        for i, record in enumerate(self.records):
            height, width = record["shape"]
            if split == "train":
                if min(height, width) < crop_size:
                    raise ValueError("Training image is smaller than crop")
                self.items.append((i, None, None))
            else:
                for top in range(0, height, crop_size):
                    for left in range(0, width, crop_size):
                        self.items.append((i, top, left))

    def set_epoch(self, epoch):
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("Epoch must be a nonnegative integer")
        self.epoch = epoch

    def __len__(self):
        return len(self.items)

    def arrays(self, index):
        """Return float32 [1,H,W] input, int64 [H,W] target and tile metadata."""
        i, top, left = self.items[index]
        record = self.records[i]
        image, mask = read_pair(self.root, record)
        if list(image.shape) != record["shape"]:
            raise ValueError("Source dimensions changed since preparation")
        for key, array in (("image_sha256", image), ("mask_sha256", mask)):
            fingerprint = pixel_fingerprint(array)
            if fingerprint != record[key]:
                raise ValueError("Source pixels changed since preparation")
        # Each sample gets repeatable randomness, with a new crop each epoch.
        sample_seed = np.random.SeedSequence([self.seed, self.epoch, index])
        rng = np.random.default_rng(sample_seed)
        size = self.crop_size
        if self.split == "train":
            top, left = crop_origin(mask, size, rng)
        image = image[top:top + size, left:left + size]
        mask = mask[top:top + size, left:left + size]
        # PyTorch expects floating-point inputs and integer class labels.
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.int64)
        height, width = mask.shape
        padding = ((0, size - height), (0, size - width))
        image = np.pad(image, padding, constant_values=0)
        mask = np.pad(mask, padding, constant_values=IGNORE_LABEL)
        if self.augment:
            if rng.random() < 0.5:
                image = np.fliplr(image)
                mask = np.fliplr(mask)
            brightness = rng.uniform(0.9, 1.1)
            image = np.clip(image * brightness, 0, 1)
        # Add the single grayscale channel: [height, width] -> [1, height, width].
        # Copy after flipping because torch.from_numpy needs positive strides.
        image = np.expand_dims(image, axis=0).copy()
        mask = mask.copy()
        if image.dtype != np.float32 or not np.isfinite(image).all():
            raise ValueError("Invalid image array")
        return image, mask, {"image": record["image"], "mask": record["mask"],
                             "top": top, "left": left, "height": height, "width": width}

    def __getitem__(self, index):
        """PyTorch calls this method to load one sample by its index."""
        import torch

        image, mask, metadata = self.arrays(index)
        sample = metadata.copy()
        sample["source_image"] = sample.pop("image")
        sample["source_mask"] = sample.pop("mask")
        sample["image"] = torch.from_numpy(image)
        sample["mask"] = torch.from_numpy(mask)
        return sample
