"""Freeze grouped source-image splits and summarize their original labels.

Run from the project directory with ``python -m src.prepare --help``.
Raw files and the existing filtered sample list are never modified.
"""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random

import numpy as np
from PIL import Image

from .data import DATA_DIRECTORY, IGNORE_LABEL, TERRAIN_CLASSES


def pixel_fingerprint(array):
    """Identify the pixel contents so changed or duplicate files can be caught."""
    contents = str(array.shape).encode() + array.tobytes()
    return hashlib.sha256(contents).hexdigest()


def read_pair(root, record):
    """Decode grayscale input and integer labels without resampling either."""
    root = Path(root).resolve()
    paths = [(root / record[key]).resolve() for key in ("image", "mask")]
    if any(not path.is_relative_to(root) for path in paths):
        raise ValueError("Sample path escapes dataset root")
    with Image.open(paths[0]) as source:
        image = np.array(source.convert("L"))
    with Image.open(paths[1]) as source:
        mask = np.array(source)
    if mask.ndim == 3 and np.all(mask == mask[..., :1]):
        mask = mask[..., 0]
    if mask.ndim != 2 or image.shape != mask.shape:
        raise ValueError("Expected matching two-dimensional image and mask")
    if not np.isin(mask, [0, 1, 2, 3, IGNORE_LABEL]).all():
        raise ValueError("Unexpected mask label")
    return image, mask


def grouped_splits(records, fractions=(0.7, 0.15, 0.15), seed=360):
    """Allocate whole command groups, approximating target image counts."""
    if len(fractions) != 3 or min(fractions) <= 0 or not np.isclose(sum(fractions), 1):
        raise ValueError("Three positive split fractions must sum to one")
    groups = {}
    for record in records:
        sequence = record["sequence"]
        if sequence not in groups:
            groups[sequence] = []
        groups[sequence].append(record)
    if len(groups) < 3:
        raise ValueError("At least three independent sequence groups are required")
    names = list(sorted(groups))
    random.Random(seed).shuffle(names)
    # Place large groups first; shuffle supplies deterministic random tie breaks.
    names.sort(key=lambda name: len(groups[name]), reverse=True)
    split_names = ["train", "val", "test"]
    targets = [fraction * len(records) for fraction in fractions]
    counts = [0, 0, 0]
    assigned = []
    for name in names:
        # Choose the split furthest below its requested proportion.
        destination = 0
        largest_gap = (targets[0] - counts[0]) / targets[0]
        for i in range(1, 3):
            gap = (targets[i] - counts[i]) / targets[i]
            if gap > largest_gap:
                destination = i
                largest_gap = gap
        split = split_names[destination]
        for record in groups[name]:
            sample = record.copy()
            sample["split"] = split
            assigned.append(sample)
        counts[destination] += len(groups[name])
    if 0 in counts:
        raise ValueError("Group sizes prevent nonempty splits; review allocation")
    return sorted(assigned, key=lambda record: record["mask"])


def inspect_record(root, record):
    image, mask = read_pair(root, record)
    if min(image.shape) < 256 or not np.any(mask != IGNORE_LABEL):
        raise ValueError(f"Unusable sample: {record['mask']}")
    histogram = np.bincount(mask.ravel(), minlength=256)
    shape = list(image.shape)
    image_hash = pixel_fingerprint(image)
    mask_hash = pixel_fingerprint(mask)
    pixel_counts = histogram[[0, 1, 2, 3, IGNORE_LABEL]]
    return shape, image_hash, mask_hash, pixel_counts


def prepare(filtered_path, root, fractions, seed, exclude_reviewed):
    source = json.loads(Path(filtered_path).read_text(encoding="utf-8"))
    pending = [r for r in source["samples"] if r["review_required"]]
    if pending and not exclude_reviewed:
        raise ValueError("Unresolved annotations: explicitly select --exclude-reviewed")
    records = [r for r in source["samples"] if not r["review_required"]]
    records = grouped_splits(records, fractions, seed)
    counts = {s: np.zeros(5, dtype=np.int64) for s in ("train", "val", "test")}
    presence = {s: np.zeros(4, dtype=np.int64) for s in counts}
    image_splits = {}
    shapes = Counter()
    # Process one image/mask pair at a time, in a fixed order.
    for index, record in enumerate(records, 1):
        shape, image_hash, mask_hash, pixels = inspect_record(root, record)
        if image_hash in image_splits:
            raise ValueError("Duplicate image remains; review before freezing splits")
        split = record["split"]
        image_splits[image_hash] = split
        record["shape"] = shape
        record["image_sha256"] = image_hash
        record["mask_sha256"] = mask_hash
        counts[split] += pixels
        presence[split] += pixels[:4] > 0
        shapes[str(tuple(shape))] += 1
        if index % 1000 == 0:
            print(f"Summarized {index}/{len(records)}", flush=True)
    summary = {}
    for split in counts:
        pixels = counts[split]
        if np.any(pixels[:4] == 0):
            raise ValueError(f"A terrain class is absent from {split}; review grouped allocation")
        labeled = int(pixels[:4].sum())
        split_records = [record for record in records if record["split"] == split]
        class_pixels = {}
        class_images = {}
        class_fractions = {}
        for class_id, class_name in enumerate(TERRAIN_CLASSES):
            class_pixels[class_name] = int(pixels[class_id])
            class_images[class_name] = int(presence[split][class_id])
            class_fractions[class_name] = int(pixels[class_id]) / labeled
        summary[split] = {
            "images": len(split_records),
            "sequences": len({record["sequence"] for record in split_records}),
            "pixels_by_class": class_pixels,
            "images_containing_class": class_images,
            "labeled_pixel_fractions": class_fractions,
            "ignored_pixels": int(pixels[4]),
        }
    return {
        "version": 1, "dataset_root": "raw/msl/ncam", "seed": seed,
        "requested_fractions": dict(zip(("train", "val", "test"), fractions)),
        "grouping": "command_sequence; groups never cross splits",
        "filtered_samples_sha256": hashlib.sha256(Path(filtered_path).read_bytes()).hexdigest(),
        "excluded_pending_review": pending,
        "expert_test": "Reserved separately; not assigned or used in these splits",
        "crop_size": 256,
        "training_augmentation": {"horizontal_flip_probability": 0.5,
                                  "brightness_multiplier_range": [0.9, 1.1]},
        "summary": summary, "source_shapes": dict(shapes),
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtered", type=Path, default=DATA_DIRECTORY.parents[2] / "filtered_samples.json")
    parser.add_argument("--root", type=Path, default=DATA_DIRECTORY)
    parser.add_argument("--output", type=Path, default=DATA_DIRECTORY.parents[2] / "prepared_splits.json")
    parser.add_argument("--fractions", nargs=3, type=float, default=(0.7, 0.15, 0.15))
    parser.add_argument("--seed", type=int, default=360)
    parser.add_argument("--exclude-reviewed", action="store_true", help="Explicitly quarantine all pending annotation reviews")
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.root.resolve()):
        parser.error("Choose a new output outside the raw dataset; frozen files are not overwritten")
    result = prepare(args.filtered, args.root, args.fractions, args.seed, args.exclude_reviewed)
    result["dataset_root"] = Path(os.path.relpath(args.root.resolve(), args.output.parent.resolve())).as_posix()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
