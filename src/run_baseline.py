"""Fit one Random Forest and measure labeled-pixel validation accuracy.

From the repository folder: python -m src.run_baseline
This performs no tuning and never reads internal-test image or mask files.
"""

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from PIL import Image

from .baseline import build_baseline, pixel_features
from .data import IGNORE_LABEL, TERRAIN_CLASSES
from .dataset import load_manifest
from .prepare import pixel_fingerprint, read_pair


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]


def read_frozen_pair(root, record):
    """Check that the source still matches the data used to freeze the split."""
    image, mask = read_pair(root, record)
    if pixel_fingerprint(image) != record["image_sha256"]:
        raise ValueError(f"Image changed: {record['image']}")
    if pixel_fingerprint(mask) != record["mask_sha256"]:
        raise ValueError(f"Mask changed: {record['mask']}")
    return image, mask


def sample_training_pixels(root, records, pixels_per_image=100, seed=360):
    """Sample labeled pixels without replacement from each training image."""
    if pixels_per_image < 1:
        raise ValueError("Pixels per image must be positive")
    rng = np.random.default_rng(seed)
    feature_samples = []
    label_samples = []

    for index, record in enumerate(records, 1):
        if record["split"] != "train":
            raise ValueError("Only training records may supply fitting pixels")
        image, mask = read_frozen_pair(root, record)
        labels = mask.ravel()
        labeled_indices = np.flatnonzero(labels != IGNORE_LABEL)
        if len(labeled_indices) == 0:
            raise ValueError(f"No labeled pixels: {record['mask']}")

        sample_size = min(pixels_per_image, len(labeled_indices))
        selected_indices = rng.choice(labeled_indices, size=sample_size, replace=False)
        features = pixel_features(image)
        feature_samples.append(features[selected_indices])
        label_samples.append(labels[selected_indices])
        if index % 100 == 0:
            print(f"Sampled {index}/{len(records)} training images", flush=True)

    if not feature_samples:
        raise ValueError("No training images supplied")
    # Each row has two features, and its target is one terrain class.
    training_features = np.concatenate(feature_samples, axis=0)
    training_labels = np.concatenate(label_samples).astype(np.int64)
    if set(np.unique(training_labels)) != {0, 1, 2, 3}:
        raise ValueError("The sampled training pixels must contain all four classes")
    return training_features, training_labels


def accuracy_counts(prediction, mask):
    """Count correct predictions and labeled pixels; label 255 is ignored."""
    if prediction.shape != mask.shape:
        raise ValueError("Prediction and mask dimensions must match")
    valid = mask != IGNORE_LABEL
    correct = int(np.count_nonzero(prediction[valid] == mask[valid]))
    total = int(np.count_nonzero(valid))
    return correct, total


def validate(model, root, records, output_directory):
    """Predict every validation image and save the first three examples."""
    total_correct = 0
    total_labeled = 0
    examples = []
    examples_directory = output_directory / "examples"
    examples_directory.mkdir(exist_ok=True)
    progress_path = output_directory / "validation_progress.json"
    completed = []
    if progress_path.exists():
        completed = json.loads(progress_path.read_text(encoding="utf-8"))
    if len(completed) > len(records):
        raise ValueError("Saved validation progress does not match this split")

    for index, record in enumerate(records, 1):
        if record["split"] != "val":
            raise ValueError("This run evaluates validation data only")
        if index <= len(completed):
            saved = completed[index - 1]
            if saved["mask"] != record["mask"]:
                raise ValueError("Saved validation progress has different membership")
            total_correct += saved["correct_pixels"]
            total_labeled += saved["labeled_pixels"]
            if index <= 3:
                examples.append(saved["example"])
            continue
        image, mask = read_frozen_pair(root, record)
        features = pixel_features(image)
        prediction = model.predict(features).reshape(mask.shape)
        correct, labeled = accuracy_counts(prediction, mask)
        total_correct += correct
        total_labeled += labeled

        # Save label PNGs, not plots. Values 0..3 are the predicted classes.
        # The true mask retains 255 for unlabeled pixels.
        if index <= 3:
            prefix = f"example_{index}"
            Image.fromarray(image).save(examples_directory / f"{prefix}_image.png")
            Image.fromarray(mask).save(examples_directory / f"{prefix}_true.png")
            Image.fromarray(prediction.astype(np.uint8)).save(
                examples_directory / f"{prefix}_prediction.png"
            )
            examples.append({"prefix": prefix, "source_image": record["image"],
                             "source_mask": record["mask"], "correct_pixels": correct,
                             "labeled_pixels": labeled})
        saved = {"mask": record["mask"], "correct_pixels": correct, "labeled_pixels": labeled}
        if index <= 3:
            saved["example"] = examples[-1]
        completed.append(saved)
        # Replace only after a complete write, keeping interrupted runs resumable.
        temporary_path = output_directory / "validation_progress.tmp"
        temporary_path.write_text(json.dumps(completed), encoding="utf-8")
        temporary_path.replace(progress_path)
        print(f"Validated {index}/{len(records)} images; accuracy so far: "
              f"{total_correct / total_labeled:.2%}", flush=True)

    if total_labeled == 0:
        raise ValueError("Validation contains no labeled pixels")
    return {"accuracy": total_correct / total_labeled,
            "correct_pixels": total_correct, "labeled_pixels": total_labeled,
            "images": len(records), "examples": examples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=PROJECT_DIRECTORY / "data/prepared_splits.json")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_DIRECTORY / "outputs/random_forest_run1")
    args = parser.parse_args()
    # Refuse to overwrite a previous experiment or write into raw data.
    data_directory = (PROJECT_DIRECTORY / "data").resolve()
    if args.output.exists() or args.output.resolve().is_relative_to(data_directory):
        parser.error("Choose a new output folder outside data/")
    root, manifest = load_manifest(args.manifest)
    if args.output.resolve().is_relative_to(root.resolve()):
        parser.error("Output must be outside the raw dataset")
    training_records = []
    validation_records = []
    for record in manifest["records"]:
        if record["split"] == "train":
            training_records.append(record)
        elif record["split"] == "val":
            validation_records.append(record)

    seed = 360
    pixels_per_image = 100
    model = build_baseline(seed)
    args.output.mkdir(parents=True)
    import sklearn

    settings = {
        "seed": seed, "pixels_per_training_image": pixels_per_image,
        "training_images": len(training_records), "validation_images": len(validation_records),
        "features": ["grayscale intensity / 255", "3x3 local mean / 255"],
        "augmentation": False, "hyperparameter_tuning": False,
        "model_parameters": model.get_params(), "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__, "classes": list(TERRAIN_CLASSES),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "metric": "Total correct labeled pixels / total labeled pixels; ignore 255",
    }
    with (args.output / "settings.json").open("w", encoding="utf-8") as stream:
        json.dump(settings, stream, indent=2)

    print("Preparing training pixels...", flush=True)
    features, labels = sample_training_pixels(
        root, training_records, pixels_per_image, seed
    )
    settings["sampled_training_pixels"] = len(labels)
    settings["sampled_pixels_by_class"] = {}
    for class_id, class_name in enumerate(TERRAIN_CLASSES):
        settings["sampled_pixels_by_class"][class_name] = int(np.count_nonzero(labels == class_id))
    with (args.output / "settings.json").open("w", encoding="utf-8") as stream:
        json.dump(settings, stream, indent=2)
    print(f"Fitting one forest on {len(labels)} sampled pixels...", flush=True)
    model.fit(features, labels)
    # Save immediately so a later validation interruption does not lose the fit.
    with (args.output / "model.pkl").open("wb") as stream:
        pickle.dump(model, stream)
    del features
    del labels

    print("Measuring validation accuracy...", flush=True)
    results = validate(model, root, validation_records, args.output)
    with (args.output / "validation.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)
    print(f"Validation labeled-pixel accuracy: {results['accuracy']:.2%}")
    print(f"Saved model, settings, results and examples to {args.output}")


if __name__ == "__main__":
    main()
