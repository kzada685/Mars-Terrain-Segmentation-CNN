"""Locate the Curiosity NavCam image/mask pairs for the initial CNN.

The selected v0.6 training pool contains 16,064 candidate labeled images.
Final usable counts will be established during the later cleaning step.
Expert-test masks are kept separate and are not included here.
"""

from pathlib import Path
import re


DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data/raw/msl/ncam"
TERRAIN_CLASSES = ("soil", "bedrock", "sand", "large_rock")
IGNORE_LABEL = 255


def command_sequence(image_name):
    """Return the group ID that must stay together when splits are later made.

    Both camera eyes and consecutive frames from a command share this ID.
    An unknown filename is rejected rather than assigned a random group.
    """
    match = re.fullmatch(
        r"N[LR][AB]_\d{9}EDR_[A-Z]\d{7}([A-Z_]{4}\d{5})[A-Z_]\d\.JPG",
        image_name.upper(),
    )
    if match is None:
        raise ValueError(f"Unrecognized Curiosity NavCam filename: {image_name}")
    return "MSL:" + match.group(1)


def image_mask_pairs(data_directory=DATA_DIRECTORY):
    """List source paths and sequence IDs without changing or splitting data."""
    data_directory = Path(data_directory)
    image_directory = data_directory / "images/edr"
    mask_directory = data_directory / "labels/train"
    if not image_directory.is_dir() or not mask_directory.is_dir():
        raise FileNotFoundError(
            f"Expected images/edr and labels/train inside {data_directory}"
        )

    images = {
        path.stem.upper(): path
        for path in image_directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    }
    masks = sorted(
        path for path in mask_directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )
    if not masks:
        raise FileNotFoundError(f"No training masks found in {mask_directory}")

    pairs = []
    for mask_path in masks:
        image_stem = mask_path.stem.removesuffix("_merged").upper()
        image_path = images.get(image_stem)
        if image_path is None:
            raise FileNotFoundError(f"No corresponding image for {mask_path.name}")
        pairs.append({
            "image": image_path,
            "mask": mask_path,
            "sequence": command_sequence(image_path.name),
        })
    return pairs


def validate_data(data_directory=DATA_DIRECTORY):
    """Check all candidate training pairs without changing any source files.

    Duplicate detection compares decoded grayscale pixels, not JPG filenames
    or compressed file bytes. Near-duplicate frames are not detected here.
    """
    from collections import Counter
    import hashlib

    import numpy as np
    from PIL import Image

    data_directory = Path(data_directory)
    image_directory = data_directory / "images/edr"
    mask_directory = data_directory / "labels/train"
    if not image_directory.is_dir() or not mask_directory.is_dir():
        raise FileNotFoundError("Expected images/edr and labels/train folders")

    images = {
        path.stem.upper(): path
        for path in image_directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    }
    masks = sorted(
        path for path in mask_directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )
    if not masks:
        raise FileNotFoundError(f"No training masks found in {mask_directory}")

    counts = Counter()
    issues = []
    seen_images = {}
    valid_image_hashes = set()
    paired_stems = set()

    for index, mask_path in enumerate(masks, start=1):
        stem = mask_path.stem.removesuffix("_merged").upper()
        image_path = images.get(stem)
        pair_issues = []
        image = None
        mask = None
        image_hash = None
        mask_hash = None
        duplicate_of = None
        duplicate_mask_matches = None

        if image_path is None:
            pair_issues.append("missing_image")
        else:
            paired_stems.add(stem)
            try:
                with Image.open(image_path) as source:
                    image = np.array(source.convert("L"))
                image_hash = hashlib.sha256(
                    str(image.shape).encode() + image.tobytes()
                ).hexdigest()
            except (OSError, ValueError) as error:
                pair_issues.append("unreadable_image")

        try:
            with Image.open(mask_path) as source:
                mask = np.array(source)
            if mask.ndim == 3 and np.all(mask == mask[..., :1]):
                mask = mask[..., 0]
            if mask.ndim != 2:
                pair_issues.append("invalid_mask_channels")
            else:
                mask_hash = hashlib.sha256(
                    str(mask.shape).encode() + mask.tobytes()
                ).hexdigest()
                if not np.isin(mask, [0, 1, 2, 3, IGNORE_LABEL]).all():
                    pair_issues.append("invalid_label_values")
                if np.all(mask == IGNORE_LABEL):
                    pair_issues.append("all_unlabeled")
                if image is not None and image.shape != mask.shape:
                    pair_issues.append("dimension_mismatch")
        except (OSError, ValueError) as error:
            pair_issues.append("unreadable_mask")

        if not pair_issues:
            counts["valid_labeled_pairs"] += 1
            valid_image_hashes.add(image_hash)

        if image_hash is not None:
            if image_hash in seen_images:
                original = seen_images[image_hash]
                duplicate_of = original["image"]
                duplicate_mask_matches = (
                    mask_hash == original["mask_hash"]
                    if mask_hash is not None and original["mask_hash"] is not None
                    else None
                )
                pair_issues.append("duplicate_image")
            else:
                seen_images[image_hash] = {
                    "image": image_path.name,
                    "mask_hash": mask_hash,
                }

        for problem in pair_issues:
            counts[problem] += 1
        if pair_issues:
            issue = {
                "image": image_path.name if image_path is not None else None,
                "mask": mask_path.name,
                "problems": pair_issues,
            }
            if duplicate_of is not None:
                issue["duplicate_of"] = duplicate_of
                issue["duplicate_mask_matches"] = duplicate_mask_matches
            issues.append(issue)

        if index % 1000 == 0 or index == len(masks):
            print(f"Checked {index}/{len(masks)} training pairs", flush=True)

    problem_names = [
        "missing_image", "unreadable_image", "unreadable_mask",
        "dimension_mismatch", "invalid_mask_channels", "invalid_label_values",
        "all_unlabeled", "duplicate_image",
    ]
    return {
        "scope": "Curiosity NavCam training-label candidates only",
        "data_directory": str(data_directory.resolve()),
        "candidate_pairs": len(masks),
        "valid_labeled_pairs_before_deduplication": counts["valid_labeled_pairs"],
        "unique_valid_labeled_images": len(valid_image_hashes),
        "raw_images_without_training_masks": len(set(images) - paired_stems),
        "problem_counts": {name: counts[name] for name in problem_names},
        "issues": issues,
        "source_files_changed": False,
    }


def filter_samples(validation_report, data_directory=DATA_DIRECTORY):
    """Exclude all-unlabeled pairs and flag unresolved duplicate annotations.

    Reuse the completed validation report. This creates a sample list only;
    no source image or mask is deleted, changed, or assigned to a split.
    """
    import json

    data_directory = Path(data_directory).resolve()
    validation_report = Path(validation_report).resolve()
    report = json.loads(validation_report.read_text(encoding="utf-8"))
    if Path(report["data_directory"]).resolve() != data_directory:
        raise ValueError("Validation report describes a different dataset")

    allowed_problems = {"all_unlabeled", "duplicate_image"}
    for problem, count in report["problem_counts"].items():
        if count and problem not in allowed_problems:
            raise ValueError(f"Resolve validation problem before filtering: {problem}")

    excluded_masks = {
        issue["mask"] for issue in report["issues"]
        if "all_unlabeled" in issue["problems"]
    }
    duplicate_reviews = []
    flagged_images = set()
    for issue in report["issues"]:
        if "duplicate_image" not in issue["problems"]:
            continue
        flagged_images.update([issue["image"], issue["duplicate_of"]])
        duplicate_reviews.append({
            "images": [issue["duplicate_of"], issue["image"]],
            "masks_identical": issue["duplicate_mask_matches"],
            "decision": None,
        })

    pairs = image_mask_pairs(data_directory)
    if len(pairs) != report["candidate_pairs"]:
        raise ValueError("Candidate count changed since validation; validate again")
    current_masks = {pair["mask"].name for pair in pairs}
    if not excluded_masks.issubset(current_masks):
        raise ValueError("An excluded mask is missing; validate again")

    samples = []
    for pair in pairs:
        if pair["mask"].name in excluded_masks:
            continue
        samples.append({
            "image": pair["image"].relative_to(data_directory).as_posix(),
            "mask": pair["mask"].relative_to(data_directory).as_posix(),
            "sequence": pair["sequence"],
            "review_required": pair["image"].name in flagged_images,
        })

    return {
        "dataset_root": "raw/msl/ncam",
        "validation_report": validation_report.name,
        "candidate_pairs": len(pairs),
        "excluded_all_unlabeled_pairs": len(excluded_masks),
        "remaining_pairs": len(samples),
        "pairs_requiring_duplicate_review": sum(
            sample["review_required"] for sample in samples
        ),
        "excluded_masks": sorted(excluded_masks),
        "duplicate_reviews": duplicate_reviews,
        "samples": samples,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Validate source pairs without changing raw data")
    parser.add_argument("--data-directory", type=Path, default=DATA_DIRECTORY)
    parser.add_argument(
        "--report", type=Path,
        default=DATA_DIRECTORY.parents[2] / "validation_report.json",
    )
    arguments = parser.parse_args()
    report_path = arguments.report.resolve()
    if report_path.is_relative_to(arguments.data_directory.resolve()):
        parser.error("Save the report outside the raw NavCam directory")
    if report_path.exists():
        parser.error("Report already exists; choose another report path")

    report = validate_data(arguments.data_directory)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "issues"}, indent=2))
