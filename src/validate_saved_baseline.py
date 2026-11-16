"""Finish validation of our saved forest without fitting it a second time."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle

from .dataset import load_manifest
from .run_baseline import PROJECT_DIRECTORY, validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=PROJECT_DIRECTORY / "outputs/random_forest_run1")
    args = parser.parse_args()
    manifest_path = PROJECT_DIRECTORY / "data/prepared_splits.json"
    settings = json.loads((args.run / "settings.json").read_text())
    fingerprint = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if fingerprint != settings["manifest_sha256"]:
        raise ValueError("The frozen manifest changed since fitting")
    # Keep the original run's results and examples untouched.
    output = args.run / "resumed_validation"
    if (output / "validation.json").exists():
        raise ValueError("Validation is already complete")
    output.mkdir(exist_ok=True)
    root, manifest = load_manifest(manifest_path)
    records = [record for record in manifest["records"] if record["split"] == "val"]
    # This loads only the model saved by our own training script.
    with (args.run / "model.pkl").open("rb") as stream:
        model = pickle.load(stream)
    results = validate(model, root, records, output)
    (output / "validation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Final validation accuracy: {results['accuracy']:.2%}", flush=True)


if __name__ == "__main__":
    main()
