"""One initial CNN experiment, using training and validation data only.

Run from the repository folder: python -m src.train_cnn
Starting settings are not tuned. Internal-test files are never read.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import IGNORE_LABEL
from .dataset import TerrainDataset
from .model import TerrainCNN


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]


def run_epoch(model, loader, device, optimizer=None):
    """Fit when an optimizer is supplied; otherwise only measure validation."""
    training = optimizer is not None
    model.train(training)
    # Sum losses so epoch averages are weighted by labeled-pixel counts.
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL, reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total_labeled = 0
    skipped_batches = 0

    for batch_number, batch in enumerate(loader, 1):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        valid = masks != IGNORE_LABEL
        labeled_count = int(valid.sum().item())
        # Some validation tiles can contain no annotations at all.
        if labeled_count == 0:
            skipped_batches += 1
            continue

        if training:
            optimizer.zero_grad()
        # Validation should not build the gradient information used for fitting.
        with torch.set_grad_enabled(training):
            scores = model(images)
            if not torch.isfinite(scores).all():
                raise ValueError("The CNN produced nonfinite scores")
            loss_sum = criterion(scores, masks)
            loss = loss_sum / labeled_count
            if not torch.isfinite(loss):
                raise ValueError("The CNN loss is nonfinite")
            if training:
                loss.backward()
                optimizer.step()

        # Dimension 1 contains the four class scores.
        predictions = scores.argmax(dim=1)
        total_correct += int(((predictions == masks) & valid).sum().item())
        total_labeled += labeled_count
        total_loss += float(loss_sum.detach().item())
        if batch_number % 50 == 0:
            print(f"Batch {batch_number}/{len(loader)}; "
                  f"accuracy so far: {total_correct / total_labeled:.2%}", flush=True)

    if total_labeled == 0:
        raise ValueError("No labeled pixels were processed")
    return {"loss": total_loss / total_labeled,
            "accuracy": total_correct / total_labeled,
            "correct_pixels": total_correct, "labeled_pixels": total_labeled,
            "all_ignore_batches_skipped": skipped_batches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--output", type=Path,
                        default=PROJECT_DIRECTORY / "outputs/cnn_run1")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("Epochs must be positive")
    if args.output.exists() or args.output.resolve().is_relative_to(PROJECT_DIRECTORY / "data"):
        parser.error("Choose a new output folder outside data/")

    seed = 360
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = PROJECT_DIRECTORY / "data/prepared_splits.json"
    train_data = TerrainDataset(manifest, "train", augment=True, seed=seed)
    val_data = TerrainDataset(manifest, "val")
    # Use ordinary single-process loading for a straightforward first version.
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False, num_workers=0)
    model = TerrainCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    args.output.mkdir(parents=True)
    settings = {"seed": seed, "epochs": args.epochs, "batch_size": 8,
                "optimizer": "Adam", "learning_rate": 0.001,
                "loss": "Unweighted cross entropy; ignore 255",
                "metric": "Labeled-pixel accuracy", "device": str(device),
                "torch_version": str(torch.__version__),
                "parameters": sum(p.numel() for p in model.parameters()),
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2))
    history = []
    best_accuracy = -1.0
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}, device: {device}", flush=True)
        train_data.set_epoch(epoch)
        training = run_epoch(model, train_loader, device, optimizer)
        # Save before validation so completed training is preserved if interrupted.
        checkpoint = {"epoch": epoch + 1, "model": model.state_dict(),
                      "optimizer": optimizer.state_dict(), "settings": settings}
        torch.save(checkpoint, args.output / "last.pt")
        validation = run_epoch(model, val_loader, device)
        history.append({"epoch": epoch + 1, "train": training, "val": validation})
        (args.output / "history.json").write_text(json.dumps(history, indent=2))
        if validation["accuracy"] > best_accuracy:
            best_accuracy = validation["accuracy"]
            torch.save(checkpoint, args.output / "best.pt")
        print(f"Training accuracy: {training['accuracy']:.2%}; "
              f"validation accuracy: {validation['accuracy']:.2%}", flush=True)


if __name__ == "__main__":
    main()
