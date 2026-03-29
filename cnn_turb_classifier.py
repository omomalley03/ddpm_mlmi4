"""
CNN classifier to quantify turbulence strength from Gaussian OAM beam images.

Architecture mirrors the MATLAB implementation:
    Conv(3x3, 8)  -> BN -> ReLU -> MaxPool(2x2)
    Conv(3x3, 16) -> BN -> ReLU -> MaxPool(2x2)
    Conv(3x3, 32) -> BN -> ReLU -> Flatten -> FC(num_classes)

Inputs are normalised the same way as the DDPM: per-image max-normalised to [-1, 1].

Usage:
    # Train on real data:
    python cnn_turb_classifier.py --mat_path /path/to/data.mat --save_dir checkpoints_cnn

    # Evaluate on DDPM-generated PNGs:
    python cnn_turb_classifier.py --eval_only \\
        --checkpoint checkpoints_cnn/best_cnn.pt \\
        --eval_dir samples/
"""

import argparse
import os
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from PIL import Image
import torchvision.transforms.functional as TF

from dataset_oam import OAMDataset


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TurbCNN(nn.Module):
    """Single-branch CNN that maps a (1, H, W) beam image to a turbulence class."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 8,  3, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # Block 2
            nn.Conv2d(8, 16, 3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            # Block 3
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.classifier = None  # built lazily on first forward pass
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)                    # (B, 32, H/4, W/4)
        flat = feat.flatten(1)                     # (B, 32 * H/4 * W/4)
        if self.classifier is None:
            self.classifier = nn.Linear(flat.shape[1], self.num_classes).to(x.device)
        return self.classifier(flat)               # (B, num_classes)


# ---------------------------------------------------------------------------
# Normalisation helper (must match OAMDataset.__getitem__)
# ---------------------------------------------------------------------------

def normalise_image(img: torch.Tensor) -> torch.Tensor:
    """Per-image normalisation: [0, max] -> [-1, 1].  Matches OAMDataset."""
    vmax = img.max()
    if vmax > 0:
        img = img / vmax
    img = img * 2.0 - 1.0
    return img


# ---------------------------------------------------------------------------
# Label remapping  (raw turb integers -> 0-indexed class indices)
# ---------------------------------------------------------------------------

def build_label_map(turb_categories):
    """Map raw turbulence label values to contiguous 0-indexed class indices."""
    return {v: i for i, v in enumerate(sorted(turb_categories))}


def remap_labels(labels: torch.Tensor, label_map: dict) -> torch.Tensor:
    out = labels.clone()
    for raw, idx in label_map.items():
        out[labels == raw] = idx
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset (gauss mode only, all turbulence levels)
    turb_levels = args.turb_levels if args.turb_levels else None
    dataset = OAMDataset(
        args.mat_path,
        modes=["gauss"],
        image_size=None,   # keep native size (128x128)
        normalize=True,    # [-1, 1] — same as DDPM
        turb_levels=turb_levels,
    )

    label_map = build_label_map(dataset.turb_categories)
    num_classes = len(label_map)
    print(f"Turbulence classes: {dataset.turb_categories}  ->  {num_classes} classes")
    print(f"Label mapping: {label_map}")

    # Train / val split (80 / 20)
    n_val = int(0.2 * len(dataset))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    print(f"Train: {n_train}  Val: {n_val}")

    # Model, loss, optimiser
    model = TurbCNN(num_classes).to(device)
    # Build classifier layer by running a dummy forward pass
    dummy = torch.zeros(1, 1, *dataset.images.shape[2:], device=device)
    _ = model(dummy)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save label map alongside checkpoint for evaluation
    with open(save_dir / "label_map.json", "w") as f:
        json.dump({"label_map": label_map, "turb_categories": dataset.turb_categories}, f)

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        # --- train ---
        model.train()
        train_loss = 0.0
        for imgs, _mode, turb in train_loader:
            imgs  = imgs.to(device)
            turb  = remap_labels(turb, label_map).to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, turb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= n_train

        # --- validate ---
        model.eval()
        val_loss = 0.0
        correct  = 0
        with torch.no_grad():
            for imgs, _mode, turb in val_loader:
                imgs  = imgs.to(device)
                turb  = remap_labels(turb, label_map).to(device)
                logits = model(imgs)
                val_loss += criterion(logits, turb).item() * imgs.size(0)
                correct  += (logits.argmax(1) == turb).sum().item()
        val_loss /= n_val
        val_acc   = correct / n_val

        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_loss": val_loss, "val_acc": val_acc})
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "num_classes": num_classes,
                "label_map": label_map,
                "turb_categories": dataset.turb_categories,
                "image_shape": list(dataset.images.shape[2:]),
            }, save_dir / "best_cnn.pt")
            print(f"  -> Saved best model (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    # Save training history
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training complete. Best val_loss={best_val_loss:.4f}")


# ---------------------------------------------------------------------------
# Evaluation on DDPM-generated PNGs
# ---------------------------------------------------------------------------

def evaluate_ddpm(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    num_classes     = ckpt["num_classes"]
    label_map       = ckpt["label_map"]
    turb_categories = ckpt["turb_categories"]
    idx_to_label    = {v: k for k, v in label_map.items()}

    model = TurbCNN(num_classes).to(device)
    # Build classifier layer with a dummy pass matching the saved image shape
    img_shape = ckpt.get("image_shape", [128, 128])
    dummy = torch.zeros(1, 1, *img_shape, device=device)
    _ = model(dummy)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Classes: {turb_categories}")

    png_files = sorted(p for p in Path(args.eval_dir).glob("*.png")
                       if not p.stem.startswith("grid_"))    
    if not png_files:
        print(f"No PNG files found in {args.eval_dir}")
        return

    print(f"Evaluating {len(png_files)} images from {args.eval_dir} ...")

    # Check for optional ground-truth CSV: eval_dir/labels.csv  (filename,turb_label)
    gt_map = {}
    labels_csv = Path(args.eval_dir) / "labels.csv"
    if labels_csv.exists():
        import csv
        with open(labels_csv) as f:
            for row in csv.DictReader(f):
                gt_map[row["filename"]] = int(row["turb_label"])
        print(f"Found ground-truth labels for {len(gt_map)} images")

    pred_counts = {c: 0 for c in turb_categories}
    correct = 0
    total   = 0

    with torch.no_grad():
        for png in png_files:
            # Load as grayscale float [0, 1]
            img = TF.to_tensor(Image.open(png).convert("L"))  # (1, H, W) in [0,1]
            # Apply same normalisation as OAMDataset
            img = normalise_image(img).unsqueeze(0).to(device)  # (1, 1, H, W)

            logits = model(img)
            pred_idx   = logits.argmax(1).item()
            pred_label = idx_to_label[pred_idx]
            pred_counts[pred_label] = pred_counts.get(pred_label, 0) + 1
            total += 1

            if gt_map:
                gt_raw = gt_map.get(png.name)
                if gt_raw is not None and gt_raw in label_map:
                    gt_idx = label_map[gt_raw]
                    correct += int(pred_idx == gt_idx)

    print("\nPredicted turbulence distribution:")
    for label in sorted(pred_counts):
        pct = 100 * pred_counts[label] / total
        print(f"  turb={label}: {pred_counts[label]:5d} ({pct:.1f}%)")

    if gt_map and total > 0:
        acc = correct / total
        print(f"\nAccuracy vs ground truth: {correct}/{total} = {acc:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="CNN turbulence classifier for OAM Gaussian beams")
    p.add_argument("--mat_path",    type=str, default=None,
                   help="Path to the .mat data file (required for training)")
    p.add_argument("--save_dir",    type=str, default="checkpoints_cnn",
                   help="Directory to save checkpoints and logs")
    p.add_argument("--epochs",      type=int, default=100)
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--patience",    type=int, default=3,
                   help="Early stopping patience (epochs without val_loss improvement)")
    p.add_argument("--turb_levels", type=int, nargs="*", default=None,
                   help="Subset of turbulence levels to train on (default: all)")
    p.add_argument("--num_workers", type=int, default=4)
    # Evaluation flags
    p.add_argument("--eval_only",   action="store_true",
                   help="Skip training; evaluate DDPM images directly")
    p.add_argument("--checkpoint",  type=str, default=None,
                   help="Path to saved best_cnn.pt (required for --eval_only)")
    p.add_argument("--eval_dir",    type=str, default=None,
                   help="Directory of PNG images to classify")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.eval_only:
        if not args.checkpoint or not args.eval_dir:
            raise ValueError("--eval_only requires --checkpoint and --eval_dir")
        evaluate_ddpm(args)
    else:
        if not args.mat_path:
            raise ValueError("--mat_path is required for training")
        train(args)
        # Optionally evaluate on DDPM images after training
        if args.eval_dir and args.eval_dir:
            best_ckpt = str(Path(args.save_dir) / "best_cnn.pt")
            if os.path.exists(best_ckpt):
                args.checkpoint = best_ckpt
                evaluate_ddpm(args)
