"""Phase 0: training loop for UNet on TN3K."""

import os
import sys
import csv
import yaml
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import make_dataloaders
from model import build_unet


def _load_cfg(cfg_path):
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ── Losses ────────────────────────────────────────────────────────────────────

def dice_loss(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits)
    probs = probs.view(-1)
    targets = targets.view(-1)
    intersection = (probs * targets).sum()
    return 1.0 - (2.0 * intersection + smooth) / (probs.sum() + targets.sum() + smooth)


def dice_ce_loss(logits, targets):
    bce = nn.BCEWithLogitsLoss()(logits, targets)
    dl = dice_loss(logits, targets)
    return 0.5 * bce + 0.5 * dl


# ── Metric helpers ────────────────────────────────────────────────────────────

def batch_dice(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    smooth = 1e-6
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


# ── Train / validate ──────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = total_dice = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = dice_ce_loss(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_dice += batch_dice(logits, masks)
    n = len(loader)
    return {"loss": total_loss / n, "dice": total_dice / n}


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss = total_dice = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        loss = dice_ce_loss(logits, masks)
        total_loss += loss.item()
        total_dice += batch_dice(logits, masks)
    n = len(loader)
    return {"loss": total_loss / n, "dice": total_dice / n}


# ── Main training loop ────────────────────────────────────────────────────────

def run_training(cfg_path):
    cfg = _load_cfg(cfg_path)
    tcfg = cfg["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device={device}")

    ckpt_dir = Path(tcfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(cfg["evaluation"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    loaders = make_dataloaders(cfg_path)
    model = build_unet(cfg).to(device)

    if tcfg["optimizer"] == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=tcfg["lr"],
            weight_decay=tcfg["weight_decay"],
        )
    else:
        raise ValueError(f"Unknown optimizer: {tcfg['optimizer']}")

    if tcfg["lr_scheduler"] == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=tcfg["epochs"], eta_min=1e-6)
    else:
        scheduler = None

    best_val_dice = -1.0
    patience_counter = 0
    log_rows = []

    for epoch in range(1, tcfg["epochs"] + 1):
        train_metrics = train_one_epoch(model, loaders["train"], optimizer, device)
        val_metrics = validate(model, loaders["val"], device)

        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 5),
            "train_dice": round(train_metrics["dice"], 5),
            "val_loss": round(val_metrics["loss"], 5),
            "val_dice": round(val_metrics["dice"], 5),
        }
        log_rows.append(row)

        print(
            f"Epoch {epoch:03d}/{tcfg['epochs']} | "
            f"train loss={train_metrics['loss']:.4f} dice={train_metrics['dice']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} dice={val_metrics['dice']:.4f}"
        )

        # Save best
        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save(model.state_dict(), ckpt_dir / "best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        # Always save last
        torch.save(model.state_dict(), ckpt_dir / "last_model.pth")

        # Early stopping
        if patience_counter >= tcfg["early_stopping_patience"]:
            print(f"[train] Early stopping at epoch {epoch} (no improvement for {patience_counter} epochs)")
            break

    # Save training log
    log_path = metrics_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"[train] Training log saved to {log_path}")
    print(f"[train] Best val Dice: {best_val_dice:.4f}")


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    run_training(cfg_path)
