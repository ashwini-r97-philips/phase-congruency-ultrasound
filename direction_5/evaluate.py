"""Phase 0 evaluation: Dice, IoU, HD95, BF-score on the test set."""

import os
import sys
import csv
import yaml
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from skimage.morphology import binary_dilation, disk

from dataset import make_dataloaders
from model import build_unet


def _load_cfg(cfg_path):
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_dice(pred, gt):
    smooth = 1e-6
    inter = (pred & gt).sum()
    return (2 * inter + smooth) / (pred.sum() + gt.sum() + smooth)


def compute_iou(pred, gt):
    smooth = 1e-6
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return (inter + smooth) / (union + smooth)


def _extract_boundary(mask):
    """Thin 1-pixel boundary via dilation XOR."""
    return binary_dilation(mask, disk(1)) ^ mask


def compute_bf_score(pred, gt, tolerance=2):
    """Boundary F-score: fraction of predicted boundary pixels within `tolerance` px of GT boundary."""
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0

    pred_b = _extract_boundary(pred)
    gt_b = _extract_boundary(gt)

    gt_b_dilated = binary_dilation(gt_b, disk(tolerance))
    pred_b_dilated = binary_dilation(pred_b, disk(tolerance))

    precision = (pred_b & gt_b_dilated).sum() / (pred_b.sum() + 1e-6)
    recall = (gt_b & pred_b_dilated).sum() / (gt_b.sum() + 1e-6)

    if precision + recall < 1e-6:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_hd95(pred, gt, backend="simpleitk"):
    """95th percentile Hausdorff distance. Returns np.nan if either mask is empty."""
    if pred.sum() == 0 or gt.sum() == 0:
        return np.nan

    if backend == "simpleitk":
        try:
            import SimpleITK as sitk
            pred_img = sitk.GetImageFromArray(pred.astype(np.uint8))
            gt_img = sitk.GetImageFromArray(gt.astype(np.uint8))
            hausdorff = sitk.HausdorffDistanceImageFilter()
            hausdorff.Execute(pred_img, gt_img)
            return hausdorff.GetAverageHausdorffDistance()
        except Exception:
            pass  # Fall through to scipy

    # scipy fallback via distance transforms
    from scipy.ndimage import distance_transform_edt
    pred_dist = distance_transform_edt(~pred)
    gt_dist = distance_transform_edt(~gt)
    # HD95: 95th percentile of symmetric distances
    forward = pred_dist[gt.astype(bool)]
    backward = gt_dist[pred.astype(bool)]
    all_dists = np.concatenate([forward, backward])
    return float(np.percentile(all_dists, 95))


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate_test_set(cfg_path):
    cfg = _load_cfg(cfg_path)
    ecfg = cfg["evaluation"]
    tcfg = cfg["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(tcfg["checkpoint_dir"]) / "best_model.pth"
    pred_dir = Path(tcfg["predictions_dir"])
    metrics_dir = Path(ecfg["metrics_dir"])
    pred_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    model = build_unet(cfg).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print(f"[eval] Loaded checkpoint: {ckpt_path}")

    loaders = make_dataloaders(cfg_path)
    test_loader = loaders["test"]
    threshold = ecfg["threshold"]
    hd95_backend = ecfg.get("hd95_backend", "simpleitk")

    rows = []
    with torch.no_grad():
        for batch in test_loader:
            image = batch["image"].to(device)
            gt_mask = batch["mask"].numpy()[0, 0]  # [H, W]
            image_id = batch["id"][0]
            orig_size = (batch["orig_size"][0].item(), batch["orig_size"][1].item())  # (W, H)

            logits = model(image)
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

            # Resize back to original spatial size for metric computation
            prob_orig = np.array(
                Image.fromarray(prob).resize(orig_size, Image.BILINEAR)
            )
            gt_orig = np.array(
                Image.fromarray(gt_mask).resize(orig_size, Image.NEAREST)
            )

            pred_bin = prob_orig > threshold
            gt_bin = gt_orig > 0.5

            dice = compute_dice(pred_bin, gt_bin)
            iou = compute_iou(pred_bin, gt_bin)
            bf = compute_bf_score(pred_bin, gt_bin)
            hd95 = compute_hd95(pred_bin, gt_bin, backend=hd95_backend)

            # Save binary prediction at training resolution
            pred_256 = (prob > threshold).astype(np.uint8) * 255
            Image.fromarray(pred_256).save(pred_dir / f"{image_id}_pred.png")

            rows.append({
                "image_id": image_id,
                "dice": round(float(dice), 5),
                "iou": round(float(iou), 5),
                "bf_score": round(float(bf), 5),
                "hd95": round(float(hd95), 3) if not np.isnan(hd95) else "nan",
            })

    # Save CSV
    out_path = metrics_dir / "test_metrics.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[eval] Saved test metrics to {out_path}")

    # Aggregate
    dices = [r["dice"] for r in rows]
    ious = [r["iou"] for r in rows]
    bfs = [r["bf_score"] for r in rows]
    hd95s = [r["hd95"] for r in rows if r["hd95"] != "nan"]

    print("\n--- Test Set Results ---")
    print(f"Dice:     {np.mean(dices):.4f} ± {np.std(dices):.4f}")
    print(f"IoU:      {np.mean(ious):.4f} ± {np.std(ious):.4f}")
    print(f"BF-score: {np.mean(bfs):.4f} ± {np.std(bfs):.4f}")
    if hd95s:
        print(f"HD95:     {np.mean(hd95s):.2f} ± {np.std(hd95s):.2f} px  (n={len(hd95s)})")
    print(f"Total samples: {len(rows)}")

    import pandas as pd
    return pd.DataFrame(rows)


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    evaluate_test_set(cfg_path)
