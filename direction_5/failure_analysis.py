"""Phase 1: error maps, boundary bands, and Boundary Error Ratio metrics."""

import sys
import csv
import yaml
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.morphology import binary_dilation, disk

from dataset import make_dataloaders


def _load_cfg(cfg_path):
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ── Core geometry helpers ─────────────────────────────────────────────────────

def extract_boundary(mask_bin):
    """1-px boundary via dilation XOR."""
    return binary_dilation(mask_bin, disk(1)) ^ mask_bin


def make_boundary_band(mask_bin, radius):
    """Dilate the GT boundary by `radius` pixels."""
    boundary = extract_boundary(mask_bin)
    return binary_dilation(boundary, disk(radius))


# ── Metric ────────────────────────────────────────────────────────────────────

def compute_ber(error_map, band):
    """Boundary Error Ratio: fraction of errors that lie within the band."""
    total_error = error_map.sum()
    if total_error == 0:
        return np.nan
    return (error_map & band).sum() / total_error


# ── Per-image analysis ────────────────────────────────────────────────────────

def analyze_single(image_id, pred_bin, gt_bin, radii, pred_dir):
    """
    Compute error map and BER metrics for one image.

    pred_bin, gt_bin: bool/uint8 arrays of same shape
    radii: list of ints, e.g. [3, 5, 10]
    pred_dir: Path — where to save the error .npy file

    Returns dict with per-image metrics.
    """
    pred_bin = pred_bin.astype(bool)
    gt_bin = gt_bin.astype(bool)

    fp = pred_bin & ~gt_bin
    fn = ~pred_bin & gt_bin
    error = fp | fn

    # Error map: 0=correct, 1=FP, 2=FN
    error_map = np.zeros(pred_bin.shape, dtype=np.uint8)
    error_map[fp] = 1
    error_map[fn] = 2
    np.save(pred_dir / f"{image_id}_error.npy", error_map)

    result = {
        "image_id": image_id,
        "error_total": int(error.sum()),
        "fp_total": int(fp.sum()),
        "fn_total": int(fn.sum()),
        "gt_pixels": int(gt_bin.sum()),
    }

    for r in radii:
        band = make_boundary_band(gt_bin, r)
        ber = compute_ber(error, band)
        result[f"ber_r{r}"] = round(float(ber), 5) if not np.isnan(ber) else "nan"

    # FP and FN BER at primary radius (r=5)
    r_primary = 5 if 5 in radii else radii[0]
    band5 = make_boundary_band(gt_bin, r_primary)
    fp_ber = compute_ber(fp, band5)
    fn_ber = compute_ber(fn, band5)
    result[f"fp_ber_r{r_primary}"] = round(float(fp_ber), 5) if not np.isnan(fp_ber) else "nan"
    result[f"fn_ber_r{r_primary}"] = round(float(fn_ber), 5) if not np.isnan(fn_ber) else "nan"

    return result


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_failure_analysis(cfg_path):
    cfg = _load_cfg(cfg_path)
    facfg = cfg["failure_analysis"]
    radii = facfg["boundary_radii"]
    pred_dir = Path(cfg["training"]["predictions_dir"])
    metrics_dir = Path(facfg["output_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)

    loaders = make_dataloaders(cfg_path)
    test_loader = loaders["test"]

    rows = []
    for batch in test_loader:
        image_id = batch["id"][0]
        gt_mask = batch["mask"].numpy()[0, 0]  # [H, W]
        orig_size = (batch["orig_size"][0].item(), batch["orig_size"][1].item())  # (W, H)

        pred_path = pred_dir / f"{image_id}_pred.png"
        if not pred_path.exists():
            print(f"[failure] WARNING: prediction not found for {image_id}, skipping")
            continue

        pred_256 = np.array(Image.open(pred_path))  # [256, 256] uint8
        # Resize GT to match prediction size (256x256)
        gt_256 = np.array(
            Image.fromarray((gt_mask * 255).astype(np.uint8)).resize(
                (pred_256.shape[1], pred_256.shape[0]), Image.NEAREST
            )
        )

        pred_bin = pred_256 > 127
        gt_bin = gt_256 > 127

        result = analyze_single(image_id, pred_bin, gt_bin, radii, pred_dir)
        rows.append(result)

    if not rows:
        print("[failure] No results — did you run evaluate.py first?")
        return None

    # Save CSV
    out_path = metrics_dir / "failure_metrics.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[failure] Saved failure metrics to {out_path}")

    # Aggregate
    _print_summary(rows, radii)

    import pandas as pd
    return pd.DataFrame(rows)


def _print_summary(rows, radii):
    print("\n--- Phase 1: Failure Analysis Summary ---")
    n = len(rows)
    print(f"Test samples: {n}")

    for r in radii:
        key = f"ber_r{r}"
        vals = [r_[key] for r_ in rows if r_[key] != "nan"]
        if vals:
            mean_ber = np.mean(vals)
            frac_above_50 = np.mean([v > 0.50 for v in vals])
            frac_above_60 = np.mean([v > 0.60 for v in vals])
            print(
                f"BER r={r}px: mean={mean_ber:.3f}  "
                f">50%={frac_above_50:.1%}  >60%={frac_above_60:.1%}"
            )

    r_primary = 5 if 5 in radii else radii[0]
    ber5_vals = [r_[f"ber_r{r_primary}"] for r_ in rows if r_[f"ber_r{r_primary}"] != "nan"]
    mean_ber5 = np.mean(ber5_vals) if ber5_vals else float("nan")

    print("\n--- Direction 5 validity check (BER r=5) ---")
    if mean_ber5 > 0.55:
        print(f"RESULT: mean BER_r5 = {mean_ber5:.3f} > 0.55  → Boundary failure is DOMINANT. Direction 5 is supported.")
    elif mean_ber5 > 0.40:
        print(f"RESULT: mean BER_r5 = {mean_ber5:.3f} (0.40–0.55) → Boundary failure is MODERATE. Direction 5 is plausible.")
    else:
        print(f"RESULT: mean BER_r5 = {mean_ber5:.3f} < 0.40  → Errors are diffuse. Direction 5 hypothesis is WEAK.")


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    run_failure_analysis(cfg_path)
