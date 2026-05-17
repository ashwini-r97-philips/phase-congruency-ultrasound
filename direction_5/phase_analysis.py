"""Phase 2: phase congruency maps, edge baselines, and AUC/zone/quartile analysis."""

import sys
import csv
import yaml
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.filters import sobel
from skimage.feature import canny
from skimage.morphology import binary_dilation, disk
from sklearn.metrics import roc_auc_score

from dataset import make_dataloaders
from failure_analysis import extract_boundary, make_boundary_band


def _load_cfg(cfg_path):
    with open(cfg_path) as f:
        return yaml.safe_load(f)


# ── Phase congruency (scipy FFT, no external lib) ─────────────────────────────

def _log_gabor(rows, cols, wavelength, sigma_on_f):
    """Log-Gabor radial filter in the frequency domain."""
    cy, cx = rows // 2, cols // 2
    y, x = np.mgrid[-cy:rows - cy, -cx:cols - cx]
    radius = np.hypot(y, x)
    radius[cy, cx] = 1.0  # avoid log(0)
    # Normalised radius so that center frequency = 1/wavelength
    fo = 1.0 / wavelength
    log_gabor = np.exp(-(np.log(radius * wavelength)) ** 2 / (2 * np.log(sigma_on_f) ** 2))
    log_gabor[cy, cx] = 0.0
    return log_gabor


def _orientation_spread(rows, cols, orientation_idx, n_orientations):
    """Orientation-selective Gaussian spread filter in frequency domain."""
    cy, cx = rows // 2, cols // 2
    y, x = np.mgrid[-cy:rows - cy, -cx:cols - cx]
    # Angle of each pixel in frequency domain
    theta = np.arctan2(y, x)
    # Target orientation angle
    angle = orientation_idx * np.pi / n_orientations
    # Difference, wrapped to [-pi/2, pi/2]
    diff = theta - angle
    diff = np.where(np.abs(diff) < np.pi / 2, diff,
                    diff - np.sign(diff) * np.pi)
    # Gaussian spread
    spread_sigma = np.pi / (n_orientations * 1.5)
    return np.exp(-diff ** 2 / (2 * spread_sigma ** 2))


def compute_phase_congruency(image_gray, n_scales=4, n_orientations=6,
                              min_wavelength=6, mult=2.1, sigma_on_f=0.55,
                              k=2.0, cutoff=0.5, g=10, noise_method=-1):
    """
    Kovesi phase congruency via scipy FFT.
    Returns PC map in [0, 1] as float32.
    """
    rows, cols = image_gray.shape

    # Pad to next power-of-2 for FFT efficiency (optional but helps speed)
    img = image_gray.astype(np.float64)
    img_fft = np.fft.fftshift(np.fft.fft2(img))

    # Accumulators
    total_sum_an = np.zeros((rows, cols), dtype=np.float64)
    total_energy = np.zeros((rows, cols), dtype=np.float64)
    # For noise estimation: collect first-scale amplitudes across orientations
    energy_list = []

    for s in range(n_scales):
        wavelength = min_wavelength * (mult ** s)
        log_gabor = _log_gabor(rows, cols, wavelength, sigma_on_f)

        for o in range(n_orientations):
            spread = _orientation_spread(rows, cols, o, n_orientations)
            kernel = log_gabor * spread

            # Filtered response in spatial domain
            filtered = np.fft.ifft2(np.fft.ifftshift(img_fft * kernel))
            even = filtered.real
            odd = filtered.imag

            amplitude = np.sqrt(even ** 2 + odd ** 2)
            total_sum_an += amplitude

            if s == 0:
                energy_list.append(amplitude)

            # Phase deviation: deviation from maximum response angle
            # Maximum response: amplitude itself
            # Phase deviation contribution: amplitude - |even|
            # (classical Kovesi formulation)
            total_energy += amplitude - np.abs(even - amplitude)

    # Noise estimation (median-based)
    if noise_method == -1:
        # Use median of first-scale amplitudes as noise estimate
        first_scale_amp = np.stack(energy_list, axis=0).mean(axis=0)
        tau = np.median(first_scale_amp) / np.sqrt(np.log(4))
        noise_threshold = tau * k * np.sqrt(2 * np.log(rows * cols))
    else:
        noise_threshold = noise_method

    # Phase congruency
    pc = np.maximum(0.0, total_energy - noise_threshold) / (total_sum_an + 1e-8)
    return pc.astype(np.float32)


def compute_phase_congruency_from_cfg(image_gray, pc_cfg):
    return compute_phase_congruency(
        image_gray,
        n_scales=pc_cfg["n_scales"],
        n_orientations=pc_cfg["n_orientations"],
        min_wavelength=pc_cfg["min_wavelength"],
        mult=pc_cfg["mult"],
        sigma_on_f=pc_cfg["sigma_on_f"],
        k=pc_cfg["k"],
        cutoff=pc_cfg["cutoff"],
        g=pc_cfg["g"],
        noise_method=pc_cfg["noise_method"],
    )


# ── Edge baselines ────────────────────────────────────────────────────────────

def compute_sobel_map(image_gray):
    """Normalised Sobel magnitude in [0, 1]."""
    s = sobel(image_gray.astype(np.float64))
    max_val = s.max()
    return (s / max_val).astype(np.float32) if max_val > 0 else s.astype(np.float32)


def compute_canny_soft(image_gray, sigma=2.0):
    """Canny edges dilated by 1px to allow meaningful AUC computation."""
    edges = canny(image_gray.astype(np.float64), sigma=sigma)
    # Dilate binary map by 1px for soft comparison
    dilated = binary_dilation(edges, disk(1))
    return dilated.astype(np.float32)


# ── Analysis helpers ──────────────────────────────────────────────────────────

def compute_zone_stats(pc_map, gt_mask, boundary_band):
    """Mean PC in: boundary band / inside object (not band) / background (not band)."""
    gt_bool = gt_mask.astype(bool)
    band_bool = boundary_band.astype(bool)
    inside = gt_bool & ~band_bool
    background = ~gt_bool & ~band_bool

    def _mean(region):
        pixels = pc_map[region]
        return float(pixels.mean()) if pixels.size > 0 else float("nan")

    return {
        "mean_pc_band": _mean(band_bool),
        "mean_pc_inside": _mean(inside),
        "mean_pc_bg": _mean(background),
    }


def compute_auc(score_map, binary_target):
    """AUC of score_map for predicting binary_target (both flattened)."""
    scores = score_map.ravel().astype(np.float64)
    labels = binary_target.ravel().astype(int)
    pos = labels.sum()
    neg = (1 - labels).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def compute_quartile_error_rates(pc_map, error_map, band):
    """Per-PC-quartile error rate for pixels within `band`."""
    band_bool = band.astype(bool)
    pc_in_band = pc_map[band_bool]
    err_in_band = error_map.astype(bool)[band_bool]

    if pc_in_band.size == 0:
        return {f"q{i}_err_rate": float("nan") for i in range(1, 5)}

    q_edges = np.percentile(pc_in_band, [0, 25, 50, 75, 100])
    result = {}
    for i in range(4):
        lo, hi = q_edges[i], q_edges[i + 1]
        # Include upper edge only for last bin
        if i < 3:
            mask = (pc_in_band >= lo) & (pc_in_band < hi)
        else:
            mask = (pc_in_band >= lo) & (pc_in_band <= hi)
        denom = mask.sum()
        rate = err_in_band[mask].sum() / denom if denom > 0 else float("nan")
        result[f"q{i + 1}_err_rate"] = round(float(rate), 5) if not np.isnan(rate) else "nan"
    return result


# ── Per-image analysis ────────────────────────────────────────────────────────

def analyze_single_phase(image_id, image_gray, pred_bin, gt_bin, cfg, pc_map_dir):
    """Full per-image Phase 2 analysis. Returns dict of metrics."""
    facfg = cfg["failure_analysis"]
    pacfg = cfg["phase_analysis"]
    primary_r = pacfg["primary_band_r"]

    # Compute maps
    pc_map = compute_phase_congruency_from_cfg(image_gray, pacfg["pc"])
    sobel_map = compute_sobel_map(image_gray)
    canny_map = compute_canny_soft(image_gray, sigma=pacfg["canny_sigma"])

    # Save PC map
    np.save(pc_map_dir / f"{image_id}_pc.npy", pc_map)
    np.save(pc_map_dir / f"{image_id}_sobel.npy", sobel_map)
    np.save(pc_map_dir / f"{image_id}_canny.npy", canny_map)

    # Boundary band for primary radius
    band = make_boundary_band(gt_bin.astype(bool), primary_r)
    gt_boundary = band  # used as GT boundary target for AUC

    # Error map
    error = (pred_bin.astype(bool) ^ gt_bin.astype(bool))

    # AUC: does PC/Sobel/Canny predict boundary pixels?
    auc_pc_boundary = compute_auc(pc_map, gt_boundary)
    auc_sobel_boundary = compute_auc(sobel_map, gt_boundary)
    auc_canny_boundary = compute_auc(canny_map, gt_boundary)

    # AUC: does PC/Sobel predict errors within the boundary band?
    error_in_band = error & band
    auc_pc_error_band = compute_auc(pc_map[band], error_in_band[band].astype(int)) if band.sum() > 0 else float("nan")
    auc_sobel_error_band = compute_auc(sobel_map[band], error_in_band[band].astype(int)) if band.sum() > 0 else float("nan")

    # Zone stats
    zone = compute_zone_stats(pc_map, gt_bin, band)

    # Quartile error rates
    quartile = compute_quartile_error_rates(pc_map, error, band)

    result = {
        "image_id": image_id,
        "auc_pc_boundary": round(float(auc_pc_boundary), 5) if not np.isnan(auc_pc_boundary) else "nan",
        "auc_sobel_boundary": round(float(auc_sobel_boundary), 5) if not np.isnan(auc_sobel_boundary) else "nan",
        "auc_canny_boundary": round(float(auc_canny_boundary), 5) if not np.isnan(auc_canny_boundary) else "nan",
        "auc_pc_error_band5": round(float(auc_pc_error_band), 5) if not np.isnan(auc_pc_error_band) else "nan",
        "auc_sobel_error_band5": round(float(auc_sobel_error_band), 5) if not np.isnan(auc_sobel_error_band) else "nan",
        **{k: (round(v, 5) if v != "nan" else "nan") for k, v in zone.items()},
        **quartile,
    }
    return result


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_phase_analysis(cfg_path):
    cfg = _load_cfg(cfg_path)
    pacfg = cfg["phase_analysis"]
    pred_dir = Path(cfg["training"]["predictions_dir"])
    pc_map_dir = Path(pacfg["output_dir"])
    metrics_dir = Path(pacfg["metrics_dir"])
    pc_map_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    loaders = make_dataloaders(cfg_path)
    test_loader = loaders["test"]

    rows = []
    for batch in test_loader:
        image_id = batch["id"][0]
        image_np = batch["image"].numpy()[0, 0]  # [H, W] float32 in [0,1]
        gt_mask = batch["mask"].numpy()[0, 0]    # [H, W]

        pred_path = pred_dir / f"{image_id}_pred.png"
        if not pred_path.exists():
            print(f"[phase] WARNING: prediction not found for {image_id}, skipping")
            continue

        pred_256 = np.array(Image.open(pred_path))
        pred_bin = pred_256 > 127
        gt_bin = gt_mask > 0.5

        result = analyze_single_phase(image_id, image_np, pred_bin, gt_bin, cfg, pc_map_dir)
        rows.append(result)
        print(f"[phase] {image_id}: auc_pc_bnd={result['auc_pc_boundary']}, auc_sob_bnd={result['auc_sobel_boundary']}")

    if not rows:
        print("[phase] No results — did you run evaluate.py first?")
        return None

    # Save CSV
    out_path = metrics_dir / "phase_metrics.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"[phase] Saved phase metrics to {out_path}")

    _print_summary(rows)

    import pandas as pd
    return pd.DataFrame(rows)


def _print_summary(rows):
    def _mean(vals):
        numeric = [v for v in vals if v != "nan"]
        return np.mean(numeric) if numeric else float("nan")

    print("\n--- Phase 2: Phase Relevance Summary ---")
    print(f"Test samples: {len(rows)}")
    print(f"\nBoundary detection AUC (does the map predict GT boundary band?):")
    print(f"  PC:    {_mean([r['auc_pc_boundary'] for r in rows]):.3f}")
    print(f"  Sobel: {_mean([r['auc_sobel_boundary'] for r in rows]):.3f}")
    print(f"  Canny: {_mean([r['auc_canny_boundary'] for r in rows]):.3f}")
    print(f"\nError prediction AUC within boundary band (does PC predict where errors are?):")
    print(f"  PC:    {_mean([r['auc_pc_error_band5'] for r in rows]):.3f}")
    print(f"  Sobel: {_mean([r['auc_sobel_error_band5'] for r in rows]):.3f}")
    print(f"\nMean PC per zone:")
    print(f"  Boundary band: {_mean([r['mean_pc_band'] for r in rows]):.4f}")
    print(f"  Inside object: {_mean([r['mean_pc_inside'] for r in rows]):.4f}")
    print(f"  Background:    {_mean([r['mean_pc_bg'] for r in rows]):.4f}")
    print(f"\nPC quartile error rates within boundary band (Q1=low PC, Q4=high PC):")
    for q in range(1, 5):
        key = f"q{q}_err_rate"
        print(f"  Q{q}: {_mean([r[key] for r in rows]):.4f}")

    # Interpretation
    q1 = _mean([r["q1_err_rate"] for r in rows])
    q4 = _mean([r["q4_err_rate"] for r in rows])
    print("\n--- Failure pattern interpretation ---")
    if q1 > q4 + 0.05:
        print("Pattern A: Higher error in Q1 (low PC) → AMBIGUITY hypothesis supported → suggest uncertainty-aware training")
    elif q4 > q1 + 0.05:
        print("Pattern B: Higher error in Q4 (high PC) → IGNORED STRUCTURE hypothesis supported → suggest phase-guided boundary loss")
    else:
        print("Pattern C: Mixed Q1/Q4 → HYBRID method may be needed")


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    run_phase_analysis(cfg_path)
