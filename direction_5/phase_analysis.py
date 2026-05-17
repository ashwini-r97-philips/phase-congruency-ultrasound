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

def _lowpass_filter(rows, cols, cutoff=0.45, order=15):
    """Butterworth low-pass to suppress aliasing near Nyquist."""
    u1 = np.fft.ifftshift((np.arange(cols) - cols // 2) / cols)
    u2 = np.fft.ifftshift((np.arange(rows) - rows // 2) / rows)
    u1, u2 = np.meshgrid(u1, u2)
    r = np.sqrt(u1 ** 2 + u2 ** 2)
    return 1.0 / (1.0 + (r / cutoff) ** (2 * order))


def compute_phase_congruency(image_gray, n_scales=4, n_orientations=6,
                              min_wavelength=6, mult=2.1, sigma_on_f=0.55,
                              k=2.0, cutoff=0.45):
    """
    Kovesi 2D phase congruency via numpy FFT.

    Key implementation notes (vs common wrong versions):
      - Frequency grid is normalized to [0, 0.5] (Nyquist = 0.5), not raw pixel radius.
      - Log-Gabor uses log(radius/fo), not log(radius * wavelength).
      - sum_even / sum_odd accumulated PER ORIENTATION across scales.
      - Energy = hypot(sum_even, sum_odd) per orientation.
      - PC_o = max(energy - noise, 0) / (sum_an + eps).
      - Final PC = max across orientations (not sum — sum smears edges).
      - Low-pass filter applied to suppress aliasing.

    Returns float32 map in [0, 1] with thin edge responses.
    """
    rows, cols = image_gray.shape
    img = image_gray.astype(np.float64)

    # Standard FFT layout — NO fftshift here; frequency grid built with ifftshift below.
    IM = np.fft.fft2(img)

    # Normalized frequency coordinates in standard FFT order (0..0.5, wrap to -0.5..0).
    u1 = np.fft.ifftshift((np.arange(cols) - cols // 2) / cols)
    u2 = np.fft.ifftshift((np.arange(rows) - rows // 2) / rows)
    u1, u2 = np.meshgrid(u1, u2)
    radius = np.sqrt(u1 ** 2 + u2 ** 2)   # normalized: 0 at DC, 0.5 at Nyquist
    theta = np.arctan2(-u2, u1)            # -u2 gives standard spatial-domain orientation
    radius[0, 0] = 1.0                     # avoid log(0) at DC

    lp = _lowpass_filter(rows, cols, cutoff=cutoff, order=15)

    # Estimate noise once from the first scale's amplitude distribution.
    # Rayleigh estimator: tau = median(amp) / sqrt(log(4)).
    fo0 = 1.0 / min_wavelength
    lg0 = np.exp(-(np.log(radius / fo0)) ** 2 / (2 * np.log(sigma_on_f) ** 2))
    lg0[0, 0] = 0.0
    lg0 *= lp
    resp0 = np.fft.ifft2(IM * lg0)
    amp0 = np.hypot(resp0.real, resp0.imag)
    tau = np.median(amp0[amp0 > 0]) / np.sqrt(np.log(4))
    noise_threshold = tau * k * np.sqrt(2 * np.log(rows * cols))

    spread_sigma = np.pi / (n_orientations * 1.5)
    pc_map = np.zeros((rows, cols), dtype=np.float64)

    for o in range(n_orientations):
        angle = o * np.pi / n_orientations

        # Angular deviation from target orientation, wrapped to [0, pi/2].
        ds = np.sin(theta) * np.cos(angle) - np.cos(theta) * np.sin(angle)
        dc = np.cos(theta) * np.cos(angle) + np.sin(theta) * np.sin(angle)
        dtheta = np.abs(np.arctan2(ds, dc))
        spread = np.exp(-dtheta ** 2 / (2 * spread_sigma ** 2))

        sum_even = np.zeros((rows, cols), dtype=np.float64)
        sum_odd  = np.zeros((rows, cols), dtype=np.float64)
        sum_an   = np.zeros((rows, cols), dtype=np.float64)

        for s in range(n_scales):
            fo = 1.0 / (min_wavelength * mult ** s)
            # Log-Gabor radial component: centered at fo in normalized frequency.
            lg = np.exp(-(np.log(radius / fo)) ** 2 / (2 * np.log(sigma_on_f) ** 2))
            lg[0, 0] = 0.0
            lg *= lp

            kernel = lg * spread
            response = np.fft.ifft2(IM * kernel)
            even = response.real
            odd  = response.imag
            amplitude = np.hypot(even, odd)

            # Accumulate per orientation across scales.
            sum_even += even
            sum_odd  += odd
            sum_an   += amplitude

        # Per-orientation energy (vector sum of phasors), then noise floor.
        energy = np.hypot(sum_even, sum_odd)
        energy_clamped = np.maximum(energy - noise_threshold, 0.0)
        pc_o = energy_clamped / (sum_an + 1e-8)

        # Max across orientations — preserves thin edge structure.
        pc_map = np.maximum(pc_map, pc_o)

    return pc_map.astype(np.float32)


def compute_phase_congruency_from_cfg(image_gray, pc_cfg):
    return compute_phase_congruency(
        image_gray,
        n_scales=pc_cfg["n_scales"],
        n_orientations=pc_cfg["n_orientations"],
        min_wavelength=pc_cfg["min_wavelength"],
        mult=pc_cfg["mult"],
        sigma_on_f=pc_cfg["sigma_on_f"],
        k=pc_cfg["k"],
        cutoff=pc_cfg.get("cutoff", 0.45),
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
    pacfg = cfg["phase_analysis"]
    primary_r = pacfg["primary_band_r"]

    pc_map = compute_phase_congruency_from_cfg(image_gray, pacfg["pc"])
    sobel_map = compute_sobel_map(image_gray)
    canny_map = compute_canny_soft(image_gray, sigma=pacfg["canny_sigma"])

    np.save(pc_map_dir / f"{image_id}_pc.npy", pc_map)
    np.save(pc_map_dir / f"{image_id}_sobel.npy", sobel_map)
    np.save(pc_map_dir / f"{image_id}_canny.npy", canny_map)

    band = make_boundary_band(gt_bin.astype(bool), primary_r)
    error = (pred_bin.astype(bool) ^ gt_bin.astype(bool))

    auc_pc_boundary    = compute_auc(pc_map, band)
    auc_sobel_boundary = compute_auc(sobel_map, band)
    auc_canny_boundary = compute_auc(canny_map, band)

    error_in_band = error & band
    auc_pc_error_band    = compute_auc(pc_map[band], error_in_band[band].astype(int)) if band.sum() > 0 else float("nan")
    auc_sobel_error_band = compute_auc(sobel_map[band], error_in_band[band].astype(int)) if band.sum() > 0 else float("nan")

    zone = compute_zone_stats(pc_map, gt_bin, band)
    quartile = compute_quartile_error_rates(pc_map, error, band)

    def _fmt(v):
        return round(float(v), 5) if not (isinstance(v, float) and np.isnan(v)) else "nan"

    result = {
        "image_id": image_id,
        "auc_pc_boundary":      _fmt(auc_pc_boundary),
        "auc_sobel_boundary":   _fmt(auc_sobel_boundary),
        "auc_canny_boundary":   _fmt(auc_canny_boundary),
        "auc_pc_error_band5":   _fmt(auc_pc_error_band),
        "auc_sobel_error_band5":_fmt(auc_sobel_error_band),
        **{k: (_fmt(v) if v != "nan" else "nan") for k, v in zone.items()},
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
        gt_mask  = batch["mask"].numpy()[0, 0]   # [H, W]

        pred_path = pred_dir / f"{image_id}_pred.png"
        if not pred_path.exists():
            print(f"[phase] WARNING: prediction not found for {image_id}, skipping")
            continue

        pred_256 = np.array(Image.open(pred_path))
        pred_bin = pred_256 > 127
        gt_bin   = gt_mask > 0.5

        result = analyze_single_phase(image_id, image_np, pred_bin, gt_bin, cfg, pc_map_dir)
        rows.append(result)
        print(f"[phase] {image_id}: auc_pc_bnd={result['auc_pc_boundary']}, auc_sob_bnd={result['auc_sobel_boundary']}")

    if not rows:
        print("[phase] No results — did you run evaluate.py first?")
        return None

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

    q1 = _mean([r["q1_err_rate"] for r in rows])
    q4 = _mean([r["q4_err_rate"] for r in rows])
    print("\n--- Failure pattern interpretation ---")
    if q1 > q4 + 0.05:
        print("Pattern A: Higher error in Q1 (low PC) → AMBIGUITY hypothesis → suggest uncertainty-aware training")
    elif q4 > q1 + 0.05:
        print("Pattern B: Higher error in Q4 (high PC) → IGNORED STRUCTURE hypothesis → suggest phase-guided boundary loss")
    else:
        print("Pattern C: Mixed Q1/Q4 → HYBRID method may be needed")


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    run_phase_analysis(cfg_path)
