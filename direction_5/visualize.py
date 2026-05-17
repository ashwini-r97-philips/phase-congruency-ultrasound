"""Visualization: 2x4 diagnostic panels, aggregate summary plots, failure galleries."""

import sys
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.filters import sobel
from skimage.feature import canny
from skimage.morphology import binary_dilation, disk

from failure_analysis import make_boundary_band, extract_boundary


def _load_cfg(cfg_path):
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _load_image(path):
    return np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def _load_pred(pred_dir, image_id):
    p = pred_dir / f"{image_id}_pred.png"
    return np.array(Image.open(p)) > 127


def _load_error(pred_dir, image_id):
    p = pred_dir / f"{image_id}_error.npy"
    return np.load(p)


def _load_pc(pc_dir, image_id):
    p = pc_dir / f"{image_id}_pc.npy"
    return np.load(p)


# ── 2×4 Panel ─────────────────────────────────────────────────────────────────

def plot_panel(image_id, image, gt_mask, pred_mask, error_map, pc_map,
               boundary_band, out_path, figsize=(20, 10), dpi=150):
    """
    2x4 diagnostic panel:
    Row 1: [US image | GT overlay | Pred overlay | Error map (FP=red, FN=blue)]
    Row 2: [PC map   | PC edges over image | Error over PC | Band+error overlay]
    """
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    fig.suptitle(f"Sample: {image_id}", fontsize=12)

    def _overlay(ax, img, mask, color, title, alpha=0.4):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        overlay = np.zeros((*img.shape, 4))
        overlay[mask, :3] = color
        overlay[mask, 3] = alpha
        ax.imshow(overlay)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    # [0,0] US image
    axes[0, 0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Ultrasound", fontsize=9)
    axes[0, 0].axis("off")

    # [0,1] GT overlay (green)
    _overlay(axes[0, 1], image, gt_mask.astype(bool), [0, 1, 0], "GT Mask")

    # [0,2] Pred overlay (blue)
    _overlay(axes[0, 2], image, pred_mask.astype(bool), [0, 0.5, 1], "Prediction")

    # [0,3] Error map: FP=red, FN=blue
    err_rgb = np.zeros((*image.shape, 3))
    err_rgb[error_map == 1] = [1, 0, 0]  # FP red
    err_rgb[error_map == 2] = [0, 0.3, 1]  # FN blue
    axes[0, 3].imshow(image, cmap="gray", vmin=0, vmax=1, alpha=0.5)
    axes[0, 3].imshow(np.concatenate([err_rgb, (error_map > 0).astype(float)[..., None]], axis=-1))
    red_p = mpatches.Patch(color=[1, 0, 0], label="FP")
    blue_p = mpatches.Patch(color=[0, 0.3, 1], label="FN")
    axes[0, 3].legend(handles=[red_p, blue_p], fontsize=7, loc="lower right")
    axes[0, 3].set_title("Error Map", fontsize=9)
    axes[0, 3].axis("off")

    # [1,0] PC map
    im = axes[1, 0].imshow(pc_map, cmap="viridis", vmin=0, vmax=pc_map.max())
    axes[1, 0].set_title("Phase Congruency", fontsize=9)
    axes[1, 0].axis("off")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # [1,1] PC edges (thresholded) over image
    pc_threshold = np.percentile(pc_map, 90)
    pc_edges = pc_map > pc_threshold
    axes[1, 1].imshow(image, cmap="gray", vmin=0, vmax=1)
    pc_overlay = np.zeros((*image.shape, 4))
    pc_overlay[pc_edges, :3] = [1, 0.8, 0]
    pc_overlay[pc_edges, 3] = 0.7
    axes[1, 1].imshow(pc_overlay)
    axes[1, 1].set_title("PC Edges (top 10%)", fontsize=9)
    axes[1, 1].axis("off")

    # [1,2] Error map over PC map
    axes[1, 2].imshow(pc_map, cmap="viridis", vmin=0, vmax=pc_map.max())
    err_o = np.zeros((*image.shape, 4))
    err_o[error_map == 1] = [1, 0, 0, 0.6]
    err_o[error_map == 2] = [0, 0.3, 1, 0.6]
    axes[1, 2].imshow(err_o)
    axes[1, 2].set_title("Error over PC", fontsize=9)
    axes[1, 2].axis("off")

    # [1,3] Boundary band + error overlay
    axes[1, 3].imshow(image, cmap="gray", vmin=0, vmax=1, alpha=0.7)
    band_o = np.zeros((*image.shape, 4))
    band_o[boundary_band.astype(bool), :3] = [0, 1, 0]
    band_o[boundary_band.astype(bool), 3] = 0.25
    err_o2 = np.zeros((*image.shape, 4))
    err_o2[error_map == 1] = [1, 0, 0, 0.7]
    err_o2[error_map == 2] = [0, 0.3, 1, 0.7]
    axes[1, 3].imshow(band_o)
    axes[1, 3].imshow(err_o2)
    green_p = mpatches.Patch(color=[0, 1, 0], label="Band r=5")
    axes[1, 3].legend(handles=[green_p, red_p, blue_p], fontsize=7, loc="lower right")
    axes[1, 3].set_title("Band + Error", fontsize=9)
    axes[1, 3].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ── Aggregate summary ─────────────────────────────────────────────────────────

def plot_aggregate_summary(failure_df, phase_df, out_dir, dpi=150):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _numeric(series):
        return pd.to_numeric(series, errors="coerce").dropna()

    # Fig A: BER at r=3,5,10 — violin plot
    ber_cols = [c for c in failure_df.columns if c.startswith("ber_r")]
    if ber_cols:
        fig, ax = plt.subplots(figsize=(7, 5))
        data = [_numeric(failure_df[c]).values for c in ber_cols]
        labels = [c.replace("ber_r", "r=") + "px" for c in ber_cols]
        vp = ax.violinplot(data, positions=range(len(data)), showmedians=True)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.axhline(0.55, color="red", linestyle="--", linewidth=1, label="0.55 threshold")
        ax.set_ylabel("Boundary Error Ratio")
        ax.set_title("BER Distribution by Band Radius")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "ber_violin.png", dpi=dpi)
        plt.close(fig)

    # Fig B: AUC comparison bar chart
    auc_keys = ["auc_pc_boundary", "auc_sobel_boundary", "auc_canny_boundary",
                "auc_pc_error_band5", "auc_sobel_error_band5"]
    auc_labels = ["PC\nboundary", "Sobel\nboundary", "Canny\nboundary",
                  "PC\nerror@band5", "Sobel\nerror@band5"]
    auc_cols = [k for k in auc_keys if k in phase_df.columns]
    if auc_cols:
        means = [_numeric(phase_df[k]).mean() for k in auc_cols]
        stds = [_numeric(phase_df[k]).std() for k in auc_cols]
        labels_used = [auc_labels[auc_keys.index(k)] for k in auc_cols]
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#2196F3" if "pc" in k else "#FF9800" if "sobel" in k else "#4CAF50" for k in auc_cols]
        ax.bar(range(len(means)), means, yerr=stds, capsize=4, color=colors)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Random (0.5)")
        ax.set_xticks(range(len(labels_used)))
        ax.set_xticklabels(labels_used, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_ylabel("AUC")
        ax.set_title("AUC Comparison: PC vs Sobel vs Canny")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "auc_comparison.png", dpi=dpi)
        plt.close(fig)

    # Fig C: PC quartile error rate bar chart
    q_keys = ["q1_err_rate", "q2_err_rate", "q3_err_rate", "q4_err_rate"]
    q_keys_present = [k for k in q_keys if k in phase_df.columns]
    if q_keys_present:
        means = [_numeric(phase_df[k]).mean() for k in q_keys_present]
        stds = [_numeric(phase_df[k]).std() for k in q_keys_present]
        labels = ["Q1\n(low PC)", "Q2", "Q3", "Q4\n(high PC)"][:len(q_keys_present)]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.bar(range(len(means)), means, yerr=stds, capsize=4,
               color=["#EF5350", "#FF9800", "#FDD835", "#66BB6A"])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("Error Rate within Boundary Band")
        ax.set_title("Error Rate by PC Quartile (within Band r=5)")
        plt.tight_layout()
        plt.savefig(out_dir / "quartile_error_rates.png", dpi=dpi)
        plt.close(fig)

    print(f"[viz] Aggregate summary saved to {out_dir}")


# ── Failure galleries ─────────────────────────────────────────────────────────

def build_gallery(cfg_path, failure_df=None, phase_df=None, n=20, dpi=150):
    cfg = _load_cfg(cfg_path)
    pred_dir = Path(cfg["training"]["predictions_dir"])
    pc_dir = Path(cfg["phase_analysis"]["output_dir"])
    fig_dir = Path(cfg["visualization"]["output_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    from dataset import make_dataloaders
    loaders = make_dataloaders(cfg_path)

    # Build lookup: image_id → (image array, gt_mask array)
    id_to_data = {}
    for batch in loaders["test"]:
        iid = batch["id"][0]
        img = batch["image"].numpy()[0, 0]
        gt = batch["mask"].numpy()[0, 0]
        id_to_data[iid] = (img, gt)

    if failure_df is None:
        failure_df = pd.read_csv(pred_dir.parent / "metrics" / "failure_metrics.csv")
    if phase_df is None:
        phase_df = pd.read_csv(pc_dir.parent / "metrics" / "phase_metrics.csv")

    df = failure_df.merge(phase_df, on="image_id", how="inner")
    df = df.apply(pd.to_numeric, errors="ignore")

    def _gallery(subset_ids, title, fname):
        subset_ids = subset_ids[:n]
        if not subset_ids:
            return
        cols = min(5, len(subset_ids))
        rows = (len(subset_ids) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.array(axes).flatten()
        for idx, iid in enumerate(subset_ids):
            if iid not in id_to_data:
                continue
            img, gt = id_to_data[iid]
            pred = _load_pred(pred_dir, iid)
            error = _load_error(pred_dir, iid)
            pc = _load_pc(pc_dir, iid)
            band = make_boundary_band(gt.astype(bool), 5)
            ax = axes[idx]
            ax.imshow(img, cmap="gray")
            overlay = np.zeros((*img.shape, 4))
            overlay[error == 1] = [1, 0, 0, 0.5]
            overlay[error == 2] = [0, 0.3, 1, 0.5]
            ax.imshow(overlay)
            ax.set_title(iid, fontsize=7)
            ax.axis("off")
        for ax in axes[len(subset_ids):]:
            ax.axis("off")
        fig.suptitle(title, fontsize=12)
        plt.tight_layout()
        plt.savefig(fig_dir / fname, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[viz] Gallery saved: {fig_dir / fname}")

    # Gallery A: phase helps — high BER + high auc_pc_error
    if "ber_r5" in df.columns and "auc_pc_error_band5" in df.columns:
        a_ids = (
            df.dropna(subset=["ber_r5", "auc_pc_error_band5"])
            .nlargest(n, "ber_r5")["image_id"].tolist()
        )
        _gallery(a_ids, "Gallery A: High BER — Phase Highlights Missed Boundary", "gallery_A.png")

    # Gallery B: ambiguity — high BER + low mean_pc_band
    if "ber_r5" in df.columns and "mean_pc_band" in df.columns:
        b_df = df.dropna(subset=["ber_r5", "mean_pc_band"])
        b_df = b_df[b_df["ber_r5"] > 0.4]
        b_ids = b_df.nsmallest(n, "mean_pc_band")["image_id"].tolist()
        _gallery(b_ids, "Gallery B: High BER, Low Phase — True Ambiguity", "gallery_B.png")

    # Gallery C: misleading phase — high mean_pc_bg
    if "mean_pc_bg" in df.columns:
        c_ids = df.dropna(subset=["mean_pc_bg"]).nlargest(n, "mean_pc_bg")["image_id"].tolist()
        _gallery(c_ids, "Gallery C: High Background PC — Phase Misleads", "gallery_C.png")


# ── Per-sample panels ─────────────────────────────────────────────────────────

def generate_all_panels(cfg_path, max_panels=None):
    cfg = _load_cfg(cfg_path)
    pred_dir = Path(cfg["training"]["predictions_dir"])
    pc_dir = Path(cfg["phase_analysis"]["output_dir"])
    fig_dir = Path(cfg["visualization"]["output_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    figsize = cfg["visualization"]["panel_figsize"]
    dpi = cfg["visualization"]["dpi"]

    from dataset import make_dataloaders
    loaders = make_dataloaders(cfg_path)

    count = 0
    for batch in loaders["test"]:
        iid = batch["id"][0]
        img = batch["image"].numpy()[0, 0]
        gt = batch["mask"].numpy()[0, 0]

        pred_path = pred_dir / f"{iid}_pred.png"
        error_path = pred_dir / f"{iid}_error.npy"
        pc_path = pc_dir / f"{iid}_pc.npy"

        if not (pred_path.exists() and error_path.exists() and pc_path.exists()):
            continue

        pred = _load_pred(pred_dir, iid)
        error = _load_error(pred_dir, iid)
        pc = _load_pc(pc_dir, iid)
        band = make_boundary_band(gt.astype(bool), 5)

        out_path = fig_dir / f"panel_{iid}.png"
        plot_panel(iid, img, gt.astype(bool), pred, error, pc, band, out_path, figsize, dpi)

        count += 1
        if max_panels and count >= max_panels:
            break

    print(f"[viz] Generated {count} diagnostic panels in {fig_dir}")


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    max_p = int(sys.argv[2]) if len(sys.argv) > 2 else None

    cfg = _load_cfg(cfg_path)
    fig_dir = cfg["visualization"]["output_dir"]
    metrics_dir = cfg["evaluation"]["metrics_dir"]

    generate_all_panels(cfg_path, max_panels=max_p)

    try:
        failure_df = pd.read_csv(Path(metrics_dir) / "failure_metrics.csv")
        phase_df = pd.read_csv(Path(metrics_dir) / "phase_metrics.csv")
        plot_aggregate_summary(failure_df, phase_df, fig_dir)
        build_gallery(cfg_path, failure_df=failure_df, phase_df=phase_df)
    except FileNotFoundError as e:
        print(f"[viz] Skipping aggregate/gallery: {e}")
