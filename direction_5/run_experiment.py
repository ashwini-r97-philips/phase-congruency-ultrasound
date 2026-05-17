"""Orchestrator: run Phase 0, 1, 2 pipeline in sequence or selectively."""

import argparse
import os
import sys
from pathlib import Path

VALID_PHASES = ["0_train", "0_eval", "1_failure", "2_phase", "viz"]


def main(cfg_path, phases):
    cfg_path = str(Path(cfg_path).resolve())
    os.chdir(Path(__file__).parent)

    if "0_train" in phases:
        print("\n" + "=" * 60)
        print("PHASE 0 — Training UNet")
        print("=" * 60)
        from train import run_training
        run_training(cfg_path)

    if "0_eval" in phases:
        print("\n" + "=" * 60)
        print("PHASE 0 — Evaluating on test set")
        print("=" * 60)
        from evaluate import evaluate_test_set
        evaluate_test_set(cfg_path)

    if "1_failure" in phases:
        print("\n" + "=" * 60)
        print("PHASE 1 — Failure Analysis")
        print("=" * 60)
        from failure_analysis import run_failure_analysis
        run_failure_analysis(cfg_path)

    if "2_phase" in phases:
        print("\n" + "=" * 60)
        print("PHASE 2 — Phase Relevance Analysis")
        print("=" * 60)
        from phase_analysis import run_phase_analysis
        run_phase_analysis(cfg_path)

    if "viz" in phases:
        print("\n" + "=" * 60)
        print("VISUALIZATION")
        print("=" * 60)
        import pandas as pd
        import yaml
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        metrics_dir = Path(cfg["evaluation"]["metrics_dir"])
        failure_path = metrics_dir / "failure_metrics.csv"
        phase_path = metrics_dir / "phase_metrics.csv"

        from visualize import generate_all_panels, plot_aggregate_summary, build_gallery, plot_edge_comparison
        generate_all_panels(cfg_path)
        plot_edge_comparison(cfg_path)

        if failure_path.exists() and phase_path.exists():
            failure_df = pd.read_csv(failure_path)
            phase_df = pd.read_csv(phase_path)
            plot_aggregate_summary(failure_df, phase_df, cfg["visualization"]["output_dir"])
            build_gallery(cfg_path, failure_df=failure_df, phase_df=phase_df)
        else:
            missing = []
            if not failure_path.exists():
                missing.append("failure_metrics.csv (run 1_failure first)")
            if not phase_path.exists():
                missing.append("phase_metrics.csv (run 2_phase first)")
            print(f"[viz] Skipping aggregate/gallery — missing: {', '.join(missing)}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Direction 5: Phase-congruency failure analysis pipeline"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--phases",
        nargs="+",
        default=VALID_PHASES,
        choices=VALID_PHASES,
        help=(
            "Phases to run. Options: "
            "0_train (train UNet), "
            "0_eval (evaluate + save predictions), "
            "1_failure (failure analysis), "
            "2_phase (phase relevance analysis), "
            "viz (visualizations). "
            "Default: all."
        ),
    )
    args = parser.parse_args()
    main(args.config, args.phases)
