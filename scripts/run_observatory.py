"""
scripts/run_observatory.py

Semantic Evolution Observatory — Post-Hoc Analysis Runner.

Analyzes an existing frame_log.csv from any pipeline run and produces:
    - 7 output CSV/JSON files
    - 5 thesis-quality PNG figures (300 DPI)
    - 1 interactive HTML dashboard

Usage on your existing 56,314-frame run:
    python scripts/run_observatory.py \
        --frame_log experiments/run/frame_log.csv \
        --output_dir results/observatory/run_01/ \
        --experiment_name "proposed_threshold_run"

For a specific baseline:
    python scripts/run_observatory.py \
        --frame_log experiments/baseline_every_frame/frame_log.csv \
        --output_dir results/observatory/every_frame/ \
        --experiment_name "baseline_every_frame"

Compare multiple runs:
    python scripts/run_observatory.py \
        --frame_log experiments/run/frame_log.csv \
        --output_dir results/observatory/run_01/ \
        --downsample 3 \
        --no_dashboard
"""

import argparse
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.observatory.analytics import SemanticAnalyticsEngine
from src.observatory.output_writer import ObservatoryOutputWriter
from src.observatory.visualizer import SemanticVisualizer
from src.observatory.dashboard import ObservatoryDashboard


def parse_args():
    p = argparse.ArgumentParser(
        description="Semantic Evolution Observatory — Post-Hoc Analysis"
    )
    p.add_argument("--frame_log", required=True,
                   help="Path to frame_log.csv from a pipeline run")
    p.add_argument("--output_dir", required=True,
                   help="Directory to write all observatory outputs")
    p.add_argument("--experiment_name", default="observatory_run",
                   help="Label for this analysis run")

    # Analytics parameters
    p.add_argument("--stability_tau", type=float, default=0.10,
                   help="Drift below this → stable region")
    p.add_argument("--transition_tau", type=float, default=0.35,
                   help="Drift above this → transition region")
    p.add_argument("--volatility_window", type=int, default=30,
                   help="Rolling window for volatility computation")
    p.add_argument("--min_region_length", type=int, default=10,
                   help="Minimum frames for a region to be recorded")

    # Visualization
    p.add_argument("--downsample", type=int, default=1,
                   help="Plot every Nth frame (use 3-5 for very large logs)")
    p.add_argument("--no_dashboard", action="store_true",
                   help="Skip interactive HTML dashboard generation")
    p.add_argument("--no_figures", action="store_true",
                   help="Skip matplotlib figure generation")

    return p.parse_args()


def load_frame_log(path: str) -> pd.DataFrame:
    """Load and validate frame_log.csv."""
    print(f"[Observatory] Loading: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"frame_log.csv not found: {path}")

    df = pd.read_csv(path)
    print(f"[Observatory] Loaded {len(df):,} frames")
    print(f"[Observatory] Columns: {list(df.columns)}")

    # Validate required columns
    required = ["frame_id", "drift_total"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Type coercion
    df["frame_id"] = df["frame_id"].astype(int)
    df["drift_total"] = df["drift_total"].astype(float)

    # Coerce boolean columns
    for col in ["vlm_called", "cache_hit"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes"])

    # Fill missing optional columns with safe defaults
    if "timestamp_ms" not in df.columns:
        df["timestamp_ms"] = df["frame_id"] * 33.3   # assume 30fps

    for col in ["drift_embed", "drift_objects", "drift_track"]:
        if col not in df.columns:
            df[col] = df["drift_total"]  # fallback

    return df


def print_analysis_summary(result, experiment_name: str):
    """Print a clean summary table to terminal."""
    print(f"\n{'='*65}")
    print(f"  SEMANTIC EVOLUTION OBSERVATORY — {experiment_name.upper()}")
    print(f"{'='*65}")

    print(f"\n  📊 DATASET")
    print(f"     Total frames:       {result.total_frames:>10,}")
    print(f"     Duration:           {result.total_duration_ms/1000:>10.1f} sec")
    print(f"     Estimated FPS:      {result.fps_estimate:>10.2f}")

    print(f"\n  📈 SEMANTIC DRIFT STATISTICS")
    print(f"     Mean D_t:           {result.mean_drift:>10.4f}")
    print(f"     Median D_t:         {result.median_drift:>10.4f}")
    print(f"     Std D_t:            {result.drift_std:>10.4f}")
    print(f"     Max D_t:            {result.max_drift:>10.4f}")
    print(f"     Mean |Acceleration|:{result.mean_acceleration:>10.4f}")
    print(f"     Mean Volatility:    {result.mean_volatility:>10.4f}")

    print(f"\n  🗺️  SEMANTIC REGIONS")
    stable_frames = sum(r.duration_frames for r in result.stable_regions)
    trans_frames  = sum(r.duration_frames for r in result.transition_regions)
    vol_frames    = sum(r.duration_frames for r in result.volatile_regions)
    total = result.total_frames

    print(f"     Stable regions:     {len(result.stable_regions):>6}  ({stable_frames/total*100:.1f}% of frames)")
    print(f"     Transition regions: {len(result.transition_regions):>6}  ({trans_frames/total*100:.1f}% of frames)")
    print(f"     Volatile regions:   {len(result.volatile_regions):>6}  ({vol_frames/total*100:.1f}% of frames)")

    print(f"\n  ⚡ SEMANTIC EVENTS")
    spikes  = sum(1 for e in result.semantic_events if e.event_type == "spike")
    bursts  = sum(1 for e in result.semantic_events if e.event_type == "burst")
    triggers= sum(1 for e in result.semantic_events if e.event_type == "vlm_trigger")
    print(f"     Drift spikes:       {spikes:>10,}")
    print(f"     Accel bursts:       {bursts:>10,}")
    print(f"     VLM trigger events: {triggers:>10,}")

    print(f"\n  🤖 VLM INVOCATION ANALYSIS")
    print(f"     VLM call rate:      {result.vlm_call_rate:>10.1%}")
    print(f"     Frames saved:       {(1-result.vlm_call_rate)*result.total_frames:>10,.0f}")

    print(f"\n  🔬 CORE HYPOTHESIS VALIDATION")
    if result.vlm_call_rate < 0.5:
        print(f"     ✓ CONFIRMED: Semantic understanding evolves more slowly")
        print(f"       than raw visual appearance. Only {result.vlm_call_rate:.1%} of frames")
        print(f"       required actual VLM reasoning — {(1-result.vlm_call_rate):.1%} served from cache.")
    else:
        print(f"     ⚠ HIGH ACTIVITY: {result.vlm_call_rate:.1%} VLM rate suggests")
        print(f"       a semantically volatile scene. Consider a more stable video.")

    print(f"{'='*65}\n")


def main():
    args = parse_args()
    t_start = time.perf_counter()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    df = load_frame_log(args.frame_log)

    # ── 2. Run analytics ──────────────────────────────────────────────────────
    print("\n[Observatory] Running semantic evolution analytics...")
    engine = SemanticAnalyticsEngine(
        volatility_window=args.volatility_window,
        stability_tau=args.stability_tau,
        transition_tau=args.transition_tau,
        min_region_length=args.min_region_length,
    )
    result = engine.analyze(df)

    # ── 3. Print summary ──────────────────────────────────────────────────────
    print_analysis_summary(result, args.experiment_name)

    # ── 4. Write output files ─────────────────────────────────────────────────
    writer = ObservatoryOutputWriter(args.output_dir)
    output_paths = writer.write_all(result)

    # ── 5. Generate thesis figures ────────────────────────────────────────────
    if not args.no_figures:
        print("\n[Observatory] Generating thesis figures...")
        figures_dir = os.path.join(args.output_dir, "figures")
        visualizer = SemanticVisualizer(
            output_dir=figures_dir,
            downsample=args.downsample,
        )
        fig_paths = visualizer.generate_all(result)
        print(f"\n[Observatory] Figures saved to: {figures_dir}/")
    else:
        fig_paths = {}

    # ── 6. Interactive dashboard ──────────────────────────────────────────────
    if not args.no_dashboard:
        print("\n[Observatory] Building interactive dashboard...")
        dashboard = ObservatoryDashboard(args.output_dir)
        dash_path = dashboard.build(result, filename="semantic_dashboard.html")
    else:
        dash_path = ""

    # ── 7. Save run config ────────────────────────────────────────────────────
    config = {
        "experiment_name": args.experiment_name,
        "frame_log": args.frame_log,
        "output_dir": args.output_dir,
        "analytics_params": {
            "stability_tau": args.stability_tau,
            "transition_tau": args.transition_tau,
            "volatility_window": args.volatility_window,
            "min_region_length": args.min_region_length,
        },
        "runtime_sec": round(time.perf_counter() - t_start, 2),
    }
    config_path = os.path.join(args.output_dir, "observatory_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # ── 8. Final summary ──────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    print(f"\n[Observatory] Analysis complete in {elapsed:.1f}s")
    print(f"[Observatory] All outputs: {args.output_dir}")
    if dash_path:
        print(f"[Observatory] Open dashboard: {dash_path}")
    if fig_paths:
        print(f"[Observatory] Key figure: {fig_paths.get('master_timeline', '')}")


if __name__ == "__main__":
    main()
