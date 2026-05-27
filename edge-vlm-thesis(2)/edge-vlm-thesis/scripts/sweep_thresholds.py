"""
scripts/sweep_thresholds.py

Automated Threshold Sweep for Pareto Frontier Analysis.

Sweeps τ_low and τ_high across a grid and records:
    - vlm_call_rate
    - semantic_retention
    - SCE
    - avg_latency_ms

Output: results/tables/pareto_sweep.csv
        results/plots/pareto_frontier.png

This is the core ablation for Section: "Threshold Policy Optimization"

Usage:
    python scripts/sweep_thresholds.py \
        --video path/to/video.mp4 \
        --output_dir results/pareto_sweep/ \
        --skip_vlm  (for fast testing without VLM inference)
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from itertools import product


TAU_LOW_VALUES  = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
TAU_HIGH_VALUES = [0.25, 0.30, 0.35, 0.40, 0.50, 0.60]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output_dir", default="results/pareto_sweep/")
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument("--skip_vlm", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def run_single(args, tau_low, tau_high, exp_name):
    """Run a single pipeline configuration as a subprocess."""
    out_dir = os.path.join(args.output_dir, exp_name)
    cmd = [
        sys.executable, "scripts/run_pipeline.py",
        "--video", args.video,
        "--policy", "threshold",
        "--tau_low", str(tau_low),
        "--tau_high", str(tau_high),
        "--experiment_name", exp_name,
        "--output_dir", out_dir,
        "--max_frames", str(args.max_frames),
        "--device", args.device,
    ]
    if args.skip_vlm:
        cmd.append("--skip_vlm")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [FAILED] {exp_name}: {result.stderr[-200:]}")
        return None

    # Load metrics
    metrics_path = os.path.join(out_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)
    return None


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("results/tables/", exist_ok=True)

    results = []
    total = len(TAU_LOW_VALUES) * len(TAU_HIGH_VALUES)
    run_idx = 0

    for tau_low, tau_high in product(TAU_LOW_VALUES, TAU_HIGH_VALUES):
        if tau_low >= tau_high:
            continue  # invalid configuration

        run_idx += 1
        exp_name = f"sweep_tl{tau_low:.2f}_th{tau_high:.2f}"
        print(f"[Sweep {run_idx}/{total}] tau_low={tau_low:.2f}, tau_high={tau_high:.2f}")

        metrics = run_single(args, tau_low, tau_high, exp_name)
        if metrics is None:
            continue

        results.append({
            "tau_low": tau_low,
            "tau_high": tau_high,
            "vlm_call_rate": metrics.get("vlm_call_rate", 0),
            "semantic_retention": metrics.get("semantic_retention", 0),
            "sce": metrics.get("sce", 0),
            "avg_latency_ms": metrics.get("avg_latency_ms", 0),
            "cache_hit_rate": metrics.get("cache_hit_rate", 0),
        })

        print(f"  → VLM rate: {metrics.get('vlm_call_rate', 0):.2%} | "
              f"Retention: {metrics.get('semantic_retention', 0):.3f} | "
              f"SCE: {metrics.get('sce', 0):.3f}")

    # Save CSV
    if results:
        csv_path = "results/tables/pareto_sweep.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"\n[Sweep] Results saved to {csv_path}")

        # Print top-5 by SCE
        sorted_results = sorted(results, key=lambda x: x["sce"], reverse=True)
        print("\n[Sweep] Top-5 configurations by SCE:")
        for r in sorted_results[:5]:
            print(f"  τ_low={r['tau_low']:.2f}, τ_high={r['tau_high']:.2f} | "
                  f"SCE={r['sce']:.3f} | "
                  f"VLM rate={r['vlm_call_rate']:.1%} | "
                  f"Retention={r['semantic_retention']:.3f}")


if __name__ == "__main__":
    main()
