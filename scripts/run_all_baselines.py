"""
scripts/run_all_baselines.py

Runs all 5 mandatory baselines on the same video and saves results
to experiments/<baseline_name>/metrics.json.

Baselines:
    1. every_frame         — VLM on every frame
    2. uniform_sampling    — VLM every 30 frames
    3. motion_gating       — VLM on motion events
    4. embedding_threshold — VLM on CLIP drift only
    5. proposed            — Full semantic scheduler (threshold policy)

Usage:
    python scripts/run_all_baselines.py \
        --video path/to/video.mp4 \
        --max_frames 300 \
        --skip_vlm

After this, run:
    python scripts/plot_results.py
"""

import argparse
import subprocess
import sys
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--max_frames", type=int, default=300)
    p.add_argument("--skip_vlm", action="store_true")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


BASELINES = [
    {
        "name": "baseline_every_frame",
        "policy": "every_frame",
        "extra": [],
    },
    {
        "name": "baseline_uniform_30",
        "policy": "fixed_interval",
        "extra": ["--interval", "30"],
    },
    {
        "name": "baseline_motion_gating",
        "policy": "motion_gating",
        "extra": ["--motion_threshold", "10.0"],
    },
    {
        "name": "baseline_embed_threshold",
        "policy": "embedding_threshold",
        "extra": ["--tau", "0.30"],
    },
    {
        "name": "proposed_semantic_scheduler",
        "policy": "threshold",
        "extra": ["--tau_low", "0.15", "--tau_high", "0.40"],
    },
]


def run_baseline(args, baseline):
    out_dir = f"experiments/{baseline['name']}/"
    cmd = [
        sys.executable, "scripts/run_pipeline.py",
        "--video", args.video,
        "--policy", baseline["policy"],
        "--experiment_name", baseline["name"],
        "--output_dir", out_dir,
        "--max_frames", str(args.max_frames),
        "--device", args.device,
        "--verbose",
    ] + baseline["extra"]

    if args.skip_vlm:
        cmd.append("--skip_vlm")

    print(f"\n{'='*60}")
    print(f"  Running: {baseline['name']}")
    print(f"{'='*60}")

    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    args = parse_args()
    successes = []

    for baseline in BASELINES:
        ok = run_baseline(args, baseline)
        successes.append((baseline["name"], ok))

    print(f"\n{'='*60}")
    print("  BASELINE RUN SUMMARY")
    print(f"{'='*60}")
    for name, ok in successes:
        status = "✓ OK" if ok else "✗ FAILED"
        print(f"  {status}  {name}")

    print("\nNext step: python scripts/plot_results.py")


if __name__ == "__main__":
    main()
