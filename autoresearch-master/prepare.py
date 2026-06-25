#!/usr/bin/env python3
"""
Setup and verification script for thesis AutoResearch experiments.

This file is READ-ONLY and defines:
- Fixed constants (time budget, video path, metrics)
- Data loading (test videos)
- Evaluation harness (metric calculation)

DO NOT MODIFY THIS FILE.

Usage:
    python prepare.py  # Check that environment is set up
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Fixed Constants (DO NOT MODIFY)
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).parent.parent
EXPERIMENTS_DIR = PROJECT_DIR / "experiments" / "test"
SCRIPTS_DIR = PROJECT_DIR / "scripts"
SRC_DIR = PROJECT_DIR / "src"

# Fixed time budget for each experiment (in seconds)
TIME_BUDGET = 600  # 10 minutes max per experiment

# Evaluation constants
MIN_RETENTION_FOR_VALID = 0.80  # Frames must be >= 80% semantically valid
METRIC_PRECISION = 4  # Decimal places for metrics

# ---------------------------------------------------------------------------
# Ground Truth Metric
# ---------------------------------------------------------------------------

def evaluate_sce(semantic_retention: float, vlm_call_rate: float) -> float:
    """
    Calculate Semantic Compute Efficiency (SCE).

    This is the PRIMARY METRIC for the thesis.
    You CANNOT override this calculation.

    SCE = semantic_retention / vlm_call_rate

    Higher is better:
    - semantic_retention: fraction of frames with valid semantic state (0.0-1.0)
    - vlm_call_rate: VLM invocations per second (lower is better for efficiency)

    Args:
        semantic_retention: float in [0.0, 1.0]
        vlm_call_rate: float >= 0.0

    Returns:
        float: SCE (higher is better)
    """
    if vlm_call_rate <= 0.0:
        return 0.0
    return semantic_retention / vlm_call_rate

# ---------------------------------------------------------------------------
# Experiment Validation
# ---------------------------------------------------------------------------

def validate_experiment_config(config: dict) -> bool:
    """
    Validate that experiment configuration is valid.

    Returns True if config is valid, False otherwise.
    """
    required_keys = ['tau_low', 'tau_high', 'alpha', 'beta', 'gamma']

    # Check all keys present
    if not all(k in config for k in required_keys):
        print(f"ERROR: Config missing required keys: {required_keys}")
        return False

    # Validate threshold constraints
    tau_low = config['tau_low']
    tau_high = config['tau_high']

    if not (0.0 <= tau_low <= 1.0):
        print(f"ERROR: tau_low={tau_low} must be in [0.0, 1.0]")
        return False

    if not (0.0 <= tau_high <= 1.0):
        print(f"ERROR: tau_high={tau_high} must be in [0.0, 1.0]")
        return False

    if tau_low >= tau_high:
        print(f"ERROR: tau_low={tau_low} must be < tau_high={tau_high}")
        return False

    # Validate weights
    alpha = config['alpha']
    beta = config['beta']
    gamma = config['gamma']

    if not all(0.0 <= w <= 1.0 for w in [alpha, beta, gamma]):
        print(f"ERROR: Weights must be in [0.0, 1.0]")
        return False

    # Weights should sum to ~1.0 (allow some tolerance)
    weight_sum = alpha + beta + gamma
    if not (0.5 <= weight_sum <= 1.5):
        print(f"WARNING: Weights sum to {weight_sum:.2f}, expected ~1.0")

    return True

# ---------------------------------------------------------------------------
# Environment Check
# ---------------------------------------------------------------------------

def check_environment():
    """Verify that all required files and dependencies exist."""
    print("Checking AutoResearch environment...")

    # Check project structure
    checks = [
        (PROJECT_DIR / "README.md", "Project README"),
        (PROJECT_DIR / "src", "Source code directory"),
        (PROJECT_DIR / "scripts", "Scripts directory"),
        (EXPERIMENTS_DIR, "Test video directory"),
        (SRC_DIR / "semantic_memory", "Semantic memory module"),
        (SRC_DIR / "scheduler", "Scheduler module"),
    ]

    all_ok = True
    for path, name in checks:
        if path.exists():
            print(f"  ✓ {name}: {path}")
        else:
            print(f"  ✗ {name}: NOT FOUND ({path})")
            all_ok = False

    # Check for test video
    video_found = False
    for ext in ['*.mp4', '*.avi', '*.mov']:
        if list(EXPERIMENTS_DIR.glob(ext)):
            video_found = True
            break

    if video_found:
        print(f"  ✓ Test video(s) found in {EXPERIMENTS_DIR}")
    else:
        print(f"  ✗ No test video found in {EXPERIMENTS_DIR}")
        print(f"    Please provide test_video.mp4 in experiments/test/")
        all_ok = False

    # Check Python dependencies
    try:
        import torch
        import yaml
        import cv2
        import numpy as np
        from pathlib import Path
        print(f"  ✓ Python dependencies installed")
    except ImportError as e:
        print(f"  ✗ Missing Python dependency: {e}")
        all_ok = False

    return all_ok

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    success = check_environment()

    if success:
        print("\n✓ Environment is ready for AutoResearch experiments")
        print("\nNext: Run `python train.py --phase threshold_sweep`")
        sys.exit(0)
    else:
        print("\n✗ Environment check failed. Please fix the issues above.")
        print("\nYou need:")
        print("  1. Test video in experiments/test/")
        print("  2. Python packages: torch, pyyaml, opencv-python, numpy")
        print("  3. Project code in src/")
        sys.exit(1)
