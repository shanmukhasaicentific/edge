"""
src/config/env_config.py

Environment-aware configuration management.
Handles both local development and Kaggle deployment.

Usage:
    from src.config.env_config import get_test_video_dir, get_output_dir, IS_KAGGLE

    if IS_KAGGLE:
        print("Running on Kaggle")

    video_dir = get_test_video_dir()
    output_dir = get_output_dir()
"""

import os
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────
# Environment Detection
# ──────────────────────────────────────────────────────────────────────────

IS_KAGGLE = os.path.exists("/kaggle/input")
IS_COLAB = os.path.exists("/content")
IS_LOCAL = not (IS_KAGGLE or IS_COLAB)

PYTHON_VERSION = f"{os.sys.version_info.major}.{os.sys.version_info.minor}"
ENVIRONMENT = "kaggle" if IS_KAGGLE else "colab" if IS_COLAB else "local"


# ──────────────────────────────────────────────────────────────────────────
# Root Directories
# ──────────────────────────────────────────────────────────────────────────

if IS_KAGGLE:
    PROJECT_ROOT = Path("/kaggle/working/edge-vlm-thesis")
    KAGGLE_INPUT = Path("/kaggle/input")
elif IS_COLAB:
    PROJECT_ROOT = Path("/content/edge-vlm-thesis")
    KAGGLE_INPUT = None
else:
    # Local: assume script is in src/config/env_config.py
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    KAGGLE_INPUT = None


# ──────────────────────────────────────────────────────────────────────────
# Configurable Paths (via Environment Variables)
# ──────────────────────────────────────────────────────────────────────────

def get_test_video_dir() -> Path:
    """
    Get test video directory.

    Priority:
    1. TEST_VIDEO_DIR environment variable
    2. Kaggle input dataset (if available)
    3. Local experiments/test directory

    Example (Kaggle):
        os.environ["TEST_VIDEO_DIR"] = "/kaggle/input/videos"
        python train.py
    """
    # Check explicit environment variable
    env_path = os.environ.get("TEST_VIDEO_DIR")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
        print(f"WARNING: TEST_VIDEO_DIR='{env_path}' does not exist")

    # Try Kaggle datasets (user can upload with custom name)
    if IS_KAGGLE:
        kaggle_datasets = list(KAGGLE_INPUT.glob("*/"))
        for dataset in kaggle_datasets:
            # Look for videos in any dataset
            videos = list(dataset.glob("*.mp4")) + list(dataset.glob("*.avi")) + list(dataset.glob("*.mov"))
            if videos:
                return dataset

    # Fallback: local experiments directory
    default_path = PROJECT_ROOT / "experiments" / "test"
    if default_path.exists():
        return default_path

    # Last resort: create it
    default_path.mkdir(parents=True, exist_ok=True)
    return default_path


def get_output_dir() -> Path:
    """
    Get output directory for results.

    Priority:
    1. OUTPUT_DIR environment variable
    2. Kaggle working directory
    3. Local outputs directory
    """
    env_path = os.environ.get("OUTPUT_DIR")
    if env_path:
        path = Path(env_path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    if IS_KAGGLE:
        path = Path("/kaggle/working/results")
    else:
        path = PROJECT_ROOT / "outputs"

    path.mkdir(parents=True, exist_ok=True)
    return path


def get_checkpoint_dir() -> Path:
    """Get checkpoint directory for resuming experiments."""
    env_path = os.environ.get("CHECKPOINT_DIR")
    if env_path:
        path = Path(env_path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    if IS_KAGGLE:
        path = Path("/kaggle/working/checkpoints")
    else:
        path = PROJECT_ROOT / "autoresearch-master" / "checkpoints"

    path.mkdir(parents=True, exist_ok=True)
    return path


def get_config_dir() -> Path:
    """Get configuration directory."""
    return PROJECT_ROOT / "experiments"


# ──────────────────────────────────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────────────────────────────────

def print_environment_info():
    """Print environment and path information."""
    print("\n" + "="*70)
    print("ENVIRONMENT CONFIGURATION")
    print("="*70)
    print(f"\nEnvironment:     {ENVIRONMENT.upper()}")
    print(f"Python:          {PYTHON_VERSION}")
    print(f"Project Root:    {PROJECT_ROOT}")
    print(f"Test Videos:     {get_test_video_dir()}")
    print(f"Output Dir:      {get_output_dir()}")
    print(f"Checkpoints:     {get_checkpoint_dir()}")
    print(f"Configs:         {get_config_dir()}")

    if IS_KAGGLE:
        print(f"Kaggle Input:    {KAGGLE_INPUT}")

    print("\n" + "="*70 + "\n")


# ──────────────────────────────────────────────────────────────────────────
# Usage in Application Code
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print_environment_info()
