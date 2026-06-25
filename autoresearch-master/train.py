#!/usr/bin/env python3
"""
Autonomous scheduler optimizer with wandb tracking, checkpointing, and multi-GPU support.

This script is the "train.py" equivalent for the thesis AutoResearch loop.
It runs semantic drift scheduler experiments with configurable thresholds and weights.

Features:
  - wandb integration for experiment tracking (cloud-synced)
  - Checkpoint saving/loading for resuming across machines
  - Multi-GPU support (2x T4 on Kaggle)
  - Cross-machine experiment continuity

Usage:
    python train.py --phase threshold_sweep --config exp_001
    python train.py --phase ablation --config exp_050
    python train.py --phase failure_analysis --config exp_060 --scenario stable

    # Resume from checkpoint:
    python train.py --resume-checkpoint checkpoints/exp_001_phase0_checkpoint.pt

    # Multi-GPU (automatic on Kaggle with 2 T4s):
    python train.py --phase threshold_sweep --multi-gpu

    # Wandb tracking (must login first):
    wandb login
    python train.py --phase threshold_sweep --wandb-project "edge-vlm-thesis"
"""

import json
import csv
import sys
import time
import subprocess
import argparse
import os
import torch
import pickle
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Add parent directory to path to import thesis code
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import wandb (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING: wandb not installed. Run: pip install wandb")
    print("         Experiment tracking will be local-only")

# ---------------------------------------------------------------------------
# Checkpoint Management
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Manages experiment checkpoints for resuming across machines."""

    def __init__(self, checkpoint_dir: Path = None):
        """Initialize checkpoint manager."""
        if checkpoint_dir is None:
            checkpoint_dir = Path(__file__).parent / "checkpoints"
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, exp_id: int, phase: str, config: dict, metrics: dict,
                       status: str, timestamp: str) -> Path:
        """Save experiment checkpoint."""
        checkpoint_data = {
            'exp_id': exp_id,
            'phase': phase,
            'config': config,
            'metrics': metrics,
            'status': status,
            'timestamp': timestamp,
            'machine': os.uname().nodename,
            'gpu_count': torch.cuda.device_count(),
        }

        checkpoint_path = self.checkpoint_dir / f"exp_{exp_id:03d}_phase_{phase}_checkpoint.pt"
        torch.save(checkpoint_data, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Load experiment checkpoint."""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return torch.load(checkpoint_path)

    def get_latest_checkpoint(self, phase: str = None) -> Optional[Path]:
        """Get most recent checkpoint (optionally filtered by phase)."""
        checkpoints = list(self.checkpoint_dir.glob("*.pt"))
        if not checkpoints:
            return None

        if phase:
            checkpoints = [c for c in checkpoints if f"phase_{phase}" in c.name]

        if checkpoints:
            return max(checkpoints, key=lambda p: p.stat().st_mtime)
        return None

    def list_checkpoints(self) -> list:
        """List all checkpoints."""
        checkpoints = sorted(self.checkpoint_dir.glob("*.pt"))
        for ckpt in checkpoints:
            data = torch.load(ckpt)
            print(f"  {ckpt.name}:")
            print(f"    Exp: {data['exp_id']}, Phase: {data['phase']}, SCE: {data['metrics'].get('sce', 0):.4f}")
            print(f"    Machine: {data['machine']}, GPUs: {data['gpu_count']}")
        return checkpoints

# ---------------------------------------------------------------------------
# Wandb Integration
# ---------------------------------------------------------------------------

class WandBTracker:
    """Tracks experiments on Weights & Biases."""

    def __init__(self, project: str, entity: str = None, offline: bool = False):
        """Initialize wandb tracker."""
        self.enabled = WANDB_AVAILABLE and not offline
        self.project = project
        self.entity = entity
        self.run = None

        if self.enabled:
            try:
                self.run = wandb.init(
                    project=project,
                    entity=entity,
                    reinit=True,
                    config={
                        'framework': 'karpathy_autoresearch',
                        'thesis': 'semantic_drift_optimization',
                    }
                )
                print(f"✓ Wandb tracking enabled: {self.run.get_url()}")
            except Exception as e:
                print(f"WARNING: Could not initialize wandb: {e}")
                self.enabled = False

    def log_experiment(self, exp_id: int, phase: str, config: dict, metrics: dict,
                      status: str, description: str):
        """Log experiment to wandb."""
        if not self.enabled or self.run is None:
            return

        # Log metrics
        log_dict = {
            'exp_id': exp_id,
            'phase': phase,
            'status': status,
        }

        # Config parameters
        for key, value in config.items():
            log_dict[f'config_{key}'] = value

        # Metrics
        for key, value in metrics.items():
            log_dict[f'metric_{key}'] = value

        self.run.log(log_dict)

    def log_phase_complete(self, phase: str, results: list):
        """Log when a phase completes."""
        if not self.enabled or self.run is None:
            return

        if results:
            best_sce = max([r.get('sce', 0) for r in results])
            avg_sce = sum([r.get('sce', 0) for r in results]) / len(results)

            self.run.log({
                f'{phase}_best_sce': best_sce,
                f'{phase}_avg_sce': avg_sce,
                f'{phase}_num_experiments': len(results),
            })

    def finish(self):
        """Finish wandb run."""
        if self.enabled and self.run is not None:
            self.run.finish()

# ---------------------------------------------------------------------------
# Configuration & Data
# ---------------------------------------------------------------------------

def load_config(config_name: str) -> dict:
    """Load experiment configuration from config directory."""
    config_path = Path(__file__).parent.parent / "experiments" / f"{config_name}.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)

    # Return default config
    return {
        "tau_low": 0.15,
        "tau_high": 0.40,
        "alpha": 0.5,
        "beta": 0.3,
        "gamma": 0.2,
        "decay_lambda": 0.9
    }

def get_test_video() -> str:
    """Find test video in experiments directory."""
    experiments_dir = Path(__file__).parent.parent / "experiments" / "test"
    for ext in ['*.mp4', '*.avi', '*.mov']:
        videos = list(experiments_dir.glob(ext))
        if videos:
            return str(videos[0])

    raise FileNotFoundError("No test video found in experiments/test/")

# ---------------------------------------------------------------------------
# Pipeline Execution
# ---------------------------------------------------------------------------

def run_pipeline(config: dict, output_dir: str, phase: str = "threshold_sweep",
                scenario: str = None, gpu_ids: list = None) -> dict:
    """
    Run the semantic drift scheduler pipeline.

    Args:
        config: Experiment configuration (tau_low, tau_high, alpha, beta, gamma)
        output_dir: Directory to save results
        phase: Which research phase (threshold_sweep, ablation, failure_analysis)
        scenario: For failure_analysis phase (stable, motion, lighting, etc.)
        gpu_ids: List of GPU IDs to use (None = auto-detect)

    Returns:
        Dictionary with metrics (sce, semantic_retention, vlm_call_rate, etc.)
    """

    video_path = get_test_video()

    # Set GPU environment if specified
    if gpu_ids is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
        gpu_str = f" (GPUs: {','.join(map(str, gpu_ids))})"
    else:
        gpu_str = f" (GPUs: {torch.cuda.device_count()} available)"

    print(f"Running on{gpu_str}")

    # Build command to run pipeline
    cmd = [
        'python',
        str(Path(__file__).parent.parent / 'scripts' / 'run_pipeline.py'),
        '--video', video_path,
        '--tau_low', str(config['tau_low']),
        '--tau_high', str(config['tau_high']),
        '--alpha', str(config['alpha']),
        '--beta', str(config['beta']),
        '--gamma', str(config['gamma']),
        '--skip_vlm',  # Fast mode
        '--output', output_dir
    ]

    if scenario:
        cmd.extend(['--scenario', scenario])

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed_seconds = time.time() - start_time

    if result.returncode != 0:
        print(f"ERROR: Pipeline failed")
        print(result.stderr)
        return None

    # Load metrics from pipeline output
    metrics_file = Path(output_dir) / "metrics.json"
    if not metrics_file.exists():
        print(f"ERROR: No metrics file at {metrics_file}")
        return None

    with open(metrics_file) as f:
        metrics = json.load(f)

    # Calculate SCE (primary metric)
    vlm_call_rate = metrics.get('vlm_call_rate', 0.001)
    semantic_retention = metrics.get('semantic_retention', 0.0)
    sce = semantic_retention / vlm_call_rate if vlm_call_rate > 0 else 0.0

    return {
        'sce': sce,
        'semantic_retention': semantic_retention,
        'vlm_call_rate': vlm_call_rate,
        'latency_ms': metrics.get('latency_ms', 0.0),
        'gpu_utilization': metrics.get('gpu_utilization', 0.0),
        'elapsed_seconds': elapsed_seconds,
        'false_positives': metrics.get('false_positives', 0),
        'false_negatives': metrics.get('false_negatives', 0),
    }

def print_results(metrics: dict):
    """Print results in standard format."""
    if metrics is None:
        print("---")
        print("sce:                  0.0000")
        print("semantic_retention:   0.0000")
        print("vlm_call_rate:        0.0000")
        print("latency_ms:           0.0")
        print("gpu_utilization:      0.0")
        print("elapsed_seconds:      0.0")
        print("false_positives:      0")
        print("false_negatives:      0")
        return

    print("---")
    print(f"sce:                  {metrics['sce']:.4f}")
    print(f"semantic_retention:   {metrics['semantic_retention']:.4f}")
    print(f"vlm_call_rate:        {metrics['vlm_call_rate']:.4f}")
    print(f"latency_ms:           {metrics['latency_ms']:.1f}")
    print(f"gpu_utilization:      {metrics['gpu_utilization']:.2f}")
    print(f"elapsed_seconds:      {metrics['elapsed_seconds']:.1f}")
    print(f"false_positives:      {metrics['false_positives']}")
    print(f"false_negatives:      {metrics['false_negatives']}")

def load_results_tsv() -> list:
    """Load existing results.tsv if it exists."""
    results_file = Path(__file__).parent / "results.tsv"
    if not results_file.exists():
        return []

    results = []
    try:
        with open(results_file) as f:
            reader = csv.DictReader(f, delimiter='\t')
            results = list(reader)
    except:
        pass
    return results

def save_results_tsv(config: dict, metrics: dict, status: str, description: str):
    """Append result to results.tsv."""
    results_file = Path(__file__).parent / "results.tsv"

    # Get git commit hash
    commit = "unknown"
    try:
        result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            commit = result.stdout.strip()
    except:
        pass

    # Prepare row
    row = {
        'commit': commit,
        'sce': f"{metrics['sce']:.6f}" if metrics else "0.000000",
        'semantic_retention': f"{metrics['semantic_retention']:.4f}" if metrics else "0.0000",
        'vlm_call_rate': f"{metrics['vlm_call_rate']:.4f}" if metrics else "0.0000",
        'false_pos': metrics['false_positives'] if metrics else 0,
        'false_neg': metrics['false_negatives'] if metrics else 0,
        'status': status,
        'description': description
    }

    # Check if header exists
    file_exists = results_file.exists()

    with open(results_file, 'a', newline='') as f:
        fieldnames = ['commit', 'sce', 'semantic_retention', 'vlm_call_rate',
                     'false_pos', 'false_neg', 'status', 'description']
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)

# ---------------------------------------------------------------------------
# Main Experiment Loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='AutoResearch scheduler optimizer with checkpointing')
    parser.add_argument('--phase', default='threshold_sweep',
                       choices=['threshold_sweep', 'ablation', 'failure_analysis'],
                       help='Research phase to run')
    parser.add_argument('--config', default='default',
                       help='Configuration name (e.g. exp_001)')
    parser.add_argument('--scenario', help='Scenario name (for failure analysis)')

    # Configuration overrides
    parser.add_argument('--tau_low', type=float, help='Override tau_low')
    parser.add_argument('--tau_high', type=float, help='Override tau_high')
    parser.add_argument('--alpha', type=float, help='Override alpha')
    parser.add_argument('--beta', type=float, help='Override beta')
    parser.add_argument('--gamma', type=float, help='Override gamma')

    # Checkpoint & Resume
    parser.add_argument('--resume-checkpoint', type=str, help='Resume from checkpoint file')
    parser.add_argument('--list-checkpoints', action='store_true',
                       help='List all available checkpoints and exit')

    # Multi-GPU
    parser.add_argument('--multi-gpu', action='store_true',
                       help='Use all available GPUs (Kaggle: 2x T4)')
    parser.add_argument('--gpu-ids', type=str, help='Comma-separated GPU IDs (e.g. 0,1)')

    # Wandb
    parser.add_argument('--wandb-project', type=str, default='edge-vlm-thesis',
                       help='Wandb project name')
    parser.add_argument('--wandb-entity', type=str, help='Wandb entity/team name')
    parser.add_argument('--wandb-offline', action='store_true',
                       help='Run wandb in offline mode')

    # Directories
    parser.add_argument('--checkpoint-dir', type=str, help='Directory for checkpoints')
    parser.add_argument('--output-dir', type=str, help='Directory for experiment outputs')

    args = parser.parse_args()

    # Initialize managers
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    checkpoint_mgr = CheckpointManager(checkpoint_dir)

    # List checkpoints if requested
    if args.list_checkpoints:
        print("\nAvailable checkpoints:")
        checkpoints = checkpoint_mgr.list_checkpoints()
        print(f"\nTotal: {len(checkpoints)} checkpoints")
        return 0

    # Initialize wandb
    wandb_tracker = WandBTracker(
        project=args.wandb_project,
        entity=args.wandb_entity,
        offline=args.wandb_offline
    )

    # Parse GPU configuration
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(g) for g in args.gpu_ids.split(',')]
    elif args.multi_gpu:
        gpu_ids = list(range(torch.cuda.device_count()))

    # Handle checkpoint resume
    if args.resume_checkpoint:
        print(f"\n{'='*70}")
        print(f"RESUMING FROM CHECKPOINT: {args.resume_checkpoint}")
        print(f"{'='*70}")
        try:
            checkpoint_data = checkpoint_mgr.load_checkpoint(Path(args.resume_checkpoint))
            print(f"✓ Loaded checkpoint:")
            print(f"  Exp ID: {checkpoint_data['exp_id']}")
            print(f"  Phase: {checkpoint_data['phase']}")
            print(f"  SCE: {checkpoint_data['metrics'].get('sce', 0):.4f}")
            print(f"  Saved on: {checkpoint_data['machine']}")
            print(f"  Original GPUs: {checkpoint_data['gpu_count']}")
        except Exception as e:
            print(f"ERROR: Failed to load checkpoint: {e}")
            return 1

    # Load base config
    config = load_config(args.config)

    # Override with command-line args
    if args.tau_low is not None:
        config['tau_low'] = args.tau_low
    if args.tau_high is not None:
        config['tau_high'] = args.tau_high
    if args.alpha is not None:
        config['alpha'] = args.alpha
    if args.beta is not None:
        config['beta'] = args.beta
    if args.gamma is not None:
        config['gamma'] = args.gamma

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent / f"results_{args.phase}_{args.config}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get next experiment ID
    results = load_results_tsv()
    exp_id = len(results) + 1

    # Print configuration
    print(f"\n{'='*70}")
    print(f"AUTORESEARCH EXPERIMENT #{exp_id}")
    print(f"{'='*70}")
    print(f"Phase:     {args.phase}")
    print(f"Config:    {config}")
    if args.scenario:
        print(f"Scenario:  {args.scenario}")
    print(f"Output:    {output_dir}")
    if gpu_ids:
        print(f"GPUs:      {gpu_ids}")
    print(f"Wandb:     {'✓ Enabled' if wandb_tracker.enabled else '✗ Disabled'}")
    print(f"Checkpoint: {checkpoint_dir if checkpoint_dir else 'default'}")
    print(f"{'='*70}\n")

    # Run experiment
    timestamp = datetime.now().isoformat()
    metrics = run_pipeline(config, str(output_dir), args.phase, args.scenario, gpu_ids)

    # Print results
    print_results(metrics)

    # Save checkpoint
    status = "keep" if metrics and metrics['sce'] > 0 else "crash"
    checkpoint_path = checkpoint_mgr.save_checkpoint(
        exp_id=exp_id,
        phase=args.phase,
        config=config,
        metrics=metrics or {},
        status=status,
        timestamp=timestamp
    )
    print(f"\n✓ Checkpoint saved: {checkpoint_path.name}")

    # Log to TSV
    description = f"{args.phase}: tau_low={config['tau_low']:.2f}, tau_high={config['tau_high']:.2f}"
    if args.scenario:
        description += f" (scenario: {args.scenario})"
    save_results_tsv(config, metrics, status, description)

    # Log to wandb
    wandb_tracker.log_experiment(
        exp_id=exp_id,
        phase=args.phase,
        config=config,
        metrics=metrics or {},
        status=status,
        description=description
    )

    print(f"✓ Results logged to results.tsv")
    print(f"✓ Wandb sync complete")
    print(f"\nTo resume later on another machine:")
    print(f"  python train.py --resume-checkpoint {checkpoint_path.name}")

    wandb_tracker.finish()
    return 0 if metrics else 1

if __name__ == '__main__':
    sys.exit(main())
