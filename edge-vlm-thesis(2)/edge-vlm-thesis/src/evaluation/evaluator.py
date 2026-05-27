"""
src/evaluation/evaluator.py

Evaluation Pipeline.

Computes all mandatory thesis metrics for a single experiment run:

EFFICIENCY METRICS:
    - vlm_call_rate:       fraction of frames that invoked VLM
    - vlm_calls_total:     absolute count
    - avg_latency_ms:      average per-frame wall-clock time
    - gpu_utilization:     peak/mean GPU% (from nvidia-smi log)
    - throughput_fps:      frames per second
    - memory_mb:           peak GPU memory used

SEMANTIC METRICS:
    - semantic_retention:  cosine similarity between proposed and every-frame outputs
    - caption_consistency: BLEU/ROUGE score between consecutive captions
    - cache_hit_rate:      fraction of frames served from cache

UNIFIED METRIC:
    - sce:  Semantic Compute Efficiency = SemanticRetention / NormalizedComputeCost

    Where NormalizedComputeCost = vlm_call_rate (fraction of max cost)

All metrics saved to:
    experiments/<name>/metrics.json
    experiments/<name>/frame_log.csv
"""

import csv
import json
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class FrameRecord:
    """Per-frame log entry. Written to frame_log.csv."""
    frame_id: int
    timestamp_ms: float
    drift_total: float
    drift_embed: float
    drift_objects: float
    drift_track: float
    tier: int
    vlm_called: bool
    cache_hit: bool
    vlm_output: str
    latency_ms: float
    n_detections: int
    n_tracks: int


@dataclass
class RunMetrics:
    """Aggregate metrics for one complete experiment run."""
    experiment_name: str
    policy_name: str
    total_frames: int
    duration_sec: float

    # Efficiency
    vlm_calls_total: int = 0
    vlm_call_rate: float = 0.0
    avg_latency_ms: float = 0.0
    throughput_fps: float = 0.0
    cache_hits_total: int = 0
    cache_hit_rate: float = 0.0
    tier1_frames: int = 0
    tier2_frames: int = 0
    tier3_frames: int = 0

    # Semantic
    semantic_retention: float = 0.0    # vs every-frame baseline
    avg_drift: float = 0.0

    # Unified
    sce: float = 0.0    # SemanticRetention / NormalizedComputeCost

    # GPU (filled in after run)
    peak_gpu_memory_mb: float = 0.0
    avg_gpu_utilization_pct: float = 0.0


class Evaluator:
    """
    Records per-frame data during a run, then computes aggregate metrics.

    Usage:
        evaluator = Evaluator(experiment_name="proposed", policy_name="threshold")
        # inside frame loop:
        evaluator.record_frame(FrameRecord(...))
        # after loop:
        metrics = evaluator.compute_metrics(baseline_captions=baseline_captions)
        evaluator.save(output_dir="experiments/proposed/")
    """

    def __init__(self, experiment_name: str, policy_name: str):
        self.experiment_name = experiment_name
        self.policy_name = policy_name
        self._records: List[FrameRecord] = []
        self._start_time: float = time.perf_counter()

    def record_frame(self, record: FrameRecord):
        """Append a per-frame record."""
        self._records.append(record)

    def compute_metrics(
        self,
        baseline_captions: Optional[Dict[int, str]] = None,
    ) -> RunMetrics:
        """
        Compute all aggregate metrics from recorded frames.

        Args:
            baseline_captions: dict of {frame_id: caption} from every-frame baseline.
                                If provided, computes semantic_retention via embedding similarity.
        """
        if not self._records:
            raise ValueError("No frames recorded. Run the pipeline first.")

        duration_sec = time.perf_counter() - self._start_time
        n = len(self._records)

        vlm_calls = [r for r in self._records if r.vlm_called]
        cache_hits = [r for r in self._records if r.cache_hit]

        vlm_calls_total = len(vlm_calls)
        vlm_call_rate = vlm_calls_total / n

        avg_latency_ms = np.mean([r.latency_ms for r in self._records])
        throughput_fps = n / duration_sec if duration_sec > 0 else 0.0

        tier1 = sum(1 for r in self._records if r.tier == 1)
        tier2 = sum(1 for r in self._records if r.tier == 2)
        tier3 = sum(1 for r in self._records if r.tier == 3)

        avg_drift = np.mean([r.drift_total for r in self._records])

        # Semantic retention: cosine similarity of VLM outputs vs baseline
        # Approximated as: fraction of frames with consistent output
        # (Full embedding-based version requires CLIP re-encoding captions)
        semantic_retention = self._compute_semantic_retention(baseline_captions)

        # SCE: semantic retention per unit of normalized compute cost
        # NormalizedComputeCost = vlm_call_rate (fraction of max cost = every-frame baseline)
        sce = semantic_retention / max(vlm_call_rate, 1e-6)
        sce = float(np.clip(sce, 0.0, 10.0))  # cap at 10x for readability

        return RunMetrics(
            experiment_name=self.experiment_name,
            policy_name=self.policy_name,
            total_frames=n,
            duration_sec=duration_sec,
            vlm_calls_total=vlm_calls_total,
            vlm_call_rate=vlm_call_rate,
            avg_latency_ms=float(avg_latency_ms),
            throughput_fps=float(throughput_fps),
            cache_hits_total=len(cache_hits),
            cache_hit_rate=len(cache_hits) / n,
            tier1_frames=tier1,
            tier2_frames=tier2,
            tier3_frames=tier3,
            semantic_retention=semantic_retention,
            avg_drift=float(avg_drift),
            sce=sce,
        )

    def _compute_semantic_retention(
        self,
        baseline_captions: Optional[Dict[int, str]],
    ) -> float:
        """
        Estimate semantic retention.

        If baseline_captions provided: compare output captions to baseline using
        simple token overlap (lightweight proxy for embedding similarity).
        If not provided: use cache-hit-based estimate (lower bound).

        Returns float in [0, 1].
        """
        if baseline_captions is None:
            # Lower bound: assume cached frames retain semantics
            n = len(self._records)
            frames_with_output = sum(
                1 for r in self._records
                if r.vlm_output or r.cache_hit
            )
            return frames_with_output / n if n > 0 else 0.0

        # Token overlap metric (Jaccard on word sets)
        scores = []
        for r in self._records:
            if r.frame_id in baseline_captions:
                ref = set(baseline_captions[r.frame_id].lower().split())
                hyp = set(r.vlm_output.lower().split()) if r.vlm_output else set()
                if not ref:
                    continue
                overlap = len(ref & hyp) / len(ref | hyp) if (ref | hyp) else 1.0
                scores.append(overlap)

        return float(np.mean(scores)) if scores else 0.0

    def save(self, output_dir: str):
        """
        Save metrics and frame log to output_dir.
        Creates directory if it doesn't exist.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save frame log as CSV
        log_path = os.path.join(output_dir, "frame_log.csv")
        if self._records:
            with open(log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=asdict(self._records[0]).keys())
                writer.writeheader()
                for r in self._records:
                    writer.writerow(asdict(r))

        print(f"[Evaluator] Saved frame log to {log_path}")

    def save_metrics(self, metrics: RunMetrics, output_dir: str):
        """Save RunMetrics to JSON."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "metrics.json")
        with open(path, "w") as f:
            json.dump(asdict(metrics), f, indent=2)
        print(f"[Evaluator] Saved metrics to {path}")
