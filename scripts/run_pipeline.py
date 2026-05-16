"""
scripts/run_pipeline.py

Main Pipeline Runner.

Wires all modules into the complete system:

    Video → FrameFilter → YOLO → ByteTrack → CLIP →
    SemanticMemory → DriftEstimator → Policy → Scheduler →
    VLM → Cache → RoboticPerceptionLayer → Evaluator

Usage:
    python scripts/run_pipeline.py \
        --video path/to/video.mp4 \
        --policy threshold \
        --tau_low 0.15 \
        --tau_high 0.40 \
        --experiment_name proposed_run \
        --output_dir experiments/proposed/

Baseline runs:
    python scripts/run_pipeline.py --policy every_frame --experiment_name baseline_every_frame
    python scripts/run_pipeline.py --policy fixed_interval --interval 30 --experiment_name uniform_30
    python scripts/run_pipeline.py --policy motion_gating --experiment_name motion_gating
    python scripts/run_pipeline.py --policy embedding_threshold --tau 0.3 --experiment_name embed_gate
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.frame_filter import FrameFilter
from src.detection.yolo_detector import YOLODetector
from src.tracking.bytetrack_wrapper import ByteTrackWrapper
from src.embeddings.clip_extractor import CLIPExtractor
from src.semantic_memory.memory import TemporalSemanticMemory
from src.semantic_memory.drift import SemanticDriftEstimator
from src.policies.adaptive_policy import build_policy
from src.scheduler.semantic_cache import SemanticCache
from src.scheduler.vlm_scheduler import VLMScheduler, ComputeTier
from src.vlm.moondream_wrapper import MoondreamWrapper
from src.robotics.perception_layer import RoboticPerceptionLayer
from src.evaluation.evaluator import Evaluator, FrameRecord


def parse_args():
    parser = argparse.ArgumentParser(description="Edge VLM Semantic Scheduler Pipeline")

    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--policy", type=str, default="threshold",
                        choices=["threshold", "every_frame", "fixed_interval",
                                 "motion_gating", "embedding_threshold"])
    parser.add_argument("--experiment_name", type=str, default="run")
    parser.add_argument("--output_dir", type=str, default="experiments/default/")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Threshold policy args
    parser.add_argument("--tau_low", type=float, default=0.15)
    parser.add_argument("--tau_high", type=float, default=0.40)

    # Fixed interval args
    parser.add_argument("--interval", type=int, default=30)

    # Motion gating args
    parser.add_argument("--motion_threshold", type=float, default=10.0)

    # Embedding threshold args
    parser.add_argument("--tau", type=float, default=0.30)

    # Drift weights
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.2)

    # Memory
    parser.add_argument("--decay_lambda", type=float, default=0.9)

    # Misc
    parser.add_argument("--max_frames", type=int, default=-1, help="-1 for all frames")
    parser.add_argument("--skip_vlm", action="store_true", help="Disable VLM loading (faster testing)")
    parser.add_argument("--save_embeddings", action="store_true", help="Cache embeddings to disk")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def build_policy_from_args(args):
    """Build the correct policy from CLI args."""
    if args.policy == "threshold":
        return build_policy("threshold", tau_low=args.tau_low, tau_high=args.tau_high)
    elif args.policy == "every_frame":
        return build_policy("every_frame")
    elif args.policy == "fixed_interval":
        return build_policy("fixed_interval", interval=args.interval)
    elif args.policy == "motion_gating":
        return build_policy("motion_gating", motion_threshold=args.motion_threshold)
    elif args.policy == "embedding_threshold":
        return build_policy("embedding_threshold", tau=args.tau)
    else:
        raise ValueError(f"Unknown policy: {args.policy}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Save run config
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"[Pipeline] Config saved to {config_path}")

    # ─── Initialize modules ───────────────────────────────────────────────
    print(f"[Pipeline] Initializing modules on device: {args.device}")

    frame_filter = FrameFilter(motion_threshold=2.0, blur_threshold=50.0)
    detector = YOLODetector(conf_threshold=0.3, device=args.device)
    tracker = ByteTrackWrapper()
    embedder = CLIPExtractor(device=args.device)
    memory = TemporalSemanticMemory(decay_lambda=args.decay_lambda)
    drift_estimator = SemanticDriftEstimator(
        alpha=args.alpha, beta=args.beta, gamma=args.gamma
    )
    policy = build_policy_from_args(args)
    cache = SemanticCache(tau_low=args.tau_low, tau_high=args.tau_high)
    scheduler = VLMScheduler(
        tau_low=args.tau_low, tau_high=args.tau_high, cache=cache
    )
    robotic_layer = RoboticPerceptionLayer()
    evaluator = Evaluator(
        experiment_name=args.experiment_name,
        policy_name=args.policy,
    )

    vlm = None
    if not args.skip_vlm:
        vlm = MoondreamWrapper(device=args.device)
        vlm.load()

    # ─── Open video ───────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[Pipeline] Video: {args.video} | FPS: {fps:.1f} | Frames: {total_frames}")

    frame_id = 0
    current_vlm_output = ""

    # ─── Main Frame Loop ──────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if args.max_frames > 0 and frame_id >= args.max_frames:
            break

        t_frame_start = time.perf_counter()

        # ── Tier 1: Lightweight Frame Filtering ──
        filter_result = frame_filter.filter(frame)
        if not filter_result.keep:
            frame_id += 1
            continue

        # ── Tier 1: YOLO Detection ──
        det_result = detector.detect(frame, frame_id=frame_id)
        bytetrack_input = detector.to_bytetrack_format(det_result)

        # ── Tier 1: ByteTrack Tracking ──
        track_result = tracker.update(bytetrack_input, frame, frame_id=frame_id)

        # ── Tier 1: CLIP Embedding ──
        embedding, clip_ms = embedder.timed_extract(frame)

        # ── Semantic Memory Update ──
        memory_state = memory.update(embedding)

        # ── Semantic Drift Estimation ──
        drift_components = drift_estimator.compute(
            current_embedding=embedding,
            memory_embedding=memory_state,
            detections=det_result.detections,
            tracking_result=track_result,
        )

        # ── Adaptive Compute Policy ──
        decision = scheduler.decide(drift=drift_components.d_total, frame_id=frame_id)

        # ── VLM Invocation (Tier 2 or 3) ──
        cache_hit = False
        vlm_called = False

        if decision.cache_result and decision.cache_result.hit:
            current_vlm_output = decision.cache_result.output
            cache_hit = True
        elif decision.invoke_vlm and vlm is not None:
            vlm_out = vlm.infer(frame, tier=int(decision.tier), frame_id=frame_id)
            current_vlm_output = vlm_out.caption
            scheduler.register_vlm_call(
                frame_id=frame_id,
                vlm_output=current_vlm_output,
                embedding=embedding,
            )
            memory.register_vlm_call(embedding)
            vlm_called = True
        elif decision.invoke_vlm and vlm is None:
            # VLM skipped for testing — record as would-be call
            current_vlm_output = f"[VLM_SKIPPED at frame {frame_id}]"
            vlm_called = True

        # ── Robotic Perception Layer ──
        percept = robotic_layer.process(
            frame_id=frame_id,
            drift=drift_components.d_total,
            detections=det_result.detections,
            tracks=track_result.tracks,
            new_track_ids=track_result.new_ids,
            vlm_caption=current_vlm_output if vlm_called else None,
            cache_hit=cache_hit,
            compute_tier=int(decision.tier),
        )

        # ── Record Frame ──
        latency_ms = (time.perf_counter() - t_frame_start) * 1000
        evaluator.record_frame(FrameRecord(
            frame_id=frame_id,
            timestamp_ms=frame_id / fps * 1000,
            drift_total=drift_components.d_total,
            drift_embed=drift_components.d_embed,
            drift_objects=drift_components.d_objects,
            drift_track=drift_components.d_track,
            tier=int(decision.tier),
            vlm_called=vlm_called,
            cache_hit=cache_hit,
            vlm_output=current_vlm_output[:200],  # truncate for CSV
            latency_ms=latency_ms,
            n_detections=len(det_result.detections),
            n_tracks=len(track_result.tracks),
        ))

        if args.verbose and frame_id % 30 == 0:
            print(
                f"[Frame {frame_id:04d}] "
                f"Drift: {drift_components.d_total:.3f} | "
                f"Tier: {int(decision.tier)} | "
                f"VLM: {vlm_called} | "
                f"Cache: {cache_hit} | "
                f"Latency: {latency_ms:.1f}ms | "
                f"Stability: {percept.scene_stability.value}"
            )

        frame_id += 1

    cap.release()
    print(f"\n[Pipeline] Processed {frame_id} frames.")

    # ─── Compute and Save Metrics ─────────────────────────────────────────
    metrics = evaluator.compute_metrics()
    evaluator.save(args.output_dir)
    evaluator.save_metrics(metrics, args.output_dir)

    print(f"\n{'='*50}")
    print(f"  EXPERIMENT: {args.experiment_name}")
    print(f"{'='*50}")
    print(f"  Frames processed:    {metrics.total_frames}")
    print(f"  VLM calls total:     {metrics.vlm_calls_total} ({metrics.vlm_call_rate:.1%})")
    print(f"  Cache hit rate:      {metrics.cache_hit_rate:.1%}")
    print(f"  Avg latency:         {metrics.avg_latency_ms:.1f} ms")
    print(f"  Throughput:          {metrics.throughput_fps:.1f} FPS")
    print(f"  Semantic retention:  {metrics.semantic_retention:.3f}")
    print(f"  SCE:                 {metrics.sce:.3f}")
    print(f"  Tier distribution:   T1={metrics.tier1_frames} T2={metrics.tier2_frames} T3={metrics.tier3_frames}")
    print(f"{'='*50}")
    print(f"  Results: {args.output_dir}")


if __name__ == "__main__":
    main()
