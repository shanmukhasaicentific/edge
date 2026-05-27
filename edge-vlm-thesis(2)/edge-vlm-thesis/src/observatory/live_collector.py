"""
src/observatory/live_collector.py

Live Observatory Collector.

Runs INSIDE the pipeline (run_pipeline.py) to collect
enriched per-frame data during inference — not post-hoc.

This is lighter than the full analytics engine. It:
    - collects raw signals every frame
    - computes rolling metrics in real-time
    - triggers observatory analysis at end of run

Integration in run_pipeline.py:
    collector = LiveObservatoryCollector(enabled=True)

    # Inside frame loop:
    collector.record(
        frame_id=frame_id,
        drift_components=drift_components,
        monitor_result=monitor_result,
        scheduler_decision=decision,
        percept=percept,
        latency_ms=latency_ms,
    )

    # After loop:
    collector.finalize(output_dir="results/observatory/live/")
"""

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class LiveFrame:
    """Single frame record collected during live pipeline run."""
    frame_id: int
    timestamp_ms: float
    drift_total: float
    drift_embed: float
    drift_objects: float
    drift_track: float
    semantic_state: str
    effective_drift: float
    drift_accumulator: float
    silent_drift: bool
    anchor_updated: bool
    tier: int
    vlm_called: bool
    cache_hit: bool
    latency_ms: float
    n_detections: int
    n_tracks: int
    vlm_output: str = ""


class LiveObservatoryCollector:
    """
    Lightweight live collector that runs inside run_pipeline.py.

    Collects richer data than the standard Evaluator — specifically
    the semantic monitoring signals (state, accumulator, anchor events)
    that are only available during live pipeline execution.

    Args:
        enabled:            set False to disable with zero overhead
        flush_every:        write buffer to disk every N frames (0=disable)
        rolling_window:     window for real-time volatility estimate
    """

    def __init__(
        self,
        enabled: bool = True,
        flush_every: int = 0,
        rolling_window: int = 30,
    ):
        self.enabled = enabled
        self.flush_every = flush_every
        self.rolling_window = rolling_window

        self._records: List[LiveFrame] = []
        self._drift_buffer: List[float] = []
        self._start_time = time.perf_counter()
        self._frame_count = 0

    def record(
        self,
        frame_id: int,
        drift_components,        # DriftComponents from drift.py
        monitor_result,          # MonitoringResult from monitoring.py
        scheduler_decision,      # SchedulerDecision from vlm_scheduler.py
        percept,                 # RoboticPercept from perception_layer.py
        latency_ms: float,
        vlm_output: str = "",
    ):
        """
        Record one frame's data. Call every processed frame inside the loop.
        """
        if not self.enabled:
            return

        self._frame_count += 1
        timestamp_ms = (time.perf_counter() - self._start_time) * 1000

        record = LiveFrame(
            frame_id=frame_id,
            timestamp_ms=timestamp_ms,
            drift_total=drift_components.d_total,
            drift_embed=drift_components.d_embed,
            drift_objects=drift_components.d_objects,
            drift_track=drift_components.d_track,
            semantic_state=monitor_result.semantic_state.value,
            effective_drift=monitor_result.effective_drift,
            drift_accumulator=monitor_result.drift_accumulator,
            silent_drift=monitor_result.silent_drift_detected,
            anchor_updated=monitor_result.anchor_updated,
            tier=int(scheduler_decision.tier),
            vlm_called=scheduler_decision.invoke_vlm,
            cache_hit=bool(
                scheduler_decision.cache_result and
                scheduler_decision.cache_result.hit
            ),
            latency_ms=latency_ms,
            n_detections=len(percept.objects),
            n_tracks=len(percept.objects),
            vlm_output=vlm_output[:200] if vlm_output else "",
        )
        self._records.append(record)
        self._drift_buffer.append(drift_components.d_total)

        # Flush buffer periodically if requested
        if self.flush_every > 0 and self._frame_count % self.flush_every == 0:
            print(f"[LiveCollector] Frame {frame_id}: "
                  f"state={monitor_result.semantic_state.value} | "
                  f"drift={drift_components.d_total:.3f} | "
                  f"tier={int(scheduler_decision.tier)} | "
                  f"accum={monitor_result.drift_accumulator:.3f}")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert collected records to DataFrame."""
        if not self._records:
            return pd.DataFrame()

        rows = []
        for r in self._records:
            rows.append({
                "frame_id": r.frame_id,
                "timestamp_ms": r.timestamp_ms,
                "drift_total": r.drift_total,
                "drift_embed": r.drift_embed,
                "drift_objects": r.drift_objects,
                "drift_track": r.drift_track,
                "semantic_state": r.semantic_state,
                "effective_drift": r.effective_drift,
                "drift_accumulator": r.drift_accumulator,
                "silent_drift": r.silent_drift,
                "anchor_updated": r.anchor_updated,
                "tier": r.tier,
                "vlm_called": r.vlm_called,
                "cache_hit": r.cache_hit,
                "latency_ms": r.latency_ms,
                "n_detections": r.n_detections,
                "n_tracks": r.n_tracks,
                "vlm_output": r.vlm_output,
            })
        return pd.DataFrame(rows)

    def finalize(self, output_dir: str, run_observatory: bool = True) -> Optional[str]:
        """
        Save enriched frame log and optionally trigger full observatory analysis.

        Args:
            output_dir:       where to save enriched_frame_log.csv
            run_observatory:  if True, run full observatory analysis after saving

        Returns:
            Path to saved CSV
        """
        if not self.enabled or not self._records:
            return None

        os.makedirs(output_dir, exist_ok=True)

        df = self.to_dataframe()
        csv_path = os.path.join(output_dir, "enriched_frame_log.csv")
        df.to_csv(csv_path, index=False)
        print(f"[LiveCollector] Saved enriched log: {csv_path} ({len(df):,} frames)")

        if run_observatory:
            print("[LiveCollector] Running observatory analysis on live data...")
            obs_dir = os.path.join(output_dir, "observatory")
            try:
                from src.observatory.analytics import SemanticAnalyticsEngine
                from src.observatory.output_writer import ObservatoryOutputWriter
                from src.observatory.visualizer import SemanticVisualizer
                from src.observatory.dashboard import ObservatoryDashboard

                engine = SemanticAnalyticsEngine()
                result = engine.analyze(df)

                writer = ObservatoryOutputWriter(obs_dir)
                writer.write_all(result)

                vis = SemanticVisualizer(
                    output_dir=os.path.join(obs_dir, "figures"),
                    downsample=max(1, len(df) // 5000),
                )
                vis.generate_all(result)

                dash = ObservatoryDashboard(obs_dir)
                dash.build(result)

                print(f"[LiveCollector] Observatory output: {obs_dir}")
            except Exception as e:
                print(f"[LiveCollector] Observatory analysis failed: {e}")

        return csv_path

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def current_volatility(self) -> float:
        """Real-time volatility estimate from recent drift buffer."""
        if len(self._drift_buffer) < 2:
            return 0.0
        window = self._drift_buffer[-self.rolling_window:]
        return float(np.std(window))
