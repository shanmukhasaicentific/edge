"""
src/observatory/output_writer.py

Saves all observatory output files to disk.

Mandatory outputs (from prompt spec):
    1. drift_values.csv
    2. semantic_events.csv
    3. vlm_invocations.csv
    4. semantic_regions.csv
    5. object_persistence.csv
    6. interaction_events.csv
    7. semantic_evolution.json
"""

import json
import os
from dataclasses import asdict

import pandas as pd

from src.observatory.analytics import ObservatoryResult


class ObservatoryOutputWriter:
    """
    Writes all observatory analysis results to disk.

    Args:
        output_dir: directory to write all files into
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write_all(self, result: ObservatoryResult) -> dict:
        """
        Write all output files. Returns dict of {filename: path}.
        """
        paths = {}
        paths["drift_values.csv"]       = self._write_drift_values(result)
        paths["semantic_events.csv"]    = self._write_semantic_events(result)
        paths["vlm_invocations.csv"]    = self._write_vlm_invocations(result)
        paths["semantic_regions.csv"]   = self._write_semantic_regions(result)
        paths["object_persistence.csv"] = self._write_object_persistence(result)
        paths["interaction_events.csv"] = self._write_interaction_events(result)
        paths["semantic_evolution.json"]= self._write_semantic_evolution(result)

        print(f"\n[Observatory] Output files written to: {self.output_dir}")
        for name, path in paths.items():
            size = os.path.getsize(path) / 1024
            print(f"  {name:<30} {size:.1f} KB")

        return paths

    def _write_drift_values(self, result: ObservatoryResult) -> str:
        """Per-frame semantic evolution signals."""
        cols = [
            "frame_id", "timestamp_ms",
            "drift_total", "drift_embed", "drift_objects", "drift_track",
            "semantic_acceleration", "semantic_acceleration_smooth",
            "semantic_volatility", "semantic_velocity",
            "scene_entropy", "semantic_progress", "semantic_progress_norm",
            "vlm_density", "frame_semantic_state",
            "tier", "vlm_called", "cache_hit", "latency_ms",
            "n_detections", "n_tracks",
        ]
        df = result.enriched_df
        available = [c for c in cols if c in df.columns]
        path = os.path.join(self.output_dir, "drift_values.csv")
        df[available].to_csv(path, index=False)
        return path

    def _write_semantic_events(self, result: ObservatoryResult) -> str:
        """Detected semantic events (spikes, bursts, VLM triggers)."""
        rows = [asdict(e) for e in result.semantic_events]
        df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["event_id", "event_type", "frame_id", "timestamp_ms",
                     "drift_at_event", "acceleration_at_event", "description"]
        )
        path = os.path.join(self.output_dir, "semantic_events.csv")
        df.to_csv(path, index=False)
        return path

    def _write_vlm_invocations(self, result: ObservatoryResult) -> str:
        """Frames where VLM was actually called, with context."""
        df = result.enriched_df
        if "vlm_called" in df.columns:
            vlm_df = df[df["vlm_called"] == True].copy()
        else:
            vlm_df = pd.DataFrame()

        cols = ["frame_id", "timestamp_ms", "drift_total",
                "semantic_acceleration", "semantic_volatility",
                "tier", "latency_ms", "n_detections", "n_tracks"]
        available = [c for c in cols if c in vlm_df.columns]
        path = os.path.join(self.output_dir, "vlm_invocations.csv")
        vlm_df[available].to_csv(path, index=False) if available else \
            pd.DataFrame().to_csv(path, index=False)
        return path

    def _write_semantic_regions(self, result: ObservatoryResult) -> str:
        """Detected semantic regions (stable / transition / volatile)."""
        all_regions = (
            result.stable_regions +
            result.transition_regions +
            result.volatile_regions
        )
        all_regions.sort(key=lambda r: r.start_frame)
        rows = [asdict(r) for r in all_regions]
        df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["region_id", "region_type", "start_frame", "end_frame",
                     "duration_frames", "mean_drift", "max_drift",
                     "mean_volatility", "vlm_calls_in_region", "dominant_state"]
        )
        path = os.path.join(self.output_dir, "semantic_regions.csv")
        df.to_csv(path, index=False)
        return path

    def _write_object_persistence(self, result: ObservatoryResult) -> str:
        """Object persistence records."""
        rows = [asdict(r) for r in result.object_persistence]
        df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["track_id", "first_frame", "last_frame",
                     "lifespan_frames", "class_id"]
        )
        path = os.path.join(self.output_dir, "object_persistence.csv")
        df.to_csv(path, index=False)
        return path

    def _write_interaction_events(self, result: ObservatoryResult) -> str:
        """
        Interaction events — frames where n_tracks changed AND drift was high.
        Proxy for human-object interaction events.
        """
        df = result.enriched_df
        interaction_rows = []

        if "n_tracks" in df.columns and "drift_total" in df.columns:
            track_delta = df["n_tracks"].diff().fillna(0)
            interaction_mask = (track_delta.abs() > 0) & \
                               (df["drift_total"] > df["drift_total"].mean())

            interaction_df = df[interaction_mask][[
                "frame_id", "timestamp_ms", "drift_total",
                "semantic_acceleration", "n_tracks", "n_detections"
            ]].copy() if interaction_mask.any() else pd.DataFrame()

            if not interaction_df.empty:
                interaction_df["track_delta"] = track_delta[interaction_mask].values
                interaction_df.to_csv(
                    os.path.join(self.output_dir, "interaction_events.csv"),
                    index=False
                )
                return os.path.join(self.output_dir, "interaction_events.csv")

        path = os.path.join(self.output_dir, "interaction_events.csv")
        pd.DataFrame().to_csv(path, index=False)
        return path

    def _write_semantic_evolution(self, result: ObservatoryResult) -> str:
        """Complete semantic evolution summary as JSON."""
        summary = {
            "run_summary": {
                "total_frames": result.total_frames,
                "total_duration_ms": result.total_duration_ms,
                "fps_estimate": result.fps_estimate,
                "vlm_call_rate": result.vlm_call_rate,
            },
            "drift_statistics": {
                "mean": result.mean_drift,
                "median": result.median_drift,
                "std": result.drift_std,
                "max": result.max_drift,
                "mean_acceleration": result.mean_acceleration,
                "mean_volatility": result.mean_volatility,
            },
            "region_summary": {
                "stable_count": len(result.stable_regions),
                "transition_count": len(result.transition_regions),
                "volatile_count": len(result.volatile_regions),
                "stable_frames": sum(r.duration_frames for r in result.stable_regions),
                "transition_frames": sum(r.duration_frames for r in result.transition_regions),
                "volatile_frames": sum(r.duration_frames for r in result.volatile_regions),
            },
            "event_summary": {
                "total_events": len(result.semantic_events),
                "spike_events": sum(1 for e in result.semantic_events if e.event_type == "spike"),
                "burst_events": sum(1 for e in result.semantic_events if e.event_type == "burst"),
                "vlm_trigger_events": sum(1 for e in result.semantic_events if e.event_type == "vlm_trigger"),
            },
            "research_findings": {
                "semantic_sparsity": f"{(1 - result.vlm_call_rate) * 100:.1f}% of frames required no VLM reasoning",
                "stability_fraction": f"{len(result.stable_regions) / max(len(result.stable_regions) + len(result.transition_regions), 1) * 100:.1f}% of detected regions were stable",
                "hypothesis_validation": "Semantic understanding evolves more slowly than raw visual appearance" if result.vlm_call_rate < 0.5 else "High semantic activity detected",
            }
        }

        path = os.path.join(self.output_dir, "semantic_evolution.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        return path
