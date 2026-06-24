"""
src/observatory/analytics.py

Semantic Evolution Analytics Engine.

Takes raw per-frame data (from frame_log.csv or live pipeline)
and computes the full suite of semantic evolution metrics:

    1. Semantic Drift          — D_t per frame (already in log)
    2. Semantic Acceleration   — A_t = D_t - D_{t-1}
    3. Semantic Volatility     — V_t = Var(D_{t-k:t}) over rolling window
    4. Semantic Velocity       — smoothed rate of drift change
    5. Scene Entropy           — Shannon entropy of object class distribution
    6. Object Persistence      — per-track lifespan in frames
    7. Semantic Stability Windows — contiguous regions of low drift
    8. Semantic Transition Zones  — contiguous regions of high drift
    9. Semantic Progress Function — cumulative semantic evolution
   10. VLM Trigger Density     — rolling VLM invocation rate

These metrics feed the visualizer and event detector.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class SemanticRegion:
    """A contiguous temporal region with a consistent semantic character."""
    region_id: int
    region_type: str          # 'stable' | 'transition' | 'volatile'
    start_frame: int
    end_frame: int
    duration_frames: int
    mean_drift: float
    max_drift: float
    mean_volatility: float
    vlm_calls_in_region: int
    dominant_state: str       # from SemanticMonitor state labels


@dataclass
class ObjectPersistenceRecord:
    """Lifespan record for a single tracked object."""
    track_id: int
    first_frame: int
    last_frame: int
    lifespan_frames: int
    class_id: int


@dataclass
class SemanticEvent:
    """A detected semantic event in the timeline."""
    event_id: int
    event_type: str           # 'transition' | 'novel' | 'burst' | 'plateau' | 'spike'
    frame_id: int
    timestamp_ms: float
    drift_at_event: float
    acceleration_at_event: float
    description: str


@dataclass
class ObservatoryResult:
    """Complete output of the analytics engine for one run."""
    # Input summary
    total_frames: int
    total_duration_ms: float
    fps_estimate: float

    # Per-frame enriched DataFrame
    enriched_df: pd.DataFrame

    # Aggregate metrics
    mean_drift: float
    median_drift: float
    drift_std: float
    max_drift: float
    mean_acceleration: float
    mean_volatility: float

    # Regions
    stable_regions: List[SemanticRegion]
    transition_regions: List[SemanticRegion]
    volatile_regions: List[SemanticRegion]

    # Events
    semantic_events: List[SemanticEvent]

    # Object persistence
    object_persistence: List[ObjectPersistenceRecord]

    # VLM analysis
    vlm_call_rate: float
    vlm_trigger_density: pd.Series   # rolling VLM rate

    # Semantic progress
    semantic_progress: pd.Series     # cumulative drift


# ─── Analytics Engine ─────────────────────────────────────────────────────────

class SemanticAnalyticsEngine:
    """
    Computes the full semantic evolution analytics suite from a frame log.

    Input: pandas DataFrame with columns from frame_log.csv:
        frame_id, timestamp_ms, drift_total, drift_embed, drift_objects,
        drift_track, tier, vlm_called, cache_hit, vlm_output, latency_ms,
        n_detections, n_tracks

    Args:
        volatility_window:   frames over which to compute drift variance
        acceleration_smooth: smoothing window for acceleration signal
        stability_tau:       drift below this → stable region
        transition_tau:      drift above this → transition region
        min_region_length:   minimum frames for a region to be recorded
        vlm_density_window:  rolling window for VLM trigger density
    """

    def __init__(
        self,
        volatility_window: int = 30,
        acceleration_smooth: int = 5,
        stability_tau: float = 0.10,
        transition_tau: float = 0.35,
        min_region_length: int = 10,
        vlm_density_window: int = 100,
    ):
        self.volatility_window = volatility_window
        self.acceleration_smooth = acceleration_smooth
        self.stability_tau = stability_tau
        self.transition_tau = transition_tau
        self.min_region_length = min_region_length
        self.vlm_density_window = vlm_density_window

    def analyze(self, df: pd.DataFrame) -> ObservatoryResult:
        """
        Run full analytics suite on a frame log DataFrame.

        Args:
            df: DataFrame from frame_log.csv

        Returns:
            ObservatoryResult with all computed metrics
        """
        df = df.copy().sort_values("frame_id").reset_index(drop=True)

        # ── Core derived signals ──────────────────────────────────────────────
        df = self._compute_acceleration(df)
        df = self._compute_volatility(df)
        df = self._compute_velocity(df)
        df = self._compute_scene_entropy(df)
        df = self._compute_semantic_progress(df)
        df = self._compute_vlm_density(df)
        df = self._classify_frame_state(df)

        # ── Region detection ──────────────────────────────────────────────────
        stable_regions, transition_regions, volatile_regions = \
            self._detect_regions(df)

        # ── Event detection ───────────────────────────────────────────────────
        semantic_events = self._detect_events(df)

        # ── Object persistence ────────────────────────────────────────────────
        object_persistence = self._compute_object_persistence(df)

        # ── Aggregate stats ───────────────────────────────────────────────────
        drift = df["drift_total"]
        duration_ms = float(df["timestamp_ms"].max()) if "timestamp_ms" in df.columns else 0.0
        fps = len(df) / (duration_ms / 1000) if duration_ms > 0 else 0.0

        vlm_rate = float(df["vlm_called"].mean()) if "vlm_called" in df.columns else 0.0

        return ObservatoryResult(
            total_frames=len(df),
            total_duration_ms=duration_ms,
            fps_estimate=fps,
            enriched_df=df,
            mean_drift=float(drift.mean()),
            median_drift=float(drift.median()),
            drift_std=float(drift.std()),
            max_drift=float(drift.max()),
            mean_acceleration=float(df["semantic_acceleration"].abs().mean()),
            mean_volatility=float(df["semantic_volatility"].mean()),
            stable_regions=stable_regions,
            transition_regions=transition_regions,
            volatile_regions=volatile_regions,
            semantic_events=semantic_events,
            object_persistence=object_persistence,
            vlm_call_rate=vlm_rate,
            vlm_trigger_density=df["vlm_density"],
            semantic_progress=df["semantic_progress"],
        )

    # ── Signal computation ────────────────────────────────────────────────────

    def _compute_acceleration(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        A_t = D_t - D_{t-1}
        Measures rate of change in semantic drift.
        High acceleration → abrupt semantic transition approaching.
        """
        df["semantic_acceleration"] = df["drift_total"].diff().fillna(0.0)
        # Smoothed version for visualization
        df["semantic_acceleration_smooth"] = (
            df["semantic_acceleration"]
            .rolling(self.acceleration_smooth, center=True, min_periods=1)
            .mean()
        )
        return df

    def _compute_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        V_t = Var(D_{t-k:t})  over rolling window of k frames.
        High volatility → unstable semantic environment, interaction-heavy scene.
        Low volatility  → stable semantic context, cache reuse appropriate.
        """
        df["semantic_volatility"] = (
            df["drift_total"]
            .rolling(self.volatility_window, center=True, min_periods=1)
            .std()
            .fillna(0.0)
        )
        return df

    def _compute_velocity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Semantic velocity: smoothed first derivative of drift.
        Indicates the sustained rate of semantic change.
        """
        df["semantic_velocity"] = (
            df["drift_total"]
            .rolling(10, center=True, min_periods=1)
            .mean()
            .diff()
            .fillna(0.0)
        )
        return df

    def _compute_scene_entropy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Shannon entropy of the object count distribution.
        High entropy → diverse, complex scene.
        Low entropy  → uniform, simple scene.

        Approximated from n_detections as proxy when class distribution
        is not available per frame.
        """
        if "n_detections" in df.columns:
            # Normalize detection counts to a probability proxy
            max_det = df["n_detections"].max()
            if max_det > 0:
                p = df["n_detections"] / (max_det + 1e-8)
                p = p.clip(1e-8, 1.0)
                df["scene_entropy"] = -(p * np.log2(p))
            else:
                df["scene_entropy"] = 0.0
        else:
            df["scene_entropy"] = 0.0
        return df

    def _compute_semantic_progress(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Semantic Progress Function (SPF):
        Cumulative sum of drift — measures total semantic distance traveled.

        A flat SPF → semantic plateau (stable scene, no meaningful change).
        A steep SPF → semantic burst (rapid scene transitions).
        """
        df["semantic_progress"] = df["drift_total"].cumsum()
        # Normalized to [0, 1]
        spf_max = df["semantic_progress"].max()
        if spf_max > 0:
            df["semantic_progress_norm"] = df["semantic_progress"] / spf_max
        else:
            df["semantic_progress_norm"] = 0.0
        return df

    def _compute_vlm_density(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rolling VLM invocation density.
        Shows temporal concentration of expensive reasoning calls.
        """
        if "vlm_called" in df.columns:
            vlm_numeric = df["vlm_called"].astype(float)
            df["vlm_density"] = (
                vlm_numeric
                .rolling(self.vlm_density_window, center=True, min_periods=1)
                .mean()
            )
        else:
            df["vlm_density"] = 0.0
        return df

    def _classify_frame_state(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify each frame into a semantic state based on drift and volatility.
        """
        conditions = [
            df["drift_total"] < self.stability_tau,
            df["drift_total"] >= self.transition_tau,
        ]
        choices = ["stable", "transition"]
        df["frame_semantic_state"] = np.select(conditions, choices, default="drifting")
        return df

    # ── Region detection ──────────────────────────────────────────────────────

    def _detect_regions(
        self, df: pd.DataFrame
    ) -> Tuple[List[SemanticRegion], List[SemanticRegion], List[SemanticRegion]]:
        """
        Detect contiguous semantic regions (stable / transition / volatile).
        Uses run-length encoding on the frame_semantic_state column.
        """
        stable, transition, volatile = [], [], []
        region_id = 0

        state_col = df["frame_semantic_state"].tolist()
        frames = df["frame_id"].tolist()
        drift = df["drift_total"].tolist()
        volatility = df["semantic_volatility"].tolist()
        vlm = df["vlm_called"].tolist() if "vlm_called" in df.columns else [False] * len(df)

        i = 0
        while i < len(state_col):
            current_state = state_col[i]
            j = i
            # Extend region as long as state is the same
            while j < len(state_col) and state_col[j] == current_state:
                j += 1

            duration = j - i
            if duration >= self.min_region_length:
                region_drift = drift[i:j]
                region_vol = volatility[i:j]
                region_vlm = vlm[i:j]

                region = SemanticRegion(
                    region_id=region_id,
                    region_type=current_state,
                    start_frame=frames[i],
                    end_frame=frames[j - 1],
                    duration_frames=duration,
                    mean_drift=float(np.mean(region_drift)),
                    max_drift=float(np.max(region_drift)),
                    mean_volatility=float(np.mean(region_vol)),
                    vlm_calls_in_region=int(sum(region_vlm)),
                    dominant_state=current_state,
                )

                if current_state == "stable":
                    stable.append(region)
                elif current_state == "transition":
                    transition.append(region)
                else:
                    volatile.append(region)

                region_id += 1
            i = j

        return stable, transition, volatile

    # ── Event detection ───────────────────────────────────────────────────────

    def _detect_events(self, df: pd.DataFrame) -> List[SemanticEvent]:
        """
        Detect point-in-time semantic events:
            - Drift spikes (sudden large D_t)
            - Acceleration bursts (sudden large A_t)
            - Semantic plateaus (long stable periods)
            - VLM trigger events
        """
        events = []
        event_id = 0

        drift = df["drift_total"].values
        accel = df["semantic_acceleration"].values
        frames = df["frame_id"].values
        timestamps = df["timestamp_ms"].values if "timestamp_ms" in df.columns else np.zeros(len(df))
        vlm = df["vlm_called"].values if "vlm_called" in df.columns else np.zeros(len(df))

        # Spike threshold: drift > mean + 2*std
        spike_thresh = drift.mean() + 2 * drift.std()
        # Burst threshold: |acceleration| > mean_accel + 2*std_accel
        accel_abs = np.abs(accel)
        burst_thresh = accel_abs.mean() + 2 * accel_abs.std()

        for i in range(len(df)):
            # Drift spike event
            if drift[i] > spike_thresh:
                events.append(SemanticEvent(
                    event_id=event_id,
                    event_type="spike",
                    frame_id=int(frames[i]),
                    timestamp_ms=float(timestamps[i]),
                    drift_at_event=float(drift[i]),
                    acceleration_at_event=float(accel[i]),
                    description=f"Drift spike: D_t={drift[i]:.3f} > threshold {spike_thresh:.3f}",
                ))
                event_id += 1

            # Acceleration burst event
            elif accel_abs[i] > burst_thresh and i > 0:
                events.append(SemanticEvent(
                    event_id=event_id,
                    event_type="burst",
                    frame_id=int(frames[i]),
                    timestamp_ms=float(timestamps[i]),
                    drift_at_event=float(drift[i]),
                    acceleration_at_event=float(accel[i]),
                    description=f"Acceleration burst: |A_t|={accel_abs[i]:.3f}",
                ))
                event_id += 1

            # VLM trigger event (Tier 3 only — important events)
            if "tier" in df.columns and df["tier"].iloc[i] == 3 and vlm[i]:
                events.append(SemanticEvent(
                    event_id=event_id,
                    event_type="vlm_trigger",
                    frame_id=int(frames[i]),
                    timestamp_ms=float(timestamps[i]),
                    drift_at_event=float(drift[i]),
                    acceleration_at_event=float(accel[i]),
                    description=f"Tier-3 VLM invocation at D_t={drift[i]:.3f}",
                ))
                event_id += 1

        return events

    # ── Object persistence ────────────────────────────────────────────────────

    def _compute_object_persistence(self, df: pd.DataFrame) -> List[ObjectPersistenceRecord]:
        """
        Infer object persistence from n_tracks column.
        Full track-level persistence requires track_ids per frame
        (available if pipeline outputs them — approximated here).
        """
        # This is an approximation from aggregate data.
        # Full implementation requires per-track data from ByteTrack.
        records = []
        if "n_tracks" not in df.columns:
            return records

        # Identify frames where track count changes (proxy for birth/death events)
        track_counts = df["n_tracks"].values
        frames = df["frame_id"].values

        for i in range(1, len(track_counts)):
            delta = int(track_counts[i]) - int(track_counts[i - 1])
            if abs(delta) > 0:
                records.append(ObjectPersistenceRecord(
                    track_id=-1,           # aggregate — no per-track ID
                    first_frame=int(frames[i - 1]),
                    last_frame=int(frames[i]),
                    lifespan_frames=1,
                    class_id=-1,
                ))

        return records
