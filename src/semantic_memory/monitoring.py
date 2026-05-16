"""
src/semantic_memory/monitoring.py

Semantic Monitoring Layer.

Inspired by the QueST philosophy of monitoring-by-design:
    - monitor semantic state transitions continuously
    - detect silent semantic drift before it accumulates
    - classify representation-level changes into named states
    - maintain semantic identity across frames

This module is architecturally distinct from the drift estimator.
The drift estimator COMPUTES a number (D_t).
The monitoring layer INTERPRETS that number as a semantic state transition
and maintains a running classification of the scene's semantic health.

Semantic States:
    STABLE:      D_t < τ_low  — scene semantically unchanged
                 → reuse cached VLM output, Tier 1 compute only

    DRIFTING:    τ_low ≤ D_t < τ_high — gradual semantic shift
                 → monitor closely, lightweight Tier 2 reasoning

    TRANSITION:  D_t ≥ τ_high — active semantic state change
                 → invoke full VLM, Tier 3 compute, update memory anchor

    NOVEL:       D_t ≥ τ_high AND no prior semantic anchor exists
                 OR objects not seen before enter the scene
                 → first-seen event, always invoke VLM, store as new anchor

Key concept — Silent Semantic Drift:
    A scene can drift semantically without large pixel changes.
    Example: a person slowly picks up an object over 30 frames.
    Per-frame D_t may stay below τ_high, but cumulative drift is large.
    The monitoring layer detects this via drift_accumulator.

Key concept — Semantic Anchor:
    After every TRANSITION or NOVEL event, the current embedding is stored
    as a semantic anchor. Future drift is measured from this anchor, not
    from the decayed memory. This aligns with QueST's persistent identity idea.

Usage in pipeline (sits BETWEEN drift estimator and scheduler):
    drift_components = drift_estimator.compute(...)
    monitor_result   = semantic_monitor.update(drift_components, embedding, detections)
    decision         = scheduler.decide(drift=monitor_result.effective_drift, ...)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set

import numpy as np

from src.detection.yolo_detector import Detection
from src.semantic_memory.drift import DriftComponents


class SemanticState(Enum):
    """
    Named semantic state classifications.

    These map directly to thesis figures — the "semantic state transition
    visualization" figure uses exactly these four states.
    """
    STABLE     = "stable"      # D_t < τ_low
    DRIFTING   = "drifting"    # τ_low ≤ D_t < τ_high (watch closely)
    TRANSITION = "transition"  # D_t ≥ τ_high (active change)
    NOVEL      = "novel"       # first-time semantic event (no prior anchor)


@dataclass
class MonitoringResult:
    """
    Output of the Semantic Monitoring Layer for a single frame.

    effective_drift: drift score to pass to the scheduler.
        For STABLE/DRIFTING: same as D_t.
        For silent drift detected: inflated to force a VLM call.

    semantic_state: named classification for logging and visualization.

    silent_drift_detected: True when cumulative drift exceeded the silent
        drift threshold even though per-frame D_t was below τ_high.

    anchor_updated: True when a new semantic anchor was stored this frame.

    frames_in_state: how many consecutive frames in the current state.
    """
    frame_id: int
    semantic_state: SemanticState
    effective_drift: float         # drift passed to scheduler (may be inflated)
    raw_drift: float               # original D_t from DriftEstimator
    drift_accumulator: float       # cumulative drift since last anchor
    silent_drift_detected: bool
    anchor_updated: bool
    frames_in_state: int
    n_novel_classes: int           # count of object classes not in anchor


@dataclass
class SemanticAnchor:
    """
    Semantic anchor — snapshot at last TRANSITION or NOVEL event.

    Anchors serve as the persistent semantic identity reference point.
    Drift is measured from the anchor, not from the frame-by-frame memory.
    """
    frame_id: int
    embedding: np.ndarray          # CLIP embedding at anchor time
    object_classes: Set[int]       # object class IDs present at anchor
    semantic_state: SemanticState  # state that triggered this anchor


class SemanticMonitor:
    """
    Semantic Monitoring Layer.

    Continuously monitors representation-level semantic state transitions
    and detects silent semantic drift that accumulates below the per-frame
    threshold. Inspired by QueST's monitoring-by-design philosophy.

    Args:
        tau_low:               drift below this → STABLE
        tau_high:              drift above this → TRANSITION
        silent_drift_tau:      cumulative drift threshold for silent drift detection
                               (recommended: 3x τ_high)
        silent_drift_window:   frames over which to accumulate drift
        embedding_dim:         CLIP ViT-B/32 = 512
    """

    def __init__(
        self,
        tau_low: float = 0.15,
        tau_high: float = 0.40,
        silent_drift_tau: float = 1.2,
        silent_drift_window: int = 90,   # ~3 sec at 30fps
        embedding_dim: int = 512,
    ):
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.silent_drift_tau = silent_drift_tau
        self.silent_drift_window = silent_drift_window
        self.embedding_dim = embedding_dim

        # Current semantic state
        self._state: SemanticState = SemanticState.NOVEL  # start as NOVEL (no anchor)
        self._frames_in_state: int = 0
        self._frame_id: int = 0

        # Anchor — set at every TRANSITION or NOVEL event
        self._anchor: Optional[SemanticAnchor] = None

        # Silent drift accumulator
        self._drift_accumulator: float = 0.0
        self._frames_since_anchor: int = 0

        # Statistics
        self._state_history: List[SemanticState] = []
        self._transition_count: int = 0
        self._novel_count: int = 0
        self._silent_drift_count: int = 0

    def update(
        self,
        drift_components: DriftComponents,
        current_embedding: np.ndarray,
        detections: List[Detection],
    ) -> MonitoringResult:
        """
        Update semantic monitoring state for the current frame.

        Args:
            drift_components:  output from SemanticDriftEstimator
            current_embedding: L2-normalized CLIP embedding (512,)
            detections:        YOLO detections this frame

        Returns:
            MonitoringResult with semantic state, effective drift, and flags
        """
        self._frame_id += 1
        self._frames_in_state += 1
        self._frames_since_anchor += 1

        raw_drift = drift_components.d_total
        effective_drift = raw_drift
        anchor_updated = False
        silent_drift_detected = False

        current_classes = {d.class_id for d in detections}

        # ── Step 1: Classify raw semantic state ────────────────────────────
        new_state = self._classify_state(raw_drift, current_classes)

        # ── Step 2: Detect silent semantic drift ───────────────────────────
        # Accumulate drift from anchor. If sum exceeds silent_drift_tau,
        # force a TRANSITION even if per-frame drift is below τ_high.
        self._drift_accumulator += raw_drift

        if (
            self._drift_accumulator >= self.silent_drift_tau
            and new_state in (SemanticState.STABLE, SemanticState.DRIFTING)
            and self._frames_since_anchor >= 10   # prevent false positives at startup
        ):
            silent_drift_detected = True
            self._silent_drift_count += 1
            # Inflate effective_drift to force Tier-3 VLM call
            effective_drift = self.tau_high + 0.05
            new_state = SemanticState.TRANSITION

        # ── Step 3: State transition bookkeeping ───────────────────────────
        if new_state != self._state:
            self._frames_in_state = 1
            self._state = new_state

        # ── Step 4: Update anchor on TRANSITION or NOVEL ───────────────────
        if new_state in (SemanticState.TRANSITION, SemanticState.NOVEL):
            self._anchor = SemanticAnchor(
                frame_id=self._frame_id,
                embedding=current_embedding.copy(),
                object_classes=current_classes.copy(),
                semantic_state=new_state,
            )
            self._drift_accumulator = 0.0
            self._frames_since_anchor = 0
            anchor_updated = True

            if new_state == SemanticState.TRANSITION:
                self._transition_count += 1
            else:
                self._novel_count += 1

        # ── Step 5: Count novel object classes ─────────────────────────────
        n_novel_classes = 0
        if self._anchor is not None:
            n_novel_classes = len(current_classes - self._anchor.object_classes)

        self._state_history.append(self._state)

        return MonitoringResult(
            frame_id=self._frame_id,
            semantic_state=self._state,
            effective_drift=float(np.clip(effective_drift, 0.0, 1.0)),
            raw_drift=raw_drift,
            drift_accumulator=self._drift_accumulator,
            silent_drift_detected=silent_drift_detected,
            anchor_updated=anchor_updated,
            frames_in_state=self._frames_in_state,
            n_novel_classes=n_novel_classes,
        )

    def _classify_state(
        self,
        drift: float,
        current_classes: Set[int],
    ) -> SemanticState:
        """
        Classify current frame into a named semantic state.

        NOVEL takes priority over TRANSITION when:
            - No anchor exists yet (cold start)
            - New object classes appear that weren't in the last anchor
        """
        # Cold start — no anchor yet
        if self._anchor is None:
            return SemanticState.NOVEL

        # New object class not seen at last anchor → NOVEL
        new_classes = current_classes - self._anchor.object_classes
        if new_classes and drift >= self.tau_low:
            return SemanticState.NOVEL

        # Standard threshold classification
        if drift < self.tau_low:
            return SemanticState.STABLE
        elif drift < self.tau_high:
            return SemanticState.DRIFTING
        else:
            return SemanticState.TRANSITION

    def anchor_drift(self, current_embedding: np.ndarray) -> float:
        """
        Compute cosine distance from the current semantic anchor embedding.
        Used as an alternative/complementary drift signal to memory-based drift.

        Returns 1.0 if no anchor exists (forces VLM call on cold start).
        """
        if self._anchor is None:
            return 1.0
        dot = float(np.dot(current_embedding, self._anchor.embedding))
        return float(1.0 - np.clip(dot, -1.0, 1.0))

    @property
    def current_state(self) -> SemanticState:
        return self._state

    @property
    def stats(self) -> dict:
        total = len(self._state_history)
        if total == 0:
            return {}
        counts = {s.value: self._state_history.count(s) for s in SemanticState}
        return {
            "total_frames_monitored": total,
            "state_counts": counts,
            "state_fractions": {k: v / total for k, v in counts.items()},
            "transition_events": self._transition_count,
            "novel_events": self._novel_count,
            "silent_drift_events": self._silent_drift_count,
        }

    def reset(self):
        self._state = SemanticState.NOVEL
        self._frames_in_state = 0
        self._frame_id = 0
        self._anchor = None
        self._drift_accumulator = 0.0
        self._frames_since_anchor = 0
        self._state_history = []
        self._transition_count = 0
        self._novel_count = 0
        self._silent_drift_count = 0