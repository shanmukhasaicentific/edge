"""
src/semantic_memory/drift.py

Semantic Drift Estimator.

Computes representation-level semantic drift D_t — the core signal
that drives adaptive compute allocation in this system.

Core formulation:
    D_t = α·D_embed + β·D_objects + γ·D_track

Where:
    D_embed   = representation-level cosine drift between CLIP embeddings
    D_objects = object-level novelty score (class-set and count change)
    D_track   = tracking identity change score (object entry/exit)

    α, β, γ ≥ 0  and  α + β + γ = 1  (weights sum to 1)

D_t ∈ [0, 1] (clipped).

Design rationale:
    - D_embed is the primary representation-level signal. Captures holistic
      semantic scene change in CLIP embedding space — robust to lighting,
      viewpoint changes, and visual noise that pixel metrics would overreact to.
    - D_objects captures object-level semantic novelty: a new class entering
      the scene is a semantic event even if D_embed is moderate.
    - D_track captures movement dynamics: persons entering or leaving the
      scene are semantically significant for robotic perception tasks.

Note on silent semantic drift:
    Per-frame D_t may stay below τ_high while the scene drifts semantically
    over many frames (e.g. a robot arm slowly moving, a person gradually
    picking up an object). The SemanticMonitor (monitoring.py) accumulates
    D_t over time to detect this silent drift — this module provides the
    per-frame signal that the monitor accumulates.

Note on anchor-based drift:
    D_t here is computed against the temporal memory M_t (decayed average).
    The SemanticMonitor also computes drift from semantic anchors — fixed
    reference points set at TRANSITION and NOVEL events. Use
    SemanticMonitor.anchor_drift(embedding) for anchor-based measurement.
"""

from dataclasses import dataclass
from typing import Counter, Dict, List, Optional, Set

import numpy as np

from src.detection.yolo_detector import Detection
from src.tracking.bytetrack_wrapper import TrackingResult


@dataclass
class DriftComponents:
    d_embed: float       # Embedding cosine drift [0, 1]
    d_objects: float     # Object novelty score [0, 1]
    d_track: float       # Track identity change [0, 1]
    d_total: float       # Weighted sum D_t [0, 1]
    alpha: float
    beta: float
    gamma: float


class SemanticDriftEstimator:
    """
    Estimates semantic drift D_t from three complementary signals.

    Args:
        alpha: weight for embedding drift (default 0.5)
        beta:  weight for object drift   (default 0.3)
        gamma: weight for track drift    (default 0.2)

    Note: alpha + beta + gamma should = 1.0
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
    ):
        assert abs(alpha + beta + gamma - 1.0) < 1e-6, \
            f"Weights must sum to 1.0, got {alpha + beta + gamma}"

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Previous state for delta computation
        self._prev_class_counter: Counter = Counter()
        self._prev_class_set: Set[int] = set()

    def compute(
        self,
        current_embedding: np.ndarray,
        memory_embedding: np.ndarray,
        detections: List[Detection],
        tracking_result: Optional[TrackingResult],
    ) -> DriftComponents:
        """
        Compute full semantic drift D_t.

        Args:
            current_embedding: L2-norm CLIP embedding (512,)
            memory_embedding:  L2-norm memory state M_t (512,)
            detections:        YOLO detections this frame
            tracking_result:   ByteTrack result (can be None)

        Returns:
            DriftComponents with all signals and final D_t
        """
        d_embed = self._embedding_drift(current_embedding, memory_embedding)
        d_objects = self._object_drift(detections)
        d_track = self._track_drift(tracking_result)

        d_total = (
            self.alpha * d_embed
            + self.beta * d_objects
            + self.gamma * d_track
        )
        d_total = float(np.clip(d_total, 0.0, 1.0))

        return DriftComponents(
            d_embed=d_embed,
            d_objects=d_objects,
            d_track=d_track,
            d_total=d_total,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
        )

    def _embedding_drift(
        self,
        current: np.ndarray,
        memory: np.ndarray,
    ) -> float:
        """
        D_embed = 1 - cosine_similarity(E_t, M_t)
        Since embeddings are unit vectors: = 1 - dot(E_t, M_t)
        """
        dot = float(np.dot(current, memory))
        dot = np.clip(dot, -1.0, 1.0)
        return float(1.0 - dot)

    def _object_drift(self, detections: List[Detection]) -> float:
        """
        D_objects captures two signals:
            1. Class set change: new class appeared or class disappeared
            2. Count change: significant change in number of objects

        Normalized to [0, 1].
        """
        if not detections:
            current_counter = Counter()
            current_class_set = set()
        else:
            current_counter = Counter(d.class_id for d in detections)
            current_class_set = set(current_counter.keys())

        if not self._prev_class_set and not current_class_set:
            d_objects = 0.0
        else:
            # Jaccard distance on class sets
            union = self._prev_class_set | current_class_set
            intersection = self._prev_class_set & current_class_set
            if len(union) == 0:
                class_set_change = 0.0
            else:
                class_set_change = 1.0 - len(intersection) / len(union)

            # Normalized count change
            prev_count = sum(self._prev_class_counter.values())
            curr_count = sum(current_counter.values())
            max_count = max(prev_count, curr_count, 1)
            count_change = abs(curr_count - prev_count) / max_count

            d_objects = 0.7 * class_set_change + 0.3 * count_change

        # Update state
        self._prev_class_counter = current_counter
        self._prev_class_set = current_class_set

        return float(np.clip(d_objects, 0.0, 1.0))

    def _track_drift(self, tracking_result: Optional[TrackingResult]) -> float:
        """
        D_track = (|new_ids| + |lost_ids|) / max(|active_tracks|, 1)
        """
        if tracking_result is None:
            return 0.0

        n_active = max(len(tracking_result.tracks), 1)
        n_changes = len(tracking_result.new_ids) + len(tracking_result.lost_ids)
        return float(np.clip(n_changes / n_active, 0.0, 1.0))

    def reset(self):
        self._prev_class_counter = Counter()
        self._prev_class_set = set()