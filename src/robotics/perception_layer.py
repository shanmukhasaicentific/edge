"""
src/robotics/perception_layer.py

Robotic Task Perception Layer.

This module is the APPLICATION FRAMING layer — it sits at the end of
the pipeline and translates VLM semantic outputs into structured
percepts that a downstream robotic system could consume.

IMPORTANT SCOPE NOTE:
This module does NOT implement robot control, manipulation, or navigation.
It implements semantic scene understanding relevant to robotic perception:
    - Object grounding (which objects are present and where)
    - Scene stability classification (safe to act / scene changing)
    - Task-relevant semantic tagging (for embodied AI framing)

This framing justifies the thesis title:
    "Adaptive Semantic Compute Allocation for Edge Robotic Vision-Language Systems"

Future integration targets:
    - ROS2 topic publisher (semantic percepts as ROS messages)
    - Object-goal navigation (semantic state → navigation waypoint)
    - Task-oriented grasping (object location + semantic label → grasp target)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.detection.yolo_detector import Detection
from src.tracking.bytetrack_wrapper import Track


class SceneStability(Enum):
    """
    Semantic stability classification for robotic decision-making.
    
    STABLE:   Scene unchanged — robot can execute planned action safely.
    DYNAMIC:  Scene changing — robot should pause and re-perceive.
    CRITICAL: High semantic drift — robot must re-plan.
    """
    STABLE = "stable"
    DYNAMIC = "dynamic"
    CRITICAL = "critical"


@dataclass
class ObjectPercept:
    """Structured semantic percept for a detected + tracked object."""
    track_id: int
    class_name: str
    bbox: List[float]           # [x1, y1, x2, y2]
    confidence: float
    semantic_label: str         # from VLM if available, else class_name
    centroid: Tuple[float, float]
    is_new: bool                # appeared this frame


@dataclass
class RoboticPercept:
    """
    Complete structured semantic percept for one frame.
    Output of the Robotic Task Perception Layer.

    This is what a downstream robotic system would consume.
    """
    frame_id: int
    scene_stability: SceneStability
    drift_score: float
    objects: List[ObjectPercept]
    scene_description: Optional[str]        # VLM caption if available
    cache_hit: bool
    compute_tier: int
    action_recommendation: str              # high-level suggestion for robot


class RoboticPerceptionLayer:
    """
    Translates pipeline outputs into robotic percepts.

    Stability thresholds:
        drift < τ_stable  → STABLE
        drift < τ_dynamic → DYNAMIC
        drift ≥ τ_dynamic → CRITICAL

    Args:
        tau_stable:  max drift for STABLE classification
        tau_dynamic: max drift for DYNAMIC classification
    """

    def __init__(
        self,
        tau_stable: float = 0.10,
        tau_dynamic: float = 0.35,
    ):
        self.tau_stable = tau_stable
        self.tau_dynamic = tau_dynamic

    def process(
        self,
        frame_id: int,
        drift: float,
        detections: List[Detection],
        tracks: List[Track],
        new_track_ids: set,
        vlm_caption: Optional[str],
        cache_hit: bool,
        compute_tier: int,
    ) -> RoboticPercept:
        """
        Build a RoboticPercept from pipeline outputs.

        Args:
            frame_id:      current frame number
            drift:         D_t semantic drift score
            detections:    YOLO detections
            tracks:        ByteTrack active tracks
            new_track_ids: set of newly appeared track IDs
            vlm_caption:   Moondream output (None if no VLM call this frame)
            cache_hit:     True if output came from cache
            compute_tier:  1, 2, or 3

        Returns:
            RoboticPercept structured output
        """
        stability = self._classify_stability(drift)
        objects = self._build_object_percepts(tracks, detections, new_track_ids)
        action = self._recommend_action(stability, objects)

        return RoboticPercept(
            frame_id=frame_id,
            scene_stability=stability,
            drift_score=drift,
            objects=objects,
            scene_description=vlm_caption,
            cache_hit=cache_hit,
            compute_tier=compute_tier,
            action_recommendation=action,
        )

    def _classify_stability(self, drift: float) -> SceneStability:
        if drift < self.tau_stable:
            return SceneStability.STABLE
        elif drift < self.tau_dynamic:
            return SceneStability.DYNAMIC
        else:
            return SceneStability.CRITICAL

    def _build_object_percepts(
        self,
        tracks: List[Track],
        detections: List[Detection],
        new_track_ids: set,
    ) -> List[ObjectPercept]:
        """Build ObjectPercept list from tracked objects."""
        percepts = []
        for t in tracks:
            x1, y1, x2, y2 = t.bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Find matching detection for class name
            class_name = str(t.class_id)
            for d in detections:
                if d.class_id == t.class_id:
                    class_name = d.class_name
                    break

            percepts.append(ObjectPercept(
                track_id=t.track_id,
                class_name=class_name,
                bbox=t.bbox,
                confidence=t.confidence,
                semantic_label=class_name,   # overridden by VLM if available
                centroid=(cx, cy),
                is_new=t.track_id in new_track_ids,
            ))
        return percepts

    def _recommend_action(
        self,
        stability: SceneStability,
        objects: List[ObjectPercept],
    ) -> str:
        """
        High-level action recommendation for a robotic system.
        This is semantic framing — not actual robot control.
        """
        if stability == SceneStability.STABLE:
            return "proceed_with_plan"
        elif stability == SceneStability.DYNAMIC:
            n_new = sum(1 for o in objects if o.is_new)
            if n_new > 0:
                return f"pause_new_objects_detected:{n_new}"
            return "caution_scene_changing"
        else:  # CRITICAL
            return "halt_replan_high_semantic_drift"
