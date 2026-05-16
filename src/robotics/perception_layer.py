"""
src/robotics/perception_layer.py

Robotic Task Perception Layer.

This module is the APPLICATION FRAMING layer — it sits at the end of
the pipeline and translates semantic monitoring outputs into structured
percepts and semantic events that a downstream robotic system could consume.

IMPORTANT SCOPE NOTE:
This module does NOT implement robot control, manipulation, or navigation.
It implements semantic scene understanding relevant to robotic perception:
    - Object grounding (which objects are present and where)
    - Scene stability classification (STABLE / DYNAMIC / CRITICAL)
    - Semantic event emission (structured events for robotic middleware)
    - Task-relevant semantic tagging (for embodied AI framing)

This framing justifies the thesis title:
    "Semantic Monitoring for Adaptive Compute Allocation
     in Edge Robotic Vision-Language Systems"

Semantic Event format (required by Prompt 3):
    {
        "object":        primary object involved in the semantic event,
        "semantic_drift": D_t score at event time,
        "task_relevance": "high" / "medium" / "low",
        "vlm_invoked":   True/False,
        "semantic_state": "stable" / "drifting" / "transition" / "novel"
    }

Future integration targets:
    - ROS2 topic publisher (semantic events as ROS messages)
    - Object-goal navigation (semantic state → navigation waypoint)
    - Task-oriented grasping (object location + semantic label → grasp target)
    - Mobile robot re-planning (CRITICAL state → halt and re-plan)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.detection.yolo_detector import Detection
from src.tracking.bytetrack_wrapper import Track


class SceneStability(Enum):
    """
    Semantic stability classification for robotic decision-making.

    STABLE:   Scene semantically unchanged — robot can execute plan safely.
    DYNAMIC:  Scene actively changing — robot should pause and re-perceive.
    CRITICAL: High semantic drift — robot must halt and re-plan.
    """
    STABLE   = "stable"
    DYNAMIC  = "dynamic"
    CRITICAL = "critical"


@dataclass
class SemanticEvent:
    """
    Structured semantic event emitted per frame.

    This is the primary output format consumed by robotic middleware.
    Format specified in Prompt 3, Module 11.

    Example:
        SemanticEvent(
            object="mug",
            semantic_drift=0.71,
            task_relevance="high",
            vlm_invoked=True,
            semantic_state="transition",
            frame_id=423,
        )
    """
    object: str                  # primary object in the scene (most prominent)
    semantic_drift: float        # D_t score (raw, not effective)
    task_relevance: str          # "high" / "medium" / "low"
    vlm_invoked: bool
    semantic_state: str          # from SemanticMonitor state name
    frame_id: int
    scene_description: Optional[str] = None   # VLM caption if available

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object": self.object,
            "semantic_drift": round(self.semantic_drift, 4),
            "task_relevance": self.task_relevance,
            "vlm_invoked": self.vlm_invoked,
            "semantic_state": self.semantic_state,
            "frame_id": self.frame_id,
            "scene_description": self.scene_description,
        }


@dataclass
class ObjectPercept:
    """Structured semantic percept for a detected + tracked object."""
    track_id: int
    class_name: str
    bbox: List[float]                # [x1, y1, x2, y2]
    confidence: float
    semantic_label: str              # from VLM if available, else class_name
    centroid: Tuple[float, float]
    is_new: bool                     # appeared this frame


@dataclass
class RoboticPercept:
    """
    Complete structured output of the Robotic Task Perception Layer.

    Contains both the full percept (for detailed analysis) and
    the concise SemanticEvent (for robotic middleware consumption).
    """
    frame_id: int
    scene_stability: SceneStability
    drift_score: float
    objects: List[ObjectPercept]
    scene_description: Optional[str]     # VLM caption if available
    cache_hit: bool
    compute_tier: int
    action_recommendation: str           # high-level robotic action hint
    semantic_event: SemanticEvent        # structured event for middleware


class RoboticPerceptionLayer:
    """
    Translates semantic monitoring outputs into structured robotic percepts
    and semantic events for downstream middleware consumption.

    Stability thresholds:
        drift < τ_stable  → STABLE
        drift < τ_dynamic → DYNAMIC
        drift ≥ τ_dynamic → CRITICAL

    Task relevance classification:
        HIGH:   new objects appear, TRANSITION or NOVEL semantic state
        MEDIUM: DRIFTING state, scene actively changing
        LOW:    STABLE state, no meaningful semantic event

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
        semantic_state: str = "stable",    # from SemanticMonitor.current_state.value
        vlm_invoked: bool = False,
    ) -> RoboticPercept:
        """
        Build a RoboticPercept and SemanticEvent from pipeline outputs.

        Args:
            frame_id:        current frame number
            drift:           D_t raw semantic drift score
            detections:      YOLO detections
            tracks:          ByteTrack active tracks
            new_track_ids:   set of newly appeared track IDs
            vlm_caption:     Moondream output (None if no VLM call this frame)
            cache_hit:       True if output came from cache
            compute_tier:    1, 2, or 3
            semantic_state:  SemanticMonitor state name ("stable"/"drifting"/
                             "transition"/"novel")
            vlm_invoked:     True if VLM was called this frame

        Returns:
            RoboticPercept with embedded SemanticEvent
        """
        stability = self._classify_stability(drift)
        objects = self._build_object_percepts(tracks, detections, new_track_ids)
        action = self._recommend_action(stability, objects)
        task_relevance = self._classify_task_relevance(semantic_state, objects)
        primary_object = self._primary_object(objects, detections)

        semantic_event = SemanticEvent(
            object=primary_object,
            semantic_drift=drift,
            task_relevance=task_relevance,
            vlm_invoked=vlm_invoked,
            semantic_state=semantic_state,
            frame_id=frame_id,
            scene_description=vlm_caption,
        )

        return RoboticPercept(
            frame_id=frame_id,
            scene_stability=stability,
            drift_score=drift,
            objects=objects,
            scene_description=vlm_caption,
            cache_hit=cache_hit,
            compute_tier=compute_tier,
            action_recommendation=action,
            semantic_event=semantic_event,
        )

    def _classify_stability(self, drift: float) -> SceneStability:
        if drift < self.tau_stable:
            return SceneStability.STABLE
        elif drift < self.tau_dynamic:
            return SceneStability.DYNAMIC
        else:
            return SceneStability.CRITICAL

    def _classify_task_relevance(self, semantic_state: str, objects: List[ObjectPercept]) -> str:
        """
        HIGH:   scene is in TRANSITION or NOVEL state, or new objects appeared
        MEDIUM: scene is DRIFTING
        LOW:    scene is STABLE
        """
        if semantic_state in ("transition", "novel"):
            return "high"
        if semantic_state == "drifting":
            return "medium"
        if any(o.is_new for o in objects):
            return "high"
        return "low"

    def _primary_object(
        self,
        objects: List[ObjectPercept],
        detections: List[Detection],
    ) -> str:
        """Return the class name of the highest-confidence detection."""
        if detections:
            best = max(detections, key=lambda d: d.confidence)
            return best.class_name
        if objects:
            return objects[0].class_name
        return "none"

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
                semantic_label=class_name,
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
        Semantic framing only — not actual robot control.
        """
        if stability == SceneStability.STABLE:
            return "proceed_with_plan"
        elif stability == SceneStability.DYNAMIC:
            n_new = sum(1 for o in objects if o.is_new)
            if n_new > 0:
                return f"pause_new_objects_detected:{n_new}"
            return "caution_scene_changing"
        else:
            return "halt_replan_high_semantic_drift"