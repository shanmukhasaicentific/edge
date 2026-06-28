"""
src/tracking/bytetrack_wrapper.py

ByteTrack multi-object tracker wrapper.
Uses boxmot library for ByteTrack implementation.

Track identity changes contribute to the γ·D_track component of semantic drift.
"""

import time
from dataclasses import dataclass
from typing import List, Optional, Set

import numpy as np


@dataclass
class Track:
    track_id: int
    bbox: List[float]   # [x1, y1, x2, y2]
    class_id: int
    confidence: float


@dataclass
class TrackingResult:
    tracks: List[Track]
    new_ids: Set[int]        # IDs that appeared this frame
    lost_ids: Set[int]       # IDs that disappeared this frame
    inference_ms: float
    frame_id: int


class ByteTrackWrapper:
    """
    Wraps ByteTrack for multi-object tracking.

    Track identity change score (D_track) is computed as:
        D_track = (|new_ids| + |lost_ids|) / max(|active_ids|, 1)

    This captures object entry/exit events which are semantically significant.

    Args:
        track_thresh: tracking confidence threshold
        track_buffer: frames to keep lost tracks alive
        match_thresh: IOU match threshold for association
        frame_rate: video FPS (for buffer calculation)
    """

    def __init__(
        self,
        track_thresh: float = 0.3,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 30,
    ):
        """Initialize ByteTrack wrapper (compatible with boxmot v10-v21+)."""
        try:
            # Try modern boxmot API (v10-v18)
            from boxmot import ByteTrack
            self.tracker = ByteTrack(
                track_thresh=track_thresh,
                track_buffer=track_buffer,
                match_thresh=match_thresh,
                frame_rate=frame_rate,
            )
        except ImportError:
            # Fallback for future boxmot versions
            # If you get here, boxmot.ByteTrack is not available
            # See: https://github.com/mikel-brostrom/yolo_tracking
            print("ERROR: boxmot.ByteTrack not found.")
            print("Recommended: pip install boxmot==10.11.62")
            print("Or update this wrapper for your boxmot version.")
            raise

        self._prev_ids: Set[int] = set()

    def update(self, detections: np.ndarray, frame: np.ndarray, frame_id: int = 0) -> TrackingResult:
        """
        Update tracker with new detections.

        Args:
            detections: numpy array of shape (N, 6) — [x1,y1,x2,y2,score,class_id]
            frame: BGR frame (needed by some trackers for ReID)
            frame_id: for logging

        Returns:
            TrackingResult with active tracks and identity change info
        """
        t0 = time.perf_counter()

        if len(detections) == 0:
            detections = np.empty((0, 6), dtype=np.float32)

        # ByteTrack returns (N, 7): [x1,y1,x2,y2,track_id,score,class_id]
        raw_tracks = self.tracker.update(detections, frame)

        tracks = []
        current_ids: Set[int] = set()

        for t in raw_tracks:
            x1, y1, x2, y2, track_id, score, class_id = t[:7]
            track_id = int(track_id)
            current_ids.add(track_id)
            tracks.append(Track(
                track_id=track_id,
                bbox=[float(x1), float(y1), float(x2), float(y2)],
                class_id=int(class_id),
                confidence=float(score),
            ))

        new_ids = current_ids - self._prev_ids
        lost_ids = self._prev_ids - current_ids
        self._prev_ids = current_ids

        inference_ms = (time.perf_counter() - t0) * 1000

        return TrackingResult(
            tracks=tracks,
            new_ids=new_ids,
            lost_ids=lost_ids,
            inference_ms=inference_ms,
            frame_id=frame_id,
        )

    def compute_track_drift(self, result: TrackingResult) -> float:
        """
        D_track = (|new_ids| + |lost_ids|) / max(|active_tracks|, 1)
        Normalized to [0, 1].
        """
        n_active = max(len(result.tracks), 1)
        n_changes = len(result.new_ids) + len(result.lost_ids)
        return min(n_changes / n_active, 1.0)

    def reset(self):
        self.tracker.reset()
        self._prev_ids = set()
