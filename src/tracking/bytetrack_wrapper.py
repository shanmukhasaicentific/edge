"""
src/tracking/bytetrack_wrapper.py

ByteTrack multi-object tracker wrapper.
Compatible with BoxMOT v10-v21+ (handles API changes).

Track identity changes contribute to the γ·D_track component of semantic drift.

Migration notes:
- BoxMOT v10-v18: from boxmot import ByteTrack (class-based)
- BoxMOT v19+: from boxmot.trackers.byte_tracker import BYTETracker (class) or boxmot.track (function)
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
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        self._prev_ids: Set[int] = set()

        # Try to load tracker (handles both old and new BoxMOT APIs)
        self.tracker = self._init_tracker()

    def _init_tracker(self):
        """Initialize tracker, handling both old (v10-v18) and new (v19+) BoxMOT APIs."""
        try:
            # Try old API: v10-v18
            from boxmot import ByteTrack
            print("✓ Using BoxMOT v10-v18 API (from boxmot import ByteTrack)")
            return ByteTrack(
                track_thresh=self.track_thresh,
                track_buffer=self.track_buffer,
                match_thresh=self.match_thresh,
                frame_rate=self.frame_rate,
            )
        except ImportError:
            pass

        try:
            # Try new API: v19+
            from boxmot.trackers.byte_tracker import BYTETracker
            print("✓ Using BoxMOT v19+ API (from boxmot.trackers.byte_tracker import BYTETracker)")
            return BYTETracker(
                track_thresh=self.track_thresh,
                track_buffer=self.track_buffer,
                match_thresh=self.match_thresh,
                frame_rate=self.frame_rate,
            )
        except ImportError:
            pass

        # If neither works, raise helpful error
        print("\n" + "="*70)
        print("ERROR: Cannot import ByteTrack from BoxMOT")
        print("="*70)
        print("\nTried:")
        print("  1. from boxmot import ByteTrack (v10-v18)")
        print("  2. from boxmot.trackers.byte_tracker import BYTETracker (v19+)")
        print("\nSolutions:")
        print("  Option A: Pin to working version")
        print("    pip install boxmot==10.11.62")
        print("\n  Option B: Check BoxMOT version")
        print("    pip show boxmot")
        print("\n  Option C: Update BoxMOT to latest")
        print("    pip install --upgrade boxmot")
        print("="*70)
        raise ImportError("BoxMOT ByteTrack not found in any supported version")

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
