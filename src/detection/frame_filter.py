"""
src/detection/frame_filter.py

Lightweight frame filter — runs before YOLO to drop redundant frames.
Operates on CPU. Uses motion magnitude and blur detection.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class FilterResult:
    keep: bool
    motion_score: float
    blur_score: float
    reason: str


class FrameFilter:
    """
    Tier-1 CPU filter: drops frames with insufficient motion or excessive blur.
    Runs BEFORE YOLO to save GPU calls entirely.

    Args:
        motion_threshold: min mean absolute difference to keep a frame
        blur_threshold: Laplacian variance below this = blurry (drop)
        skip_first_n: always keep the first N frames (warmup)
    """

    def __init__(
        self,
        motion_threshold: float = 2.0,
        blur_threshold: float = 50.0,
        skip_first_n: int = 5,
    ):
        self.motion_threshold = motion_threshold
        self.blur_threshold = blur_threshold
        self.skip_first_n = skip_first_n
        self._prev_gray: Optional[np.ndarray] = None
        self._frame_count = 0

    def filter(self, frame: np.ndarray) -> FilterResult:
        """
        Args:
            frame: BGR uint8 numpy array (H, W, 3)
        Returns:
            FilterResult with keep decision and diagnostic scores
        """
        self._frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Always keep warmup frames
        if self._frame_count <= self.skip_first_n:
            self._prev_gray = gray
            return FilterResult(keep=True, motion_score=0.0, blur_score=0.0, reason="warmup")

        # Blur detection
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < self.blur_threshold:
            self._prev_gray = gray
            return FilterResult(keep=False, motion_score=0.0, blur_score=blur_score, reason="blurry")

        # Motion detection
        motion_score = 0.0
        if self._prev_gray is not None:
            diff = cv2.absdiff(gray, self._prev_gray)
            motion_score = float(diff.mean())

        self._prev_gray = gray

        if motion_score < self.motion_threshold:
            return FilterResult(keep=False, motion_score=motion_score, blur_score=blur_score, reason="low_motion")

        return FilterResult(keep=True, motion_score=motion_score, blur_score=blur_score, reason="passed")

    def reset(self):
        self._prev_gray = None
        self._frame_count = 0
