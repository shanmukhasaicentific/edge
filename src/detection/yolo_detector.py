"""
src/detection/yolo_detector.py

YOLOv8 nano detector wrapper.
Returns normalized detection results as typed dataclasses.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch


@dataclass
class Detection:
    bbox: List[float]       # [x1, y1, x2, y2] — absolute pixel coords
    confidence: float
    class_id: int
    class_name: str


@dataclass
class DetectionResult:
    detections: List[Detection]
    inference_ms: float
    frame_id: int


class YOLODetector:
    """
    Wraps YOLOv8 nano for object detection.
    Uses ultralytics YOLO API.

    Args:
        model_path: path to .pt file or 'yolov8n.pt' for auto-download
        conf_threshold: minimum confidence to keep detection
        device: 'cuda' or 'cpu'
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.3,
        device: str = "cuda",
    ):
        # Lazy import to avoid loading at module level
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = device
        self._class_names: Optional[dict] = None

    def detect(self, frame: np.ndarray, frame_id: int = 0) -> DetectionResult:
        """
        Run YOLOv8 inference on a single frame.

        Args:
            frame: BGR uint8 numpy array
            frame_id: for logging

        Returns:
            DetectionResult with typed detections and timing
        """
        t0 = time.perf_counter()

        results = self.model(
            frame,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        for r in results:
            if self._class_names is None:
                self._class_names = r.names

            boxes = r.boxes
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu())
                cls_id = int(boxes.cls[i].cpu())
                cls_name = self._class_names.get(cls_id, str(cls_id))

                detections.append(Detection(
                    bbox=bbox,
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                ))

        inference_ms = (time.perf_counter() - t0) * 1000

        return DetectionResult(
            detections=detections,
            inference_ms=inference_ms,
            frame_id=frame_id,
        )

    def to_bytetrack_format(self, result: DetectionResult) -> np.ndarray:
        """
        Convert DetectionResult to ByteTrack input format:
        [[x1, y1, x2, y2, score, class_id], ...]
        """
        if not result.detections:
            return np.empty((0, 6), dtype=np.float32)

        rows = []
        for d in result.detections:
            rows.append([*d.bbox, d.confidence, d.class_id])
        return np.array(rows, dtype=np.float32)
