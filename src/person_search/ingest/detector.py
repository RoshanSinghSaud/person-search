"""YOLO11n person detector (Ultralytics).

Extracted from the detection block of the legacy ``run_reid`` loop (identical in
both repos): downscale frames wider than ``resize_width``, run YOLO on class 0,
scale boxes back to the original frame, keep ``conf >= confidence``.

License: Ultralytics YOLO is AGPL-3.0 (see THIRD_PARTY_NOTICES.md).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from person_search.errors import ModelLoadError
from person_search.types import Detection

log = logging.getLogger(__name__)


class YoloDetector:
    def __init__(
        self,
        weights: str | Path,
        *,
        confidence: float = 0.4,
        person_class_id: int = 0,
        resize_width: int = 640,
        device: str = "cpu",
    ) -> None:
        weights = Path(weights)
        if not weights.is_file():
            # Ultralytics would otherwise try to download a file of that name.
            raise ModelLoadError(f"YOLO weights not found: {weights}")
        try:
            from ultralytics import YOLO

            self.model = YOLO(str(weights))
        except Exception as exc:
            raise ModelLoadError(f"cannot load YOLO weights {weights}: {exc}") from exc
        # Same mechanism as legacy: the threshold is also YOLO's internal conf.
        self.model.overrides["conf"] = confidence  # type: ignore[assignment]
        self.confidence = confidence
        self.classes = [person_class_id]
        self.resize_width = resize_width
        self.device = device
        log.info("YOLO loaded from %s (conf=%.2f)", weights, confidence)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        w = frame.shape[1]
        scale = 1.0
        small = frame
        if w > self.resize_width:
            scale = self.resize_width / w
            small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        results: Any = self.model(small, verbose=False, classes=self.classes, device=self.device)
        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            for box, c in zip(xyxy, conf, strict=True):
                x1, y1, x2, y2 = box
                # numpy-scalar division (float32 / float -> float64), as the legacy loop did
                if scale != 1.0:
                    x1, y1, x2, y2 = x1 / scale, y1 / scale, x2 / scale, y2 / scale
                if float(c) < self.confidence:
                    continue
                detections.append(
                    Detection(float(x1), float(y1), float(x2 - x1), float(y2 - y1), float(c))
                )
        return detections
