"""DeepSORT tracker (deep-sort-realtime).

Extracted from the legacy ``DeepSort(...)`` construction and the
``update_tracks`` loop. Parameters come from :class:`TrackerSettings`; the
defaults are repo 2's (max_age 30, n_init 3, mobilenet appearance embedder,
library-default max_cosine_distance 0.2).

License: deep-sort-realtime is GPL-3.0 (see THIRD_PARTY_NOTICES.md).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from person_search.config import TrackerSettings
from person_search.types import Detection, TrackState


class DeepSortTracker:
    def __init__(self, settings: TrackerSettings | None = None) -> None:
        self.settings = settings or TrackerSettings()
        self._tracker = self._build()

    def _build(self) -> Any:  # deep_sort_realtime.deepsort_tracker.DeepSort
        from deep_sort_realtime.deepsort_tracker import DeepSort

        s = self.settings
        return DeepSort(
            max_age=s.max_age,
            n_init=s.n_init,
            max_iou_distance=s.max_iou_distance,
            max_cosine_distance=s.max_cosine_distance,
            nn_budget=s.nn_budget,
            embedder=s.embedder,
            embedder_gpu=s.embedder_gpu,
        )

    def reset(self) -> None:
        self._tracker = self._build()

    def update(self, detections: Sequence[Detection], frame: np.ndarray) -> list[TrackState]:
        raw = [([d.x, d.y, d.w, d.h], d.confidence, d.label) for d in detections]
        tracks = self._tracker.update_tracks(raw, frame=frame)
        out: list[TrackState] = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            if not self.settings.include_unmatched_tracks and track.time_since_update > 0:
                continue
            left, top, right, bottom = (float(v) for v in track.to_ltrb())
            out.append(
                TrackState(
                    track_id=str(track.track_id),
                    ltrb=(left, top, right, bottom),
                    time_since_update=int(track.time_since_update),
                )
            )
        return out
