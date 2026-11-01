"""Video decoding with the legacy time-based sampler.

The legacy loop reads every frame and keeps frame ``i`` (1-based) when
``i / fps >= next_t``, then advances ``next_t`` by ``1 / target_fps``. For the
12 fps test videos this keeps every frame; for 30 fps footage it keeps ~12 of
every 30. :func:`sample_schedule` reproduces that arithmetic without decoding,
so labels in sampled-frame space can be mapped back to raw frame numbers.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from person_search.errors import VideoReadError
from person_search.types import VideoInfo


@dataclass(frozen=True, slots=True)
class SampledFrame:
    frame_index: int  # 1-based raw index
    sample_index: int  # 0-based index in the sampled stream
    timestamp_s: float
    image: np.ndarray


def probe(path: str | Path, fallback_fps: float = 30.0) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoReadError(f"cannot open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        return VideoInfo(
            path=str(path),
            fps=fps if fps > 0 else fallback_fps,
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


class TimeSampler:
    """Stateful form of the legacy sampler: feed raw indices in order."""

    def __init__(self, fps: float, target_fps: float) -> None:
        if fps <= 0 or target_fps <= 0:
            raise ValueError("fps and target_fps must be positive")
        self.fps = fps
        self.step = 1.0 / target_fps
        self._next_t = 0.0

    def keep(self, frame_index: int) -> bool:
        if frame_index / self.fps < self._next_t:
            return False
        self._next_t += self.step
        return True


def sample_schedule(frame_count: int, fps: float, target_fps: float) -> list[int]:
    """1-based raw indices the sampler keeps, in order (position = sample index)."""
    sampler = TimeSampler(fps, target_fps)
    return [i for i in range(1, frame_count + 1) if sampler.keep(i)]


def iter_sampled_frames(
    path: str | Path,
    target_fps: float,
    fallback_fps: float = 30.0,
    max_samples: int | None = None,
) -> Iterator[SampledFrame]:
    """Yield the sampled frames of a video (full resolution, BGR)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise VideoReadError(f"cannot open video: {path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps > 0 else fallback_fps
        sampler = TimeSampler(fps, target_fps)
        frame_index = 0
        sample_index = 0
        while max_samples is None or sample_index < max_samples:
            ok, image = cap.read()
            if not ok:
                break
            frame_index += 1
            if not sampler.keep(frame_index):
                continue
            yield SampledFrame(frame_index, sample_index, frame_index / fps, image)
            sample_index += 1
    finally:
        cap.release()
