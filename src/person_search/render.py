"""Output clips.

Extracted from the legacy ``create_annotated_video`` (repo 2) and
``create_video_from_memory_crops`` (repo 1). The annotated clip contains every
sampled frame, with a green "Target" box on the selected ones, so its frame
``i`` is sample index ``i`` (the space ``evaluation.json`` is labelled in).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from person_search.ingest.video import iter_sampled_frames
from person_search.search.results import FrameHit

log = logging.getLogger(__name__)

CODECS = ("avc1", "mp4v", "XVID")
GREEN = (0, 255, 0)


def _even(n: int) -> int:
    return n if n % 2 == 0 else n - 1


def open_writer(path: str | Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    for codec in CODECS:
        fourcc = cv2.VideoWriter_fourcc(*codec)  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(str(path), fourcc, fps, size)
        if writer.isOpened():
            log.debug("writing %s with codec %s", path, codec)
            return writer
        writer.release()
    raise RuntimeError(f"no working video codec for {path} (tried {', '.join(CODECS)})")


def draw_target(frame: np.ndarray, hit: FrameHit, label: str = "Target") -> None:
    b = hit.bbox
    cv2.rectangle(frame, (b.left, b.top), (b.right, b.bottom), GREEN, 2)
    cv2.putText(frame, label, (b.left, b.top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, GREEN, 2)


def write_annotated_video(
    video: str | Path,
    hits: Sequence[FrameHit],
    out_path: str | Path,
    target_fps: float = 12.0,
) -> int:
    """Every sampled frame, with the selected person boxed. Returns frames written."""
    by_frame = {h.frame_index: h for h in hits}
    writer = None
    written = 0
    try:
        for sample in iter_sampled_frames(video, target_fps):
            frame = sample.image
            if writer is None:
                size = (_even(frame.shape[1]), _even(frame.shape[0]))
                writer = open_writer(out_path, target_fps, size)
            hit = by_frame.get(sample.frame_index)
            if hit is not None:
                draw_target(frame, hit)
            if (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size)
            writer.write(frame)
            written += 1
    finally:
        if writer is not None:
            writer.release()
    log.info("annotated video %s: %d frames, %d boxed", out_path, written, len(by_frame))
    return written


def write_crop_clip(
    video: str | Path,
    hits: Sequence[FrameHit],
    out_path: str | Path,
    target_fps: float = 12.0,
) -> int:
    """The selected crops back to back, resized to the first crop's size."""
    by_frame = {h.frame_index: h for h in hits}
    writer = None
    size = (0, 0)
    written = 0
    try:
        for sample in iter_sampled_frames(video, target_fps):
            hit = by_frame.get(sample.frame_index)
            if hit is None:
                continue
            crop = hit.bbox.crop(sample.image)
            if crop.size == 0:
                continue
            if writer is None:
                size = (max(2, _even(crop.shape[1])), max(2, _even(crop.shape[0])))
                writer = open_writer(out_path, target_fps, size)
            writer.write(cv2.resize(crop, size))
            written += 1
    finally:
        if writer is not None:
            writer.release()
    return written
