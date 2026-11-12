"""Result clips: the frames where the person was found, as an MP4 a browser can play.

The final-year pipeline wrote ``output_annotated.mp4`` (the target boxed) and
``output.mp4`` (the target's crops back to back). The API cuts the same kind
of video from the stored original, keeping only the result's frames:

* ``boxed`` - the full frame (scaled to at most 960 px wide) with a green box,
  the track ID and the time in the video;
* ``crop``  - the person's crop, 128 x 256, like ``output.mp4``.

Frames come from the same 12 fps sampler the index was built with, so the clip
plays in real time. Encoding is H.264 (libx264, from imageio-ffmpeg's bundled
ffmpeg): OpenCV's pip wheels can only write MPEG-4 part 2, which browsers do
not play. Clips are cached by content: the same result is rendered once.
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from person_search.ingest.video import iter_sampled_frames
from person_search.search.results import FrameHit

log = logging.getLogger(__name__)

Style = Literal["boxed", "crop"]
STYLES: tuple[Style, ...] = ("boxed", "crop")
MAX_WIDTH = 960
CROP_SIZE = (128, 256)  # width, height: OSNet's aspect ratio
GREEN = (0, 255, 0)
NAME_RE = re.compile(r"^[0-9a-f]{16}_[tp][0-9A-Za-z-]{1,40}_(boxed|crop)\.mp4$")


class ClipError(Exception):
    """A clip cannot be made (video not stored, unknown track, nothing to show)."""


def clip_name(video_id: str, key: str, style: Style) -> str:
    safe = re.sub(r"[^0-9A-Za-z-]", "-", key)[:40]
    return f"{video_id}_{safe}_{style}.mp4"


def track_key(track_id: str) -> str:
    return f"t{track_id}"


def hits_key(hits: Sequence[FrameHit]) -> str:
    """A photo result's clip key: a digest of exactly which frames and tracks it shows."""
    data = ",".join(f"{h.frame_index}:{h.track_id}" for h in sorted(hits, key=_frame))
    return "p" + hashlib.sha1(data.encode()).hexdigest()[:12]


def _frame(h: FrameHit) -> int:
    return h.frame_index


def _even(n: int) -> int:
    return max(2, n - n % 2)


def _boxed(image: np.ndarray, hit: FrameHit, t: float) -> np.ndarray:
    frame = image.copy()
    height, width = frame.shape[:2]
    b = hit.bbox.clip(width, height)
    thick = max(2, width // 400)
    cv2.rectangle(frame, (b.left, b.top), (b.right, b.bottom), GREEN, thick)
    scale = max(0.6, width / 1100)  # ~0.85 after scaling to 960 px wide
    cv2.putText(
        frame,
        f"track {hit.track_id}",
        (b.left, max(20, b.top - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        GREEN,
        thick,
    )
    stamp = f"{int(t // 60)}:{t % 60:04.1f}"
    cv2.putText(
        frame, stamp, (10, int(30 * scale + 10)), cv2.FONT_HERSHEY_SIMPLEX, scale, GREEN, thick
    )
    if width > MAX_WIDTH:
        frame = cv2.resize(frame, (MAX_WIDTH, round(height * MAX_WIDTH / width)))
    h, w = frame.shape[:2]
    return frame[: _even(h), : _even(w)]


def _crop(image: np.ndarray, hit: FrameHit) -> np.ndarray | None:
    height, width = image.shape[:2]
    crop = hit.bbox.clip(width, height).crop(image)
    return None if crop.size == 0 else cv2.resize(crop, CROP_SIZE)


def render(
    video: Path, hits: Sequence[FrameHit], out: Path, style: Style, fps: float = 12.0
) -> int:
    """Write the frames of ``hits`` (one box per frame) to ``out``. Returns frames written."""
    import imageio_ffmpeg

    if style not in STYLES:
        raise ClipError(f"style must be one of {STYLES}")
    by_frame = {h.frame_index: h for h in hits}
    if not by_frame:
        raise ClipError("nothing to show: the result has no frames")
    last = max(by_frame)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.stem}.{uuid.uuid4().hex[:8]}.mp4")  # rename when complete
    writer = None
    size = (0, 0)
    written = 0
    try:
        for sample in iter_sampled_frames(video, fps):
            if sample.frame_index > last:
                break
            hit = by_frame.get(sample.frame_index)
            if hit is None:
                continue
            frame = (
                _boxed(sample.image, hit, sample.timestamp_s)
                if style == "boxed"
                else _crop(sample.image, hit)
            )
            if frame is None:
                continue
            if writer is None:
                size = (frame.shape[1], frame.shape[0])
                writer = imageio_ffmpeg.write_frames(
                    str(tmp),
                    size,
                    fps=fps,
                    codec="libx264",
                    quality=None,
                    macro_block_size=2,
                    ffmpeg_log_level="error",
                    output_params=["-preset", "veryfast", "-crf", "23", "-movflags", "+faststart"],
                )
                writer.send(None)
            if (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size)
            writer.send(np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).tobytes())
            written += 1
        if writer is None:
            raise ClipError("none of the result's frames could be read from the video")
        writer.close()
        writer = None
        tmp.replace(out)
    finally:
        if writer is not None:
            writer.close()
        tmp.unlink(missing_ok=True)
    log.info("clip %s: %d frames (%s)", out.name, written, style)
    return written


def prune(clip_dir: Path, keep: int) -> None:
    """Delete the oldest cached clips beyond ``keep``."""
    clips = sorted(clip_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in clips[keep:]:
        old.unlink(missing_ok=True)
