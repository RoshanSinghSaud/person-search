"""Search result types."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from person_search.core.scoring import ThresholdDecision
from person_search.types import BBox


@dataclass(frozen=True, slots=True)
class FrameHit:
    frame_index: int
    sample_index: int
    track_id: str
    bbox: BBox


@dataclass(frozen=True, slots=True)
class Segment:
    """A run of consecutive sampled frames, inclusive on both ends."""

    start_sample: int
    end_sample: int
    start_frame: int
    end_frame: int

    @property
    def n_samples(self) -> int:
        return self.end_sample - self.start_sample + 1


@dataclass(frozen=True)
class PhotoSearchResult:
    decision: ThresholdDecision
    hits: tuple[FrameHit, ...]

    @property
    def matched(self) -> bool:
        return self.decision.matched

    @property
    def selected_samples(self) -> list[int]:
        return [h.sample_index for h in self.hits]

    @property
    def selected_frames(self) -> list[int]:
        return [h.frame_index for h in self.hits]

    def segments(self, max_gap: int = 1) -> list[Segment]:
        return to_segments(self.hits, max_gap)


def to_segments(hits: Sequence[FrameHit], max_gap: int = 1) -> list[Segment]:
    """Merge selected samples into segments; gaps up to ``max_gap`` samples are bridged."""
    ordered = sorted(hits, key=lambda h: h.sample_index)
    segments: list[Segment] = []
    for h in ordered:
        if segments and h.sample_index - segments[-1].end_sample <= max_gap:
            last = segments[-1]
            segments[-1] = Segment(
                last.start_sample, h.sample_index, last.start_frame, h.frame_index
            )
        else:
            segments.append(Segment(h.sample_index, h.sample_index, h.frame_index, h.frame_index))
    return segments
