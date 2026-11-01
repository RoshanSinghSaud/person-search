"""Ingest: sampling, detection, tracking, and the embedding pass."""

from person_search.ingest.pipeline import IngestPipeline, IngestResult
from person_search.ingest.video import SampledFrame, iter_sampled_frames, probe, sample_schedule

__all__ = [
    "IngestPipeline",
    "IngestResult",
    "SampledFrame",
    "iter_sampled_frames",
    "probe",
    "sample_schedule",
]
