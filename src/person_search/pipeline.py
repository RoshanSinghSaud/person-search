"""End-to-end entry points built from the modules.

``run(video, photo)`` is the paper's photo query done the legacy way (ingest the
video, then score), assembled from the new modules. The golden test pins its
output to the frames repo 2 selected, so later refactors (batching, ONNX,
indexing) can be checked against it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from person_search.config import Settings, load_settings
from person_search.core.scoring import TrackScore
from person_search.errors import QueryError
from person_search.factory import Components, build_components
from person_search.ingest.pipeline import IngestResult
from person_search.search.photo import photo_search
from person_search.search.results import PhotoSearchResult
from person_search.search.text import text_search
from person_search.types import IRRA, OSNET


@dataclass
class PhotoRun:
    ingest: IngestResult
    result: PhotoSearchResult


@dataclass
class TextRun:
    ingest: IngestResult
    ranked: list[TrackScore]


def read_query_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise QueryError(f"query image not found or unreadable: {path}")
    return img


def run(
    video: str | Path,
    photo: str | Path,
    settings: Settings | None = None,
    components: Components | None = None,
) -> PhotoRun:
    """Photo query on one video, reprocessing it (the legacy behaviour)."""
    settings = settings or (components.settings if components else load_settings())
    components = components or build_components(settings, encoders=[OSNET])
    query = components.osnet.encode_images([read_query_image(photo)])[0]
    ingest = components.ingest_pipeline().run(video)
    return PhotoRun(ingest, photo_search(ingest.observations, query, settings.scoring, OSNET))


def run_text(
    video: str | Path,
    description: str,
    top_k: int = 5,
    settings: Settings | None = None,
    components: Components | None = None,
) -> TextRun:
    """Text query on one video, reprocessing it."""
    settings = settings or (components.settings if components else load_settings())
    components = components or build_components(settings, encoders=[IRRA])
    query = components.irra.encode_texts([description])[0]
    ingest = components.ingest_pipeline().run(video)
    return TextRun(ingest, text_search(ingest.observations, query, top_k, settings.scoring, IRRA))
