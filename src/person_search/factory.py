"""Composition root: builds concrete components from :class:`Settings`.

The rest of the code depends on the protocols in ``interfaces.py``; only this
module knows which implementation backs each one.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import torch

from person_search.config import Settings
from person_search.errors import ConfigurationError
from person_search.ingest.detector import YoloDetector
from person_search.ingest.pipeline import IngestPipeline
from person_search.ingest.tracker import DeepSortTracker
from person_search.interfaces import ImageEncoder
from person_search.models.irra_encoder import IRRAEncoder
from person_search.models.osnet_encoder import OSNetEncoder
from person_search.types import IRRA, OSNET

log = logging.getLogger(__name__)


def configure_torch(settings: Settings) -> None:
    if settings.torch_threads:
        torch.set_num_threads(settings.torch_threads)


def build_detector(settings: Settings) -> YoloDetector:
    d = settings.detector
    return YoloDetector(
        d.weights,
        confidence=d.confidence,
        person_class_id=d.person_class_id,
        resize_width=d.resize_width,
        device=settings.device,
        torch_threads=settings.torch_threads,
    )


def build_tracker(settings: Settings) -> DeepSortTracker:
    return DeepSortTracker(settings.tracker)


def build_osnet(settings: Settings) -> OSNetEncoder:
    s = settings.osnet
    return OSNetEncoder(
        s.weights,
        device=settings.device,
        input_size=s.input_size,
        batch_size=s.batch_size,
        version=s.version,
    )


def build_irra(settings: Settings) -> IRRAEncoder:
    s = settings.irra
    return IRRAEncoder(
        s.weights,
        device=settings.device,
        batch_size=s.batch_size,
        precision=s.precision,
        version=s.version,
    )


@dataclass
class Components:
    """Loaded models, built once per process and shared across requests."""

    settings: Settings
    detector: YoloDetector
    tracker: DeepSortTracker
    encoders: dict[str, ImageEncoder] = field(default_factory=dict)

    @property
    def osnet(self) -> OSNetEncoder:
        enc = self.encoders.get(OSNET)
        if enc is None:
            raise ConfigurationError("OSNet encoder not loaded")
        return enc  # type: ignore[return-value]

    @property
    def irra(self) -> IRRAEncoder:
        enc = self.encoders.get(IRRA)
        if enc is None:
            raise ConfigurationError("IRRA encoder not loaded (settings.irra.enabled?)")
        return enc  # type: ignore[return-value]

    def ingest_pipeline(self) -> IngestPipeline:
        return IngestPipeline(
            self.detector,
            self.tracker,
            self.encoders,
            sampling=self.settings.sampling,
            min_box_size=self.settings.tracker.min_box_size,
        )


def build_components(settings: Settings, encoders: Iterable[str] | None = None) -> Components:
    """Load the detector, tracker and the requested encoders.

    ``encoders`` defaults to OSNet plus IRRA when ``settings.irra.enabled``.
    """
    configure_torch(settings)
    wanted = (
        list(encoders)
        if encoders is not None
        else [OSNET] + ([IRRA] if settings.irra.enabled else [])
    )
    builders: dict[str, Callable[[Settings], ImageEncoder]] = {
        OSNET: build_osnet,
        IRRA: build_irra,
    }
    unknown = set(wanted) - set(builders)
    if unknown:
        raise ConfigurationError(f"unknown encoder(s): {sorted(unknown)}")
    loaded: dict[str, ImageEncoder] = {name: builders[name](settings) for name in wanted}
    return Components(settings, build_detector(settings), build_tracker(settings), loaded)
