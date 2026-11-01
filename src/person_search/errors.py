"""Exception hierarchy.

Every failure the legacy code swallowed (ResNet50 fallback, base-CLIP fallback,
``print`` + ``return None``) is an exception here. Callers decide what to do;
nothing silently degrades.
"""

from __future__ import annotations

from collections.abc import Iterable


class PersonSearchError(Exception):
    """Base class for all errors raised by this package."""


class ConfigurationError(PersonSearchError):
    """Settings are missing or inconsistent."""


class ModelLoadError(PersonSearchError):
    """A model or its weights could not be loaded."""


class CheckpointMismatchError(ModelLoadError):
    """A checkpoint does not match the architecture it is loaded into."""

    def __init__(
        self,
        model_name: str,
        path: str,
        missing: Iterable[str] = (),
        unexpected: Iterable[str] = (),
        shape_mismatch: Iterable[str] = (),
    ) -> None:
        self.model_name = model_name
        self.path = path
        self.missing = sorted(missing)
        self.unexpected = sorted(unexpected)
        self.shape_mismatch = sorted(shape_mismatch)
        parts = [f"{model_name}: checkpoint {path} does not match the model."]
        for label, keys in (
            ("missing", self.missing),
            ("unexpected", self.unexpected),
            ("shape mismatch", self.shape_mismatch),
        ):
            if keys:
                preview = ", ".join(keys[:8]) + (" ..." if len(keys) > 8 else "")
                parts.append(f"{len(keys)} {label} key(s): {preview}")
        super().__init__(" ".join(parts))


class VideoReadError(PersonSearchError):
    """A video could not be opened or decoded."""


class QueryError(PersonSearchError):
    """A search query is invalid (missing photo/text, unreadable image, bad alpha)."""
