"""Typed settings. No path or threshold lives anywhere else in the code.

Precedence (highest first):

1. keyword arguments to :func:`load_settings`
2. environment variables, prefix ``PS_`` with ``__`` for nesting
   (``PS_SCORING__BASE_THRESHOLD=0.7``)
3. ``.env`` file in the working directory
4. a YAML profile (``configs/*.yaml``) given by ``config_file`` or ``PS_CONFIG_FILE``
5. the defaults below

The defaults are repo 2's committed values (``pole/reid_main.py``), the final
code of the published system; the golden test pins them. See AUDIT.md (F4, F5)
and ``docs/decisions/0002-config-profile.md``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

DEFAULT_MODELS_DIR = Path("models")


class DetectorSettings(BaseModel):
    weights: Path = DEFAULT_MODELS_DIR / "yolo11n_person.pt"
    confidence: float = Field(0.4, ge=0.0, le=1.0)
    person_class_id: int = 0
    resize_width: int = Field(
        640, gt=0, description="frames wider than this are downscaled for detection"
    )


class TrackerSettings(BaseModel):
    max_age: int = Field(30, gt=0)
    n_init: int = Field(3, gt=0)
    max_iou_distance: float = 0.7
    # deep-sort-realtime's default. The report and paper say 0.3 but the committed
    # code never set it; 0.2 is what the committed code ran with.
    max_cosine_distance: float = 0.2
    nn_budget: int | None = None
    embedder: Literal["mobilenet", "torchreid", "clip_RN50", "clip_ViT-B/16"] = "mobilenet"
    embedder_gpu: bool = False
    min_box_size: int = Field(
        10, ge=1, description="crops narrower or shorter than this are dropped"
    )
    # Legacy behaviour records confirmed tracks that were *not* matched this frame
    # (Kalman-predicted boxes). Kept True for parity; see AUDIT.md F11.
    include_unmatched_tracks: bool = True


class SamplingSettings(BaseModel):
    target_fps: float = Field(12.0, gt=0)
    fallback_fps: float = Field(30.0, gt=0, description="used when the container reports fps <= 0")
    embed_every: int = Field(3, ge=1, description="extract embeddings on every N-th sampled frame")


class OSNetSettings(BaseModel):
    weights: Path = DEFAULT_MODELS_DIR / "osnet_campus_v1.pth.tar"
    version: str = "osnet-v1"
    input_size: tuple[int, int] = (256, 128)
    batch_size: int = Field(32, gt=0)


class IRRASettings(BaseModel):
    weights: Path = DEFAULT_MODELS_DIR / "irra_v1.pth"
    version: str = "irra-v1"
    hf_repo_id: str = "prasish17/clip-text-query"
    hf_filename: str = "best_model_text.pth"
    batch_size: int = Field(16, gt=0)
    precision: Literal["fp32", "fp16"] = "fp32"
    enabled: bool = Field(True, description="embed IRRA image vectors during ingest")
    image_onnx: Path | None = Field(
        None, description="ONNX image tower (FP32 or INT8) used instead of PyTorch's"
    )


class ScoringSettings(BaseModel):
    """The paper's track-level robust scoring and dynamic threshold."""

    top_fraction: float = Field(0.5, gt=0.0, le=1.0)
    base_threshold: float = Field(
        0.65, description="tau_base: absolute floor on the best track score"
    )
    relative_margin: float = Field(
        0.1, ge=0.0, description="delta: accept tracks within this of the best"
    )
    frame_threshold_ratio: float = Field(
        0.9, gt=0.0, le=1.0, description="per-frame floor = ratio * effective threshold"
    )


class FusionSettings(BaseModel):
    alpha: float = Field(
        0.5, ge=0.0, le=1.0, description="weight of the OSNet score in hybrid queries"
    )
    normalization: Literal["minmax", "zscore"] = "minmax"


class ApiSettings(BaseModel):
    """The HTTP service (``person_search.api``); see docs/decisions/0012-web-api.md."""

    index_path: Path = Field(
        Path("data/index"), description="Qdrant local-mode folder; created empty if missing"
    )
    upload_dir: Path = Field(
        Path("data/uploads"), description="uploaded videos wait here until ingested"
    )
    max_photo_mb: float = Field(10.0, gt=0)
    max_video_mb: float = Field(500.0, gt=0)
    max_pending_jobs: int = Field(
        4, ge=1, description="queued + running ingest jobs; more and /ingest answers 429"
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PS_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
    )

    device: str = "cpu"
    torch_threads: int | None = Field(None, ge=1)
    output_dir: Path = Path("outputs")

    detector: DetectorSettings = DetectorSettings()
    tracker: TrackerSettings = TrackerSettings()
    sampling: SamplingSettings = SamplingSettings()
    osnet: OSNetSettings = OSNetSettings()
    irra: IRRASettings = IRRASettings()
    scoring: ScoringSettings = ScoringSettings()
    fusion: FusionSettings = FusionSettings()
    api: ApiSettings = ApiSettings()

    @field_validator("device")
    @classmethod
    def _check_device(cls, v: str) -> str:
        if v != "cpu" and not v.startswith(("cuda", "mps")):
            raise ValueError(f"unsupported device {v!r}; use cpu, cuda[:N] or mps")
        return v

    @model_validator(mode="after")
    def _check_fp16(self) -> Settings:
        if self.irra.precision == "fp16" and self.device == "cpu":
            raise ValueError("irra.precision=fp16 needs a GPU device")
        return self

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)


class _YamlSource(PydanticBaseSettingsSource):
    def __init__(self, settings_cls: type[BaseSettings], path: Path | None) -> None:
        super().__init__(settings_cls)
        self._data: dict[str, Any] = {}
        if path is not None:
            raw = yaml.safe_load(Path(path).read_text()) or {}
            if not isinstance(raw, dict):
                raise ValueError(f"{path}: top level must be a mapping")
            self._data = raw

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        return self._data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        return dict(self._data)


def load_settings(config_file: str | Path | None = None, **overrides: Any) -> Settings:
    """Build :class:`Settings` from a YAML profile, the environment and overrides."""
    path = config_file or os.environ.get("PS_CONFIG_FILE")
    resolved = Path(path) if path else None
    if resolved is not None and not resolved.is_file():
        raise FileNotFoundError(f"config file not found: {resolved}")

    class _ProfiledSettings(Settings):
        @classmethod
        def settings_customise_sources(
            cls,
            settings_cls: type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
        ) -> tuple[PydanticBaseSettingsSource, ...]:
            return (
                init_settings,
                env_settings,
                dotenv_settings,
                _YamlSource(settings_cls, resolved),
                file_secret_settings,
            )

    return _ProfiledSettings(**overrides)
