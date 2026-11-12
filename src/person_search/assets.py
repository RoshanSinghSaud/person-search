"""Weights and indexes the service needs, downloaded when they are missing.

``configs/assets.yaml`` says where each file comes from (a Hugging Face Hub
repo) and what it must be (size, SHA-256). :func:`plan_assets` works out, from
the settings, which files the configured service reads and which of them are
absent; :func:`fetch_assets` downloads those, checks them, and only then moves
them into place, so a failed or interrupted start never leaves a half-written
model or index behind. ``person-search serve`` runs this before loading the
models (decision 0013).

Kept as they are:

* a weight file that exists with the manifest's size (checking 1 GB of SHA-256
  on every start would add seconds; the hash is checked when it is downloaded);
* an index folder that already holds the collection for the loaded versions,
  whatever its size: ``/ingest`` grows it, and re-downloading would erase that.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from person_search.config import Settings
from person_search.errors import ConfigurationError
from person_search.types import IRRA, OSNET

log = logging.getLogger(__name__)

STAGING = ".download"


class AssetError(ConfigurationError):
    """An asset is missing and cannot be fetched, or a download is not what the manifest says."""


@dataclass(frozen=True, slots=True)
class Source:
    repo: str
    filename: str  # path inside the repo
    revision: str

    def __str__(self) -> str:
        return f"hf://{self.repo}/{self.filename}@{self.revision[:12]}"


@dataclass(frozen=True, slots=True)
class Asset:
    dest: Path
    source: Source
    size: int
    sha256: str
    staging: Path  # same filesystem as dest, so the final move is a rename


@dataclass
class Plan:
    present: list[Path] = field(default_factory=list)
    fetch: list[Asset] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "present": [str(p) for p in self.present],
            "fetch": [
                {"dest": str(a.dest), "source": str(a.source), "mb": round(a.size / 1e6, 1)}
                for a in self.fetch
            ],
            "problems": self.problems,
            "notes": self.notes,
        }


Downloader = Callable[[Source, Path], Path]


# ------------------------------------------------------------------- manifest
def load_manifest(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text()) or {}
    for key in ("weights", "indexes"):
        data.setdefault(key, {})
    for name, entry in data["weights"].items():
        _check_entry(f"weights.{name}", entry)
    for name, files in data["indexes"].items():
        for rel, entry in files.items():
            _check_entry(f"indexes.{name}.{rel}", entry)
    return data


def _check_entry(where: str, entry: dict[str, Any]) -> None:
    sha = str(entry.get("sha256", ""))
    if len(sha) != 64 or not isinstance(entry.get("size"), int):
        raise AssetError(f"assets manifest: {where} needs an int size and a 64-hex sha256")


def collection_for(settings: Settings) -> str:
    from person_search.index.qdrant import collection_name

    return collection_name({OSNET: settings.osnet.version, IRRA: settings.irra.version})


def needed_weights(settings: Settings) -> list[Path]:
    paths = [settings.detector.weights, settings.osnet.weights, settings.irra.weights]
    if settings.irra.image_onnx is not None:
        paths.append(settings.irra.image_onnx)
    return paths


# ----------------------------------------------------------------------- plan
def plan_assets(settings: Settings, manifest: dict[str, Any]) -> Plan:
    plan = Plan()
    repo = settings.assets.repo or manifest.get("repo")
    revision = settings.assets.revision or manifest.get("revision") or "main"

    def source(entry: dict[str, Any], default_filename: str, what: str) -> Source | None:
        src_repo = entry.get("repo") or repo
        if not src_repo:
            plan.problems.append(
                f"{what} is missing and comes from the assets repo, which is not set: "
                "set PS_ASSETS__REPO (see `make publish-assets`) or put the file in place"
            )
            return None
        return Source(
            src_repo,
            entry.get("filename") or default_filename,
            entry.get("revision") or (revision if not entry.get("repo") else "main"),
        )

    for path in needed_weights(settings):
        entry = manifest["weights"].get(path.name)
        if path.is_file() and (entry is None or path.stat().st_size == entry["size"]):
            plan.present.append(path)
            continue
        if entry is None:
            plan.problems.append(f"{path} is missing and {path.name} is not in the manifest")
            continue
        if path.is_file():
            plan.notes.append(f"{path} has the wrong size; downloading it again")
        src = source(entry, f"weights/{path.name}", str(path))
        if src is not None:
            plan.fetch.append(
                Asset(path, src, entry["size"], entry["sha256"], path.parent / STAGING)
            )

    name = collection_for(settings)
    root = settings.api.index_path
    collections = root / "collection"
    if (collections / name).is_dir():
        plan.present.append(collections / name)
    elif name not in manifest["indexes"]:
        plan.notes.append(f"no published index for {name}: the service starts with an empty one")
    elif collections.is_dir() and any(collections.iterdir()):
        others = sorted(p.name for p in collections.iterdir())
        plan.problems.append(
            f"{root} holds {others}, not {name}: use an empty PS_API__INDEX_PATH for these models"
        )
    else:
        files = manifest["indexes"][name]
        # meta.json first: the collection folder appears last, so its presence
        # means the whole index was moved into place.
        for rel in sorted(files, key=lambda r: r.startswith("collection/")):
            entry = files[rel]
            src = source(entry, f"indexes/{name}/{rel}", f"index {name}")
            if src is None:
                break
            plan.fetch.append(
                Asset(root / rel, src, entry["size"], entry["sha256"], root / STAGING)
            )
    return plan


# ---------------------------------------------------------------------- fetch
def sha256_of(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def hf_download(source: Source, out_dir: Path) -> Path:
    """Download one file from the Hugging Face Hub into ``out_dir`` (HF_TOKEN for private repos)."""
    from huggingface_hub import hf_hub_download

    try:
        return Path(
            hf_hub_download(
                source.repo, source.filename, revision=source.revision, local_dir=out_dir
            )
        )
    except Exception as exc:
        raise AssetError(
            f"cannot download {source}: {type(exc).__name__}: {exc}. "
            "A private repo needs HF_TOKEN; check PS_ASSETS__REPO and the revision."
        ) from exc


def fetch_assets(plan: Plan, downloader: Downloader = hf_download) -> list[Path]:
    """Download, verify, then move every planned file into place. Returns the new paths."""
    if plan.problems:
        raise AssetError("cannot start: " + "; ".join(plan.problems))
    for note in plan.notes:
        log.warning(note)
    staged: list[tuple[Asset, Path]] = []
    try:
        for asset in plan.fetch:
            asset.staging.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            log.info("downloading %s (%.1f MB) -> %s", asset.source, asset.size / 1e6, asset.dest)
            path = downloader(asset.source, asset.staging)
            size = path.stat().st_size
            if size != asset.size:
                raise AssetError(f"{asset.source}: {size} bytes, expected {asset.size}")
            digest = sha256_of(path)
            if digest != asset.sha256:
                raise AssetError(f"{asset.source}: sha256 {digest}, expected {asset.sha256}")
            seconds = time.perf_counter() - t0
            log.info(
                "verified %s in %.1fs (%.1f MB/s)", asset.dest.name, seconds, size / 1e6 / seconds
            )
            staged.append((asset, path))
        for asset, path in staged:  # every file checked: now move them into place
            asset.dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(asset.dest))
    finally:
        for staging in {a.staging for a in plan.fetch}:
            shutil.rmtree(staging, ignore_errors=True)
    return [a.dest for a in plan.fetch]


def ensure_assets(settings: Settings, downloader: Downloader = hf_download) -> Plan:
    """Plan and fetch everything the configured service needs."""
    plan = plan_assets(settings, load_manifest(settings.assets.manifest))
    if plan.fetch:
        t0 = time.perf_counter()
        fetch_assets(plan, downloader)
        log.info("fetched %d file(s) in %.1fs", len(plan.fetch), time.perf_counter() - t0)
    elif plan.problems:
        fetch_assets(plan, downloader)  # raises with the problems
    else:
        for note in plan.notes:
            log.warning(note)
        log.info("all assets present (%d)", len(plan.present))
    return plan
