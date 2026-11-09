"""Campus ReID test split (Market-1501 layout) and image-to-image ReID metrics.

The split is the team's private campus dataset (``custom_reid_dataset/``):
``query/`` and ``bounding_box_test/`` hold the 30 identities kept out of
fine-tuning, named ``<pid>_c<cam>s<seq>_<frame>_<n>.jpg``.

Two scoring protocols:

* ``euclidean_raw`` - torchreid's default test protocol, which produced the
  published campus numbers (75.4% mAP / 81.7% Rank-1 for osnet v1, 29.2% mAP for
  Market-1501 osnet v0): Euclidean distance between unnormalised features.
* ``cosine`` - what the service does: cosine similarity of L2-normalised
  features. The gate uses this one, because it measures the deployed behaviour.

Images are read with PIL (as torchreid does) and handed to the encoders as BGR
arrays (as the video pipeline does).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
from PIL import Image

from person_search.eval.reid import RetrievalMetrics, evaluate_reid

_NAME = re.compile(r"([-\d]+)_c(\d+)")
ScoreProtocol = Literal["cosine", "euclidean_raw"]


class FeatureEncoder(Protocol):
    """An image encoder that can also return its raw (unnormalised) features."""

    def encode_images(
        self, images: Sequence[np.ndarray], *, normalize: bool = True
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class ReidSplit:
    query: list[Path]
    q_pids: np.ndarray
    q_camids: np.ndarray
    gallery: list[Path]
    g_pids: np.ndarray
    g_camids: np.ndarray

    @property
    def n_ids(self) -> int:
        return len(set(self.q_pids.tolist()) | set(self.g_pids.tolist()))


def _scan(folder: Path) -> tuple[list[Path], np.ndarray, np.ndarray]:
    paths, pids, cams = [], [], []
    for p in sorted([*folder.glob("*.jpg"), *folder.glob("*.png")]):
        m = _NAME.search(p.name)
        if m is None:
            raise ValueError(f"unexpected file name in {folder}: {p.name}")
        pid, cam = int(m.group(1)), int(m.group(2))
        if pid == -1:  # junk images, as in torchreid
            continue
        paths.append(p)
        pids.append(pid)
        cams.append(cam - 1)
    if not paths:
        raise FileNotFoundError(f"no images in {folder}")
    return paths, np.array(pids), np.array(cams)


def load_market_split(root: str | Path) -> ReidSplit:
    root = Path(root)
    q, qp, qc = _scan(root / "query")
    g, gp, gc = _scan(root / "bounding_box_test")
    return ReidSplit(q, qp, qc, g, gp, gc)


def read_bgr(path: Path) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(Image.open(path).convert("RGB"))[:, :, ::-1])


def encode_paths(
    encoder: FeatureEncoder, paths: Sequence[Path], *, normalize: bool = True, chunk: int = 64
) -> np.ndarray:
    feats = []
    for i in range(0, len(paths), chunk):
        imgs = [read_bgr(p) for p in paths[i : i + chunk]]
        feats.append(encoder.encode_images(imgs, normalize=normalize))
    return np.concatenate(feats).astype(np.float32)


def similarity(qf: np.ndarray, gf: np.ndarray, protocol: ScoreProtocol) -> np.ndarray:
    if protocol == "cosine":
        return qf @ gf.T
    if protocol == "euclidean_raw":
        d = (qf**2).sum(1)[:, None] + (gf**2).sum(1)[None, :] - 2.0 * qf @ gf.T
        return -np.sqrt(np.maximum(d, 0.0))
    raise ValueError(f"unknown protocol {protocol!r}")


def evaluate_encoder(
    encoder: FeatureEncoder,
    split: ReidSplit,
    protocols: Sequence[ScoreProtocol] = ("cosine", "euclidean_raw"),
) -> dict[str, RetrievalMetrics]:
    """Campus mAP and CMC for an image encoder, under each protocol."""
    out: dict[str, RetrievalMetrics] = {}
    raw = {"q": encode_paths(encoder, split.query, normalize=False)}
    raw["g"] = encode_paths(encoder, split.gallery, normalize=False)
    for proto in protocols:
        if proto == "cosine":
            qf = raw["q"] / (np.linalg.norm(raw["q"], axis=1, keepdims=True) + 1e-6)
            gf = raw["g"] / (np.linalg.norm(raw["g"], axis=1, keepdims=True) + 1e-6)
        else:
            qf, gf = raw["q"], raw["g"]
        sim = similarity(qf.astype(np.float64), gf.astype(np.float64), proto)
        out[proto] = evaluate_reid(sim, split.q_pids, split.g_pids, split.q_camids, split.g_camids)
    return out
