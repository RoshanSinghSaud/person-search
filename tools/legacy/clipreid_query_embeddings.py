"""Repo 1's CLIP-ReID image embedding of each test case's query photo (Day 12).

Repo 1's hybrid query adds this image embedding to an IRRA text embedding and
compares the sum with IRRA image vectors (AUDIT.md F2). To measure that bug
faithfully, the image half comes from repo 1's own code, untouched:
``ClipFeatureExtractor(image_ckpt_path=best_model_image.pth).extract_image_embedding``.

Needs OpenAI's ViT-B/16 in ``~/.cache/clip/`` (``clip.load`` downloads it) and
the CLIP-ReID checkpoint from Hugging Face ``prasish17/clip-image-query``.

Usage::

    python tools/legacy/clipreid_query_embeddings.py --repo1 ../legacy/person-reid-clip \
        --cases-dir ../legacy/amrit-pbvs --ckpt models/clipreid_image_legacy.pth \
        --out ../legacy/runs/week2/clipreid_query_embeddings.npz
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import sys
from pathlib import Path

import cv2
import numpy as np

CASES = ("test_case_001", "test_case_002", "test_case_003", "test_case_004")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo1", type=Path, required=True)
    ap.add_argument("--cases-dir", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    repo1 = args.repo1.resolve()
    sys.path.insert(0, str(repo1))
    spec = importlib.util.spec_from_file_location(
        "legacy_clip_inference", repo1 / "clip_inference.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        ext = mod.ClipFeatureExtractor(
            "cpu", image_ckpt_path=str(args.ckpt.resolve()), text_ckpt_path=None
        )
    if not ext.image_ckpt_loaded:
        sys.exit(f"CLIP-ReID checkpoint did not load:\n{log.getvalue()}")

    embs = {}
    for case in CASES:
        img = cv2.imread(str(args.cases_dir / case / "query.jpg"))
        feat = ext.extract_image_embedding(img)
        if feat is None:
            sys.exit(f"{case}: extraction failed")
        embs[case] = np.asarray(feat, dtype=np.float32).ravel()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **embs, log=np.array(log.getvalue()))
    print({c: v.shape for c, v in embs.items()}, "image_ckpt_loaded", ext.image_ckpt_loaded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
