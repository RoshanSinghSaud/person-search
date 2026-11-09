"""Contact sheets of every track in an ingest, next to the query photo.

Used on Day 9 to label which tracks are each test case's target person (the
text test set's ground truth). Labelling is done by eye: the frame labels
cannot say which person in a crowded frame is the target, so we look at the
crops. Each sheet shows the query photo, then one row per track with crops
spread over the track's life and the track's frame span.

Usage::

    python tools/analysis/track_contact_sheet.py --ingest ../legacy/runs/ingest_v1/test_case_001.npz \
        --video ../legacy/amrit-pbvs/test_case_001/test_video.mp4 \
        --query ../legacy/amrit-pbvs/test_case_001/query.jpg --out ../legacy/runs/text_targets/test_case_001

The sheets show people from the test videos: keep them out of the repo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from person_search.ingest.pipeline import IngestResult
from person_search.ingest.video import iter_sampled_frames

CELL_H, CELL_W = 160, 80
PER_ROW = 6
ROWS_PER_PAGE = 12


def _fit(img: np.ndarray, h: int = CELL_H, w: int = CELL_W) -> np.ndarray:
    scale = min(h / img.shape[0], w / img.shape[1])
    small = cv2.resize(img, (max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale))))
    out = np.full((h, w, 3), 255, np.uint8)
    y, x = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
    out[y : y + small.shape[0], x : x + small.shape[1]] = small
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ingest", type=Path, required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--query", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="output prefix")
    ap.add_argument("--fps", type=float, default=12.0)
    args = ap.parse_args()

    result = IngestResult.load(args.ingest)
    tracks = result.by_track()
    wanted: dict[int, list[tuple[str, int]]] = {}
    for tid, obs in tracks.items():
        picks = np.linspace(0, len(obs) - 1, num=min(PER_ROW, len(obs))).round().astype(int)
        for k, i in enumerate(sorted(set(picks.tolist()))):
            wanted.setdefault(obs[i].sample_index, []).append((tid, k))

    crops: dict[tuple[str, int], np.ndarray] = {}
    lookup = {(o.track_id, o.sample_index): o for o in result.observations}
    for sample in iter_sampled_frames(args.video, args.fps, 30.0):
        for tid, k in wanted.get(sample.sample_index, []):
            box = lookup[(tid, sample.sample_index)].bbox
            crops[(tid, k)] = box.crop(sample.image).copy()

    query = cv2.imread(str(args.query))
    order = sorted(tracks, key=lambda t: (tracks[t][0].sample_index, int(t) if t.isdigit() else 0))
    label_w = 150
    pages = [order[i : i + ROWS_PER_PAGE] for i in range(0, len(order), ROWS_PER_PAGE)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for p, page in enumerate(pages, 1):
        sheet_w = label_w + PER_ROW * CELL_W + CELL_W * 2
        sheet = np.full((max(len(page) * CELL_H, CELL_H * 2 + 30), sheet_w, 3), 255, np.uint8)
        q = _fit(query, CELL_H * 2, CELL_W * 2)
        sheet[: CELL_H * 2, -CELL_W * 2 :] = q
        cv2.putText(sheet, "QUERY", (sheet_w - CELL_W * 2 + 5, CELL_H * 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        for r, tid in enumerate(page):
            obs = tracks[tid]
            y = r * CELL_H
            cv2.putText(sheet, f"track {tid}", (5, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            span = f"s{obs[0].sample_index}-{obs[-1].sample_index}"
            cv2.putText(sheet, span, (5, y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
            cv2.putText(sheet, f"n={len(obs)}", (5, y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
            for k in range(PER_ROW):
                if (tid, k) in crops:
                    x = label_w + k * CELL_W
                    sheet[y : y + CELL_H, x : x + CELL_W] = _fit(crops[(tid, k)])
            cv2.line(sheet, (0, y + CELL_H - 1), (label_w + PER_ROW * CELL_W, y + CELL_H - 1), (200, 200, 200), 1)
        path = args.out.with_name(f"{args.out.name}_p{p}.jpg")
        cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
