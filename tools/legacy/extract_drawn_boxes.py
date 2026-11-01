"""Recover which frames the report's run selected from its committed videos.

Repo 2 commits ``test_case_00X/output_annotated.mp4``: every sampled frame,
with a pure-green (0, 255, 0) box and "Target" label on the selected ones.
Counting green pixels per frame recovers the selection exactly (box positions
are approximate because the label text is green too).

Usage::

    python tools/legacy/extract_drawn_boxes.py --cases-dir ../legacy/amrit-pbvs \
        --out ../legacy/runs/drawn_boxes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

MIN_GREEN_PIXELS = 300


def drawn_frames(video: Path) -> tuple[list[int], dict[int, list[int]]]:
    cap = cv2.VideoCapture(str(video))
    frames, boxes, i = [], {}, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        green = (f[:, :, 1] > 200) & (f[:, :, 0] < 90) & (f[:, :, 2] < 90)
        if green.sum() > MIN_GREEN_PIXELS:
            ys, xs = np.nonzero(green)
            frames.append(i)
            boxes[i] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        i += 1
    cap.release()
    return frames, boxes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for case in sorted(args.cases_dir.glob("test_case_*")):
        frames, boxes = drawn_frames(case / "output_annotated.mp4")
        labels = json.loads((case / "evaluation.json").read_text())
        labelled = set(labels["true_positives"]) | set(labels["false_positives"])
        (args.out / f"{case.name}.json").write_text(json.dumps({"frames": frames, "boxes": boxes}))
        print(
            f"{case.name}: {len(frames)} boxed frames; labelled TP+FP {len(labelled)}; "
            f"boxed-not-labelled {len(set(frames) - labelled)}, "
            f"labelled-not-boxed {len(labelled - set(frames))}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
