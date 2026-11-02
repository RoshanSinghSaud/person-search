"""Environment fingerprint: where does this machine's pipeline diverge from another's?

Prints (and optionally saves) a JSON with, stage by stage:

1. versions and platform,
2. md5 of the first decoded frames of each test video (OpenCV/FFmpeg decode),
3. YOLO11n detections on frame 1 of each video (boxes rounded to 0.01 px),
4. OSNet embedding of each query photo (sum and first values, 6 decimals).

Run it on two machines and diff the files: the first stage that differs is
where the outputs start to drift.

Usage::

    uv run python tools/diagnostics/fingerprint.py --cases-dir ../legacy/amrit-pbvs \
        --out fingerprint-$(uname -s)-$(uname -m).json
    uv run python tools/diagnostics/fingerprint.py --compare a.json b.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import platform
import sys
from pathlib import Path
from typing import Any

CASES = ("test_case_001", "test_case_002", "test_case_003", "test_case_004")
PACKAGES = ("torch", "torchvision", "opencv-python", "numpy", "pillow", "ultralytics", "scipy")


def platform_tag() -> str:
    return f"{sys.platform}-{platform.machine().lower()}"


def collect(cases_dir: Path, models_dir: Path, n_frames: int) -> dict[str, Any]:
    import cv2
    import numpy as np
    import torch

    from person_search.ingest.detector import YoloDetector
    from person_search.models.osnet_encoder import OSNetEncoder

    out: dict[str, Any] = {
        "platform": platform_tag(),
        "python": platform.python_version(),
        "versions": {p: md.version(p) for p in PACKAGES},
        "torch_threads": torch.get_num_threads(),
        "decode": {},
        "yolo_frame1": {},
        "osnet_query": {},
    }
    detector = YoloDetector(models_dir / "yolo11n_person.pt")
    osnet = OSNetEncoder(models_dir / "osnet_campus_v1.pth.tar")
    for case in CASES:
        cap = cv2.VideoCapture(str(cases_dir / case / "test_video.mp4"))
        hashes, first = [], None
        for _ in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                break
            first = frame if first is None else first
            hashes.append(hashlib.md5(frame.tobytes()).hexdigest()[:12])
        cap.release()
        out["decode"][case] = hashes
        dets = detector.detect(first)
        out["yolo_frame1"][case] = [
            [round(d.x, 2), round(d.y, 2), round(d.w, 2), round(d.h, 2), round(d.confidence, 4)]
            for d in dets
        ]
        q = osnet.encode_images([cv2.imread(str(cases_dir / case / "query.jpg"))])[0]
        out["osnet_query"][case] = {
            "sum": round(float(np.sum(q)), 6),
            "head": [round(float(v), 6) for v in q[:5]],
        }
    return out


def compare(a: dict[str, Any], b: dict[str, Any]) -> int:
    print(f"A: {a['platform']}  B: {b['platform']}")
    for p in PACKAGES:
        va, vb = a["versions"].get(p), b["versions"].get(p)
        if va != vb:
            print(f"  version differs: {p} {va} vs {vb}")
    first_diff = None
    for stage in ("decode", "yolo_frame1", "osnet_query"):
        same = [c for c in CASES if a[stage].get(c) == b[stage].get(c)]
        status = (
            "identical"
            if len(same) == len(CASES)
            else f"differs in {len(CASES) - len(same)}/4 cases"
        )
        print(f"  {stage:<12} {status}")
        if first_diff is None and len(same) != len(CASES):
            first_diff = stage
            c = next(c for c in CASES if c not in same)
            print(f"    e.g. {c}:\n      A {a[stage][c]}\n      B {b[stage][c]}")
    print(f"first diverging stage: {first_diff or 'none'}")
    return 0 if first_diff is None else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cases-dir", type=Path, default=Path("../legacy/amrit-pbvs"))
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("A", "B"))
    args = ap.parse_args()
    if args.compare:
        a, b = (json.loads(p.read_text()) for p in args.compare)
        return compare(a, b)
    result = collect(args.cases_dir, args.models_dir, args.frames)
    text = json.dumps(result, indent=1)
    if args.out:
        args.out.write_text(text)
        print(f"saved {args.out}")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
