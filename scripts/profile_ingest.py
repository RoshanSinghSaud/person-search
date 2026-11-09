"""Day 13: where ingest time goes, and what batching crops buys.

Runs the ingest pipeline on one test video with both encoders and reports the
time per stage (decode, YOLO11n, DeepSORT, OSNet, IRRA image) from the
pipeline's stage timer. ``--batch-size 1`` reproduces the legacy behaviour (one
crop per forward pass); the default batches all crops of a frame. ``--cprofile``
also writes the top functions by cumulative time.

For a sampling profile of a full run (no code changes, low overhead)::

    py-spy record -o profile.svg -- python scripts/profile_ingest.py --case test_case_002

Usage::

    python scripts/profile_ingest.py --cases-dir ../legacy/amrit-pbvs --case test_case_002 \
        --out ../legacy/runs/week2/profile_case2_batched.json
    python scripts/profile_ingest.py --cases-dir ../legacy/amrit-pbvs --case test_case_002 \
        --batch-size 1 --max-samples 240 --out ../legacy/runs/week2/profile_case2_b1_240.json
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import platform
import pstats
import time
from pathlib import Path

import torch

from person_search.config import load_settings
from person_search.factory import build_components


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cases-dir", type=Path, required=True)
    ap.add_argument("--case", default="test_case_002")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument(
        "--batch-size", type=int, default=None, help="crops per forward pass (1 = legacy)"
    )
    ap.add_argument(
        "--cprofile", type=Path, default=None, help="write top-40 cumulative functions here"
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    overrides = {}
    if args.batch_size:
        overrides = {
            "osnet": {"batch_size": args.batch_size},
            "irra": {"batch_size": args.batch_size},
        }
    settings = load_settings(**overrides)
    t0 = time.perf_counter()
    comps = build_components(settings)
    load_s = time.perf_counter() - t0
    pipe = comps.ingest_pipeline()
    video = args.cases_dir / args.case / "test_video.mp4"

    prof = cProfile.Profile() if args.cprofile else None
    if prof:
        prof.enable()
    result = pipe.run(video, max_samples=args.max_samples)
    if prof:
        prof.disable()
        buf = io.StringIO()
        pstats.Stats(prof, stream=buf).sort_stats("cumulative").print_stats(40)
        args.cprofile.parent.mkdir(parents=True, exist_ok=True)
        args.cprofile.write_text(buf.getvalue())

    total = sum(v["seconds"] for v in result.timings.values())
    n_crops = sum(o.has_embeddings for o in result.observations)
    stages = {
        name: {
            "seconds": v["seconds"],
            "share": v["seconds"] / total if total else 0.0,
            "calls": v["calls"],
        }
        for name, v in result.timings.items()
    }
    for name in ("embed:osnet", "embed:irra"):
        if name in stages and n_crops:
            stages[name]["ms_per_crop"] = 1000 * stages[name]["seconds"] / n_crops
    out = {
        "case": args.case,
        "batch_size": args.batch_size
        or {"osnet": settings.osnet.batch_size, "irra": settings.irra.batch_size},
        "samples": result.n_samples,
        "observations": len(result.observations),
        "embedded_crops": n_crops,
        "tracks": len(result.track_ids),
        "wall_seconds": round(result.wall_seconds, 2),
        "samples_per_s": round(result.fps_processed, 3),
        "model_load_seconds": round(load_s, 2),
        "stages": stages,
        "machine": {
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "torch": torch.__version__,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "stages"}, indent=2))
    for name, v in stages.items():
        print(
            f"  {name:<12} {v['seconds']:8.1f} s  {v['share']:6.1%}"
            + (f"  {v['ms_per_crop']:.0f} ms/crop" if "ms_per_crop" in v else "")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
