"""Day 2: time one text query with repo 1, untouched.

Runs ``reid_main.run_reid(model_type="clip", clip_query_mode="text")`` on one
video. Only the output-video writers are wrapped (skipped) so the number is the
query itself: model loading, decode, detection, tracking, IRRA embedding and
scoring. Model loading (HF download check + two CLIP models + both
checkpoints) is timed once on its own first, so ``seconds_excl_loading`` is an
estimate of the query cost alone.

The description is deliberately generic: Day 9's witness descriptions must be
written before any text search is looked at, so this script prints timings
only, never rankings.

Usage::

    python tools/legacy/time_repo1_text.py --repo1 ../legacy/person-reid-clip \
        --video ../legacy/amrit-pbvs/test_case_001/test_video.mp4 --out ../legacy/runs/repo1_text
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo1", type=Path, required=True)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--text", default="a person walking")
    args = ap.parse_args()

    repo1 = args.repo1.resolve()
    sys.path.insert(0, str(repo1))  # reid_main imports clip_inference from its own folder
    spec = importlib.util.spec_from_file_location("legacy_repo1_reid_main", repo1 / "reid_main.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    mod.create_video_from_memory_crops = lambda *a, **k: True
    mod.create_annotated_video = lambda *a, **k: True
    mod.dump_selected_crops_to_disk = lambda *a, **k: 0

    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / "run.log"

    import torch
    from huggingface_hub import hf_hub_download

    # download (untimed) so both timed runs read from the local HF cache
    image_ckpt = hf_hub_download("prasish17/clip-image-query", "best_model_image.pth")
    text_ckpt = hf_hub_download("prasish17/clip-text-query", "best_model_text.pth")
    with (args.out / "load.log").open("w") as log, contextlib.redirect_stdout(log):
        t_load = time.perf_counter()
        mod_ci = sys.modules.get("clip_inference") or __import__("clip_inference")
        mod_ci.ClipFeatureExtractor(torch.device("cpu"), image_ckpt, text_ckpt)
        load_seconds = time.perf_counter() - t_load
    t0 = time.perf_counter()
    with log_path.open("w") as log, contextlib.redirect_stdout(log):
        mod.run_reid(
            str(args.video.resolve()),
            None,
            str((args.out / "reid_output").resolve()),
            model_type="clip",
            clip_query_mode="text",
            INPUT_PERSON_TEXT=args.text,
        )
    total = time.perf_counter() - t0
    log_text = log_path.read_text()
    result = {
        "video": args.video.name,
        "seconds_total": round(total, 1),
        "seconds_model_loading": round(load_seconds, 1),
        "seconds_excl_loading": round(total - load_seconds, 1),
        "irra_prompts_loaded": "IRRA prompt tokens loaded" in log_text,
        "base_clip_fallback": "Using base CLIP weights" in log_text,
        "note": "includes loading two CLIP models + checkpoints; output videos skipped",
    }
    (args.out / "timing.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
