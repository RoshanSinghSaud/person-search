"""Day 2 harness: run repo 2's ``run_reid`` *untouched* on the four test cases.

Nothing in the legacy checkout is edited. The harness only:

* overrides two module-level path constants (``YOLO_PATH`` is hardcoded to
  Amrit's disk, ``OSNET_MODEL_PATH`` points inside the repo),
* optionally overrides config constants / the DeepSort constructor so other
  configurations (report, paper) can be compared,
* wraps the two video writers so the selected frames are captured (and the
  slow video encoding can be skipped).

Output per test case (``<out>/<case>/``):

* ``result.json``  - config, selected frames (raw 1-based and 0-based annotated
  index), frame-level metrics vs ``evaluation.json``, timings, track index.
* ``run.log``      - the legacy stdout (check for "OSNet loaded successfully").
* ``tracks_crops/`` - the legacy crops (Day 9 needs their track IDs).

Usage::

    python tools/legacy/reproduce_repo2.py \
        --legacy-root ../legacy/amrit-pbvs \
        --yolo ../legacy/models_path/drive/yolo_finetuned_best.pt \
        --osnet ../legacy/models_path/drive/osnet_finetuned_model.pth.tar-60 \
        --out ../legacy/runs/repo2_default
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import importlib.util
import json
import re
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any


def load_legacy_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sampled_raw_indices(n_frames: int, fps: float, target_fps: float) -> list[int]:
    """Replicate the legacy time-based sampler: returns the 1-based raw frame
    indices that are processed (and written to output_annotated.mp4)."""
    out: list[int] = []
    step = 1.0 / target_fps
    next_t = 0.0
    for idx in range(1, n_frames + 1):
        if idx / fps < next_t:
            continue
        next_t += step
        out.append(idx)
    return out


def frame_metrics(selected: set[int], evaluation: dict[str, list[int]]) -> dict[str, Any]:
    gt = set(evaluation["true_positives"]) | set(evaluation["false_negatives"])
    report_selected = set(evaluation["true_positives"]) | set(evaluation["false_positives"])
    tp = len(selected & gt)
    fp = len(selected - gt)
    fn = len(gt - selected)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": p,
        "recall": r,
        "f1": f1,
        "matches_report_selection": selected == report_selected,
        "selection_jaccard_vs_report": (
            len(selected & report_selected) / len(selected | report_selected)
            if selected | report_selected
            else 1.0
        ),
        "missing_vs_report": sorted(report_selected - selected),
        "extra_vs_report": sorted(selected - report_selected),
    }


def index_tracks(crops_dir: Path) -> dict[str, list[dict[str, Any]]]:
    pat = re.compile(r"(\d+)_(-?\d+)_(-?\d+)_(-?\d+)_(-?\d+)\.jpg$")
    tracks: dict[str, list[dict[str, Any]]] = {}
    if not crops_dir.exists():
        return tracks
    for tdir in sorted(crops_dir.iterdir()):
        if not tdir.is_dir() or not tdir.name.startswith("id_"):
            continue
        tid = tdir.name[3:]
        rows = []
        for f in sorted(tdir.iterdir()):
            m = pat.match(f.name)
            if m:
                fr, left, top, right, bottom = map(int, m.groups())
                rows.append({"frame": fr, "bbox": [left, top, right, bottom]})
        tracks[tid] = rows
    return tracks


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--legacy-root", type=Path, required=True)
    ap.add_argument("--yolo", type=Path, required=True)
    ap.add_argument("--osnet", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cases", nargs="+", default=["001", "002", "003", "004"])
    ap.add_argument("--detection-confidence", type=float, default=None)
    ap.add_argument("--sim-threshold", type=float, default=None)
    ap.add_argument("--max-cosine-distance", type=float, default=None)
    ap.add_argument(
        "--write-videos", action="store_true", help="also encode the output videos (slow)"
    )
    args = ap.parse_args()

    legacy_root = args.legacy_root.resolve()
    mod = load_legacy_module(legacy_root / "pole" / "reid_main.py", "legacy_repo2_reid_main")
    mod.YOLO_PATH = str(args.yolo.resolve())
    mod.OSNET_MODEL_PATH = str(args.osnet.resolve())
    if args.detection_confidence is not None:
        mod.DETECTION_CONFIDENCE = args.detection_confidence
    if args.sim_threshold is not None:
        mod.SIM_THRESHOLD = args.sim_threshold
    if args.max_cosine_distance is not None:
        mod.DeepSort = functools.partial(mod.DeepSort, max_cosine_distance=args.max_cosine_distance)

    captured: dict[str, Any] = {}
    real_annotated = mod.create_annotated_video
    real_crop_video = mod.create_video_from_frames

    def capture_annotated(original_video_path, unique_frames, output_path, fps=10):
        captured["unique_frames"] = {int(k): dict(v) for k, v in unique_frames.items()}
        if args.write_videos:
            return real_annotated(original_video_path, unique_frames, output_path, fps)
        return True

    def maybe_crop_video(frame_list, output_path, fps=10):
        if args.write_videos:
            return real_crop_video(frame_list, output_path, fps)
        return True

    mod.create_annotated_video = capture_annotated
    mod.create_video_from_frames = maybe_crop_video

    import cv2  # legacy dependency, available in the legacy env

    config = {
        "DETECTION_CONFIDENCE": mod.DETECTION_CONFIDENCE,
        "SIM_THRESHOLD": mod.SIM_THRESHOLD,
        "RELATIVE_SCORE_MARGIN": mod.RELATIVE_SCORE_MARGIN,
        "OUTPUT_FPS": mod.OUTPUT_FPS,
        "RESIZE_WIDTH": mod.RESIZE_WIDTH,
        "max_cosine_distance": args.max_cosine_distance
        if args.max_cosine_distance is not None
        else 0.2,
        "yolo": mod.YOLO_PATH,
        "osnet": mod.OSNET_MODEL_PATH,
    }
    summary: dict[str, Any] = {"config": config, "cases": {}}
    args.out.mkdir(parents=True, exist_ok=True)

    for case in args.cases:
        case_dir = legacy_root / f"test_case_{case}"
        out_dir = (args.out / f"test_case_{case}").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        # run_reid wipes OUTPUT_DIR, so give it a dedicated subfolder
        reid_out = out_dir / "reid_output"
        captured.clear()
        log_path = out_dir / "run.log"
        t0 = time.perf_counter()
        with log_path.open("w") as log, contextlib.redirect_stdout(log):
            _results, best = mod.run_reid(
                str(case_dir / "test_video.mp4"), str(case_dir / "query.jpg"), str(reid_out)
            )
        seconds = time.perf_counter() - t0
        log_text = log_path.read_text()

        cap = cv2.VideoCapture(str(case_dir / "test_video.mp4"))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        sampled = sampled_raw_indices(n_frames, fps, mod.OUTPUT_FPS)
        raw_to_annot = {raw: i for i, raw in enumerate(sampled)}

        uf = captured.get("unique_frames", {})
        selected_raw = sorted(uf)
        selected_annot = sorted(raw_to_annot[r] for r in selected_raw if r in raw_to_annot)
        per_frame = {}
        for raw in selected_raw:
            info = uf[raw]
            tid_match = re.search(r"[/\\]id_([^/\\]+)[/\\]", info.get("path", ""))
            per_frame[str(raw_to_annot.get(raw, -1))] = {
                "raw_frame": raw,
                "track_id": tid_match.group(1) if tid_match else None,
                "bbox": list(info["bbox"]) if info.get("bbox") else None,
            }

        evaluation = json.loads((case_dir / "evaluation.json").read_text())
        metrics = frame_metrics(set(selected_annot), evaluation)
        tracks = index_tracks(reid_out / "tracks_crops")
        result = {
            "case": case,
            "config": config,
            "osnet_loaded": "OSNet loaded successfully" in log_text,
            "resnet_fallback": "Falling back to ResNet50" in log_text,
            "seconds": round(seconds, 2),
            "n_frames": n_frames,
            "fps": fps,
            "best_track": best,
            "selected_frames_raw": selected_raw,
            "selected_frames": selected_annot,
            "selected_detail": per_frame,
            "metrics_vs_ground_truth": metrics,
            "n_tracks": len(tracks),
        }
        (out_dir / "result.json").write_text(json.dumps(result, indent=2))
        (out_dir / "tracks.json").write_text(json.dumps(tracks))
        summary["cases"][case] = {
            k: result[k]
            for k in ("osnet_loaded", "resnet_fallback", "seconds", "best_track", "n_tracks")
        } | {
            "metrics": {
                k: v
                for k, v in metrics.items()
                if k not in ("missing_vs_report", "extra_vs_report")
            }
        }
        print(
            f"[{case}] {seconds:.1f}s  osnet_loaded={result['osnet_loaded']}  "
            f"F1={metrics['f1']:.4f}  matches_report={metrics['matches_report_selection']} "
            f"jaccard={metrics['selection_jaccard_vs_report']:.3f}",
            flush=True,
        )

    tp = sum(c["metrics"]["TP"] for c in summary["cases"].values())
    fp = sum(c["metrics"]["FP"] for c in summary["cases"].values())
    fn = sum(c["metrics"]["FN"] for c in summary["cases"].values())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    summary["system"] = {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["system"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
