"""Sweep the scoring parameters over cached ingests (no re-detection).

Day 2 question: which configuration produced the report's run? Detection and
tracking settings need a re-ingest; scoring settings do not. This loads the
``person-search ingest`` outputs for the 4 test cases and compares each scoring
configuration with three references:

* ``golden``  - repo 2 untouched (tests/golden/data)
* ``drawn``   - boxes drawn in the committed output_annotated.mp4 (the report's run)
* ``labels``  - frame-level F1 against evaluation.json ground truth (TP + FN)

Usage::

    python tools/analysis/sweep_scoring.py --ingest-dir ../legacy/runs/ingest \
        --cases-dir ../legacy/amrit-pbvs --drawn-dir ../legacy/runs/drawn_boxes
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import cv2

from person_search.config import ScoringSettings, load_settings
from person_search.eval.frame_metrics import FrameLabels, metrics_for_selection, system_metrics
from person_search.eval.golden import GoldenCase
from person_search.factory import build_osnet
from person_search.ingest.pipeline import IngestResult
from person_search.search.photo import photo_search

GOLDEN = Path(__file__).resolve().parents[2] / "tests" / "golden" / "data"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ingest-dir", type=Path, required=True)
    ap.add_argument("--cases-dir", type=Path, required=True)
    ap.add_argument("--drawn-dir", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    settings = load_settings(args.config)
    osnet = build_osnet(settings)
    cases = {}
    for npz in sorted(args.ingest_dir.glob("test_case_*.npz")):
        name = npz.stem
        query = osnet.encode_images([cv2.imread(str(args.cases_dir / name / "query.jpg"))])[0]
        drawn = None
        if args.drawn_dir and (args.drawn_dir / f"{name}.json").exists():
            drawn = set(json.loads((args.drawn_dir / f"{name}.json").read_text())["frames"])
        golden = (
            GoldenCase.load(GOLDEN / f"{name}.json") if (GOLDEN / f"{name}.json").exists() else None
        )
        cases[name] = {
            "obs": IngestResult.load(npz).observations,
            "query": query,
            "labels": FrameLabels.load(args.cases_dir / name / "evaluation.json"),
            "drawn": drawn,
            "golden": set(golden.selected_samples) if golden else None,
        }

    grid = itertools.product([0.5, 0.6, 0.65, 0.7], [0.05, 0.1], [0.9, 0.95, 1.0])
    rows = []
    for base, margin, ratio in grid:
        cfg = ScoringSettings(
            base_threshold=base, relative_margin=margin, frame_threshold_ratio=ratio
        )
        per_case, eq_golden, eq_drawn, jac_drawn = {}, 0, 0, []
        for name, c in cases.items():
            sel = set(photo_search(c["obs"], c["query"], cfg).selected_samples)
            per_case[name] = metrics_for_selection(sel, c["labels"])
            eq_golden += c["golden"] is not None and sel == c["golden"]
            if c["drawn"] is not None:
                eq_drawn += sel == c["drawn"]
                union = sel | c["drawn"]
                jac_drawn.append(len(sel & c["drawn"]) / len(union) if union else 1.0)
        s = system_metrics(per_case)
        rows.append(
            {
                "base": base,
                "margin": margin,
                "frame_ratio": ratio,
                "f1": round(s.f1, 4),
                "precision": round(s.precision, 4),
                "recall": round(s.recall, 4),
                "cases_equal_golden": eq_golden,
                "cases_equal_drawn": eq_drawn,
                "mean_jaccard_drawn": round(sum(jac_drawn) / len(jac_drawn), 4)
                if jac_drawn
                else None,
            }
        )
    rows.sort(key=lambda r: (-(r["mean_jaccard_drawn"] or 0), -r["f1"]))
    print(
        f"{'base':>5} {'margin':>6} {'ratio':>5} {'F1':>7} {'P':>7} {'R':>7}"
        "  =golden  =drawn  jacc(drawn)"
    )
    for r in rows:
        print(
            f"{r['base']:>5} {r['margin']:>6} {r['frame_ratio']:>5} {r['f1']:7.2%} "
            f"{r['precision']:7.2%} {r['recall']:7.2%}  {r['cases_equal_golden']:>7}  "
            f"{r['cases_equal_drawn']:>6}  {r['mean_jaccard_drawn']}"
        )
    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
