"""Day 8: score every model version on the gate set, then record them in the MLflow registry.

``evaluate`` scores the versions in ``configs/models.yaml`` on the gate set
(``scripts/gate_set.py build``) and writes one JSON file:

* OSNet versions: campus mAP / Rank-k (cosine, and torchreid's Euclidean
  protocol that produced the published numbers) and photo frame F1 on the 4
  cached test cases;
* IRRA versions: text Hit@1 / Hit@5 / MRR on the 12 witness descriptions, and
  the image tower's campus mAP as a secondary check. The college text-image
  split is still missing, so IRRA's published Rank-1 cannot be re-measured yet.

``register`` logs one MLflow run per version (params, metrics, a small
``model_ref.json`` with the weights' source and SHA-256) and registers it under
``person-search-osnet`` / ``person-search-irra``, with the ``production`` alias
on the versions pinned in ``configs/models.yaml``. The weights themselves stay
where they are (Drive / Hugging Face); the registry records which file, by hash.

Tracking server: ``MLFLOW_TRACKING_URI`` (DagsHub: ``https://dagshub.com/<user>/<repo>.mlflow``
plus ``MLFLOW_TRACKING_USERNAME`` / ``MLFLOW_TRACKING_PASSWORD``), else a local
SQLite file given by ``--tracking-uri``.

Usage::

    python scripts/eval_models.py evaluate --gate-dir ../legacy/gate_set \
        --out ../legacy/runs/week2/model_eval.json
    python scripts/eval_models.py register --results ../legacy/runs/week2/model_eval.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

from person_search.config import load_settings
from person_search.eval.campus import load_market_split
from person_search.eval.frame_metrics import FrameLabels
from person_search.eval.gate import (
    campus_metrics,
    compare_versions,
    photo_metrics,
    text_metrics,
)
from person_search.eval.gate_set import CachedCase, load_case
from person_search.eval.text_e2e import load_targets, load_text_queries
from person_search.factory import configure_torch
from person_search.ingest.pipeline import IngestResult
from person_search.models.irra_encoder import IRRAEncoder
from person_search.models.osnet_encoder import OSNetEncoder
from person_search.types import Observation

CASES = ("test_case_001", "test_case_002", "test_case_003", "test_case_004")
REPO = Path(__file__).resolve().parents[1]
FAMILY_MODEL = {"osnet": "person-search-osnet", "irra": "person-search-irra"}


def reuse_ingest(
    args: argparse.Namespace,
    version: str,
    name: str,
    enc: Any,
    cases: dict[str, CachedCase],
    details: dict[str, Any],
) -> dict[str, list[Observation]] | None:
    """Take the vectors from the ingest when it was made by this very version.

    The cached crops would give bit-identical vectors (they were cut from that
    ingest); a sample of frames per case is re-embedded to prove it. Saves
    re-embedding ~5,500 crops on a CPU. The CI gate always re-embeds.
    """
    if args.reuse_ingest is None:
        return None
    ingests = {c: IngestResult.load(args.reuse_ingest / f"{c}.npz") for c in cases}
    if any(r.encoder_versions.get(name) != version for r in ingests.values()):
        return None
    checked = sum(case.spot_check(ingests[c], name, enc) for c, case in cases.items())
    details["vectors"] = (
        f"from the ingest in {args.reuse_ingest.name}; {checked} re-embedded crops "
        "bit-identical to it"
    )
    return {c: case.from_ingest(ingests[c], name) for c, case in cases.items()}


def evaluate(args: argparse.Namespace) -> int:
    cfg = yaml.safe_load(args.config.read_text())
    versions = args.versions or list(cfg["models"])
    settings = load_settings()
    configure_torch(settings)  # PS_TORCH_THREADS: thread count changes vectors in the last bit
    gate = args.gate_dir
    split = load_market_split(gate / "campus")
    cases = {c: load_case(gate / "cases" / f"{c}.npz") for c in args.cases}
    labels = {c: FrameLabels.load(gate / "cases" / c / "evaluation.json") for c in args.cases}
    queries = [
        q for q in load_text_queries(gate / "text" / "text_queries.yaml") if q.case in args.cases
    ]
    targets = load_targets(gate / "text" / "text_targets.yaml")
    results: dict[str, Any] = (
        json.loads(args.out.read_text()) if args.out.exists() and args.resume else {}
    )
    results.setdefault("versions", {})
    results["production"] = cfg["production"]
    results["gate_manifest_built"] = json.loads((gate / "manifest.json").read_text())["built"]
    for version in versions:
        if version in results["versions"] and args.resume:
            print("skip (done)", version)
            continue
        spec = cfg["models"][version]
        path = args.models_dir / spec["file"]
        t0 = time.perf_counter()
        metrics: dict[str, Any] = {}
        details: dict[str, Any] = {}
        reuse = None
        if spec["family"] == "osnet":
            enc = OSNetEncoder(path, version=version)
            reuse = reuse_ingest(args, version, "osnet", enc, cases, details)
            metrics |= campus_metrics(enc, split, ("cosine", "euclidean_raw"))
            pm = photo_metrics(enc, cases, labels, gate / "cases", settings.scoring, reuse)
            details["photo"] = {k: pm.pop(k) for k in ("per_case", "selected_samples")}
            metrics |= pm
        else:
            onnx = spec.get("image_onnx")
            enc = IRRAEncoder(
                path,
                version=version,
                image_onnx=args.models_dir / onnx if onnx else None,
                onnx_threads=settings.torch_threads,
            )
            reuse = reuse_ingest(args, version, "irra", enc, cases, details)
            metrics |= campus_metrics(enc, split, ("cosine",))
            tm = text_metrics(enc, cases, queries, targets, labels, reuse)
            details["text"] = tm.pop("per_query")
            metrics |= tm
        results["versions"][version] = {
            "family": spec["family"],
            "file": spec["file"],
            "sha256": spec["sha256"],
            "source": spec["source"],
            "role": spec["role"],
            "metrics": metrics,
            "details": details,
            "seconds": round(time.perf_counter() - t0, 1),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1))
        print(version, {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})
    return 0


def register(args: argparse.Namespace) -> int:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    results = json.loads(args.results.read_text())
    uri = os.environ.get("MLFLOW_TRACKING_URI") or args.tracking_uri
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(args.experiment)
    client = mlflow.MlflowClient()
    production = results["production"]
    for version, r in sorted(results["versions"].items()):  # v0 before v1
        name = FAMILY_MODEL[r["family"]]
        with mlflow.start_run(run_name=f"gate-{version}") as run:
            mlflow.set_tags({"semver": version, "family": r["family"], "role": r["role"]})
            mlflow.log_params(
                {
                    "version": version,
                    "file": r["file"],
                    "sha256": r["sha256"],
                    "source": r["source"][:250],
                    "gate_set_built": results["gate_manifest_built"],
                }
            )
            mlflow.log_metrics(
                {k: float(v) for k, v in r["metrics"].items() if isinstance(v, (int, float))}
            )
            with tempfile.TemporaryDirectory() as tmp:
                ref = Path(tmp) / "model_ref.json"
                ref.write_text(
                    json.dumps(
                        {k: r[k] for k in ("family", "file", "sha256", "source", "role")}
                        | {"version": version},
                        indent=2,
                    )
                )
                (Path(tmp) / "gate_details.json").write_text(json.dumps(r["details"], indent=1))
                mlflow.log_artifacts(tmp, artifact_path="model")
            artifact = f"{run.info.artifact_uri}/model"
        with contextlib.suppress(mlflow.exceptions.MlflowException):  # already exists
            client.create_registered_model(
                name, description=f"{r['family']} encoder versions (gate-scored)"
            )
        mv = client.create_model_version(
            name,
            source=artifact,
            run_id=run.info.run_id,
            tags={"semver": version, "sha256": r["sha256"]},
            description=f"{version}: {r['role']}",
        )
        if production.get(r["family"]) == version:
            client.set_registered_model_alias(name, "production", mv.version)
        print(
            f"registered {name} v{mv.version} = {version}"
            + (" (production)" if production.get(r["family"]) == version else "")
        )
    print("tracking:", uri)
    return 0


def compare(args: argparse.Namespace) -> int:
    """Gate verdict for a candidate version against the production version of its family."""
    versions: dict[str, Any] = {}
    production: dict[str, str] = {}
    for path in args.results:
        data = json.loads(path.read_text())
        versions.update(data["versions"])
        production.update(data.get("production", {}))
    cand = versions[args.candidate]
    prod_name = production[cand["family"]]
    failures = compare_versions(versions[prod_name], cand)
    pm, cm = versions[prod_name]["metrics"], cand["metrics"]
    for key in sorted(set(pm) & set(cm)):
        if isinstance(pm[key], float):
            print(f"  {key:<28} {prod_name}: {pm[key]:.4f}   {args.candidate}: {cm[key]:.4f}")
    if failures:
        print(f"BLOCKED: {args.candidate} vs {prod_name}: " + "; ".join(failures))
        return 1
    print(f"PASSED: {args.candidate} vs {prod_name}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--gate-dir", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--models-dir", type=Path, default=REPO / "models")
    e.add_argument("--config", type=Path, default=REPO / "configs" / "models.yaml")
    e.add_argument("--versions", nargs="*")
    e.add_argument("--resume", action="store_true", help="skip versions already in --out")
    e.add_argument(
        "--reuse-ingest",
        type=Path,
        default=None,
        help="ingest dir: reuse its vectors for the version that made it (spot-checked)",
    )
    e.add_argument("--cases", nargs="+", default=list(CASES))
    e.set_defaults(func=evaluate)
    c = sub.add_parser("compare", help="gate verdict: candidate vs production (exit 1 = blocked)")
    c.add_argument("--results", type=Path, nargs="+", required=True)
    c.add_argument("--candidate", required=True)
    c.set_defaults(func=compare)
    r = sub.add_parser("register")
    r.add_argument("--results", type=Path, required=True)
    r.add_argument(
        "--tracking-uri",
        default="sqlite:///" + (REPO.parent / "legacy/runs/mlflow/mlflow.db").as_posix(),
    )
    r.add_argument("--experiment", default="person-search-gate")
    r.set_defaults(func=register)
    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
