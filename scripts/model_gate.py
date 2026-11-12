"""Day 24: the model gate in CI. A pull request that changes which model version
production runs is scored on the private gate set, and blocked if it is worse.

.github/workflows/model-gate.yml runs this on every pull request (decision 0016):

``plan``
    Compare the PR's ``configs/models.yaml`` with the base branch's: which
    ``production`` pins change (one pair per family: production -> candidate).
    Versions are immutable, so changing an existing version's file or hash in
    place is refused (add a new version instead). No pin changed: nothing to
    score, and the check passes in seconds.
``fetch``
    Download the weights both sides need into ``models/``, SHA-256 checked.
    Sources, in order: ``configs/assets.yaml``; OpenAI's URL for vanilla CLIP
    (``scripts/fetch_weights.py``); else ``<assets repo>/weights/<file>``.
``verdict``
    Apply the gate rule (``person_search.eval.gate.compare_versions``, decision
    0010) to the two ``eval_models.py evaluate`` outputs and write a Markdown
    report (job summary and PR comment). Exit 1 = blocked.

Rules (decision 0010): campus mAP may drop at most 1 point; OSNet photo frame F1
at most 1 point; IRRA may not lose any text query that production ranks first.

Usage, from the repo root::

    python -m scripts.model_gate plan --base base_models.yaml --head configs/models.yaml \
        --out plan.json
    python -m scripts.model_gate fetch --plan plan.json --models-dir models
    python -m scripts.model_gate verdict --plan plan.json --baseline baseline.json \
        --candidate candidate.json --report gate.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from scripts.fetch_weights import URL_FILES

REPO = Path(__file__).resolve().parents[1]
SPEC_KEYS = ("family", "file", "image_onnx", "sha256")
VERSION_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
MARKER = "<!-- model-gate -->"

# (metric key, label, format, rule prefix in compare_versions' messages, rule text)
ROWS: dict[str, list[tuple[str, str, str, str | None, str]]] = {
    "osnet": [
        ("campus_mAP_cosine", "campus mAP", "pct", "campus mAP", "may drop at most 1 pt"),
        ("campus_rank1_cosine", "campus Rank-1", "pct", None, ""),
        ("photo_f1", "photo frame F1 (4 videos)", "pct", "photo F1", "may drop at most 1 pt"),
    ],
    "irra": [
        ("text_hit1_count", "text Hit@1", "count", "text Hit@1", "no query lost"),
        ("text_hit5_count", "text Hit@5", "count", None, ""),
        ("text_mrr", "text MRR", "float", None, ""),
        (
            "campus_mAP_cosine",
            "image tower campus mAP",
            "pct",
            "campus mAP",
            "may drop at most 1 pt",
        ),
        ("campus_rank1_cosine", "image tower campus Rank-1", "pct", None, ""),
    ],
}


class GateError(Exception):
    """The PR's model configuration cannot be gated as it is (exit 2)."""


# ---------------------------------------------------------------------- plan
def make_plan(base: dict[str, Any], head: dict[str, Any]) -> dict[str, Any]:
    """Which production pins the PR changes, as (family, baseline, candidate) pairs."""
    bad_names = sorted(v for v in head["models"] if not VERSION_NAME.match(v))
    if bad_names:
        raise GateError(f"version names must match {VERSION_NAME.pattern}: {bad_names}")
    edited = sorted(
        v
        for v in set(base["models"]) & set(head["models"])
        if any(base["models"][v].get(k) != head["models"][v].get(k) for k in SPEC_KEYS)
    )
    if edited:
        raise GateError(
            f"model versions are immutable, but {', '.join(edited)} changed file or hash in "
            "place: add a new version (e.g. irra-v3) and point production at it"
        )
    removed = sorted(set(base["production"]) - set(head["production"]))
    if removed:
        raise GateError(f"production pins removed: {removed}")
    pairs = []
    for family, candidate in sorted(head["production"].items()):
        spec = head["models"].get(candidate)
        if spec is None:
            raise GateError(f"production.{family} = {candidate} is not listed under models:")
        if spec["family"] != family:
            raise GateError(f"production.{family} = {candidate} is a {spec['family']} model")
        baseline = base["production"].get(family)
        if baseline is None:
            raise GateError(f"no production {family} on the base branch to compare with")
        if baseline != candidate:
            pairs.append({"family": family, "baseline": baseline, "candidate": candidate})
    return {
        "pairs": pairs,
        "baseline_models": {p["baseline"]: base["models"][p["baseline"]] for p in pairs},
        "candidate_models": {p["candidate"]: head["models"][p["candidate"]] for p in pairs},
    }


def plan_report(plan: dict[str, Any]) -> str:
    if not plan["pairs"]:
        return f"{MARKER}\n## Model gate: no production model change\n\nNothing to score.\n"
    lines = [f"{MARKER}\n## Model gate: scoring on the gate set\n"]
    lines += [f"- {p['family']}: `{p['baseline']}` -> `{p['candidate']}`" for p in plan["pairs"]]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------- fetch
@dataclass(frozen=True)
class Source:
    kind: str  # "hf" or "url"
    where: str  # repo id or URL
    filename: str = ""
    revision: str = "main"

    def __str__(self) -> str:
        if self.kind == "url":
            return self.where
        return f"hf://{self.where}/{self.filename}@{self.revision[:12]}"


def needed_files(plan: dict[str, Any]) -> list[str]:
    specs = [*plan["baseline_models"].values(), *plan["candidate_models"].values()]
    names = {s["file"] for s in specs} | {s["image_onnx"] for s in specs if s.get("image_onnx")}
    return sorted(names)


def expected_hashes(models: dict[str, Any], manifest: dict[str, Any]) -> dict[str, str]:
    """File name -> SHA-256: configs/assets.yaml first, else the models.yaml entry owning it
    (an entry with ``image_onnx`` hashes that file; its ``file`` is another version's)."""
    out: dict[str, str] = {}
    for spec in models.values():
        out.setdefault(spec.get("image_onnx") or spec["file"], spec["sha256"])
    for name, entry in manifest.get("weights", {}).items():
        out[name] = entry["sha256"]
    return out


def source_for(name: str, manifest: dict[str, Any], assets_repo: str | None) -> Source:
    entry = manifest.get("weights", {}).get(name)
    if entry and entry.get("repo"):
        return Source(
            "hf", entry["repo"], entry.get("filename", name), entry.get("revision", "main")
        )
    if name in URL_FILES:
        return Source("url", URL_FILES[name])
    if not assets_repo:
        raise GateError(f"{name}: no source (set `repo:` in configs/assets.yaml)")
    # a file in the service's manifest comes from its pinned commit; a gate-only file
    # (e.g. the Market-1501 OSNet) from the latest one. The SHA-256 check holds either way.
    revision = str(manifest.get("revision") or "main") if entry else "main"
    return Source("hf", assets_repo, f"weights/{name}", revision)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(src: Source, tmp: Path, token: str | None) -> Path:
    if src.kind == "url":
        out = tmp / Path(src.where).name
        with urllib.request.urlopen(src.where, timeout=60) as r, out.open("wb") as f:
            shutil.copyfileobj(r, f)
        return out
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            src.where, src.filename, revision=src.revision, local_dir=tmp, token=token or None
        )
    )


def fetch(
    plan: dict[str, Any],
    models: dict[str, Any],
    manifest: dict[str, Any],
    models_dir: Path,
    assets_repo: str | None,
    token: str | None,
) -> None:
    hashes = expected_hashes(models, manifest)
    models_dir.mkdir(parents=True, exist_ok=True)
    for name in needed_files(plan):
        want = hashes.get(name)
        if not want:
            raise GateError(f"{name}: no SHA-256 in configs/assets.yaml or configs/models.yaml")
        dest = models_dir / name
        if dest.is_file() and sha256_file(dest) == want:
            print(f"present  {name}")
            continue
        src = source_for(name, manifest, assets_repo)
        with tempfile.TemporaryDirectory(dir=models_dir) as tmp:
            try:
                path = download(src, Path(tmp), token)
            except Exception as e:
                raise GateError(
                    f"{name}: cannot download {src}: {type(e).__name__}: {e}. A private repo "
                    "needs HF_TOKEN (repository secret HF_READ_TOKEN)"
                ) from e
            got = sha256_file(path)
            if got != want:
                raise GateError(f"{name} from {src}: sha256 {got}, expected {want}")
            shutil.move(path, dest)
        print(f"fetched  {name}  {dest.stat().st_size / 1e6:.1f} MB  from {src}")


# ------------------------------------------------------------------- verdict
def _fmt(value: Any, kind: str, n: int) -> str:
    if value is None:
        return "n/a"
    if kind == "pct":
        return f"{100 * value:.2f}%"
    if kind == "count":
        return f"{value}/{n}"
    return f"{value:.3f}"


def _delta(p: Any, c: Any, kind: str) -> str:
    if p is None or c is None:
        return ""
    if kind == "pct":
        return f"{100 * (c - p):+.2f} pts"
    if kind == "count":
        return f"{c - p:+d}"
    return f"{c - p:+.3f}"


def judge(pair: dict[str, str], prod: dict[str, Any], cand: dict[str, Any]) -> tuple[bool, str]:
    """(blocked, Markdown section) for one family."""
    from person_search.eval.gate import compare_versions

    failures = compare_versions(prod, cand)
    p, c = prod["metrics"], cand["metrics"]
    n = int(p.get("text_n") or c.get("text_n") or 0)
    head = f"`{pair['baseline']}` (production) -> `{pair['candidate']}` (candidate)"
    mark = "❌ blocked" if failures else "✅ passed"
    lines = [
        f"### {pair['family']}: {head}: {mark}",
        "",
        f"| Metric | `{pair['baseline']}` | `{pair['candidate']}` | Change | Gate rule |",
        "|---|---:|---:|---:|---|",
    ]
    for key, label, kind, rule, rule_text in ROWS[pair["family"]]:
        if key not in p and key not in c:
            continue
        if rule is None:
            verdict = "(info)"
        else:
            verdict = ("❌ " if any(f.startswith(rule) for f in failures) else "✅ ") + rule_text
        lines.append(
            f"| {label} | {_fmt(p.get(key), kind, n)} | {_fmt(c.get(key), kind, n)} "
            f"| {_delta(p.get(key), c.get(key), kind)} | {verdict} |"
        )
    if failures:
        lines += ["", "**Why:** " + "; ".join(failures) + "."]
    secs = f"{prod.get('seconds', '?')} s and {cand.get('seconds', '?')} s"
    lines += ["", f"<sub>Scored on this runner: {secs}; every crop re-embedded.</sub>", ""]
    return bool(failures), "\n".join(lines)


def verdict_report(
    plan: dict[str, Any], baseline: dict[str, Any], candidate: dict[str, Any]
) -> tuple[bool, str]:
    sections, blocked = [], False
    for pair in plan["pairs"]:
        b, s = judge(
            pair, baseline["versions"][pair["baseline"]], candidate["versions"][pair["candidate"]]
        )
        blocked |= b
        sections.append(s)
    title = "❌ blocked" if blocked else "✅ passed"
    built = candidate.get("gate_manifest_built", "?")
    intro = (
        f"{MARKER}\n## Model gate: {title}\n\n"
        "Both versions scored on the private gate set (campus ReID test split, cached crops "
        f"of the 4 test videos, 12 witness descriptions; built {built}). Rules: decision 0010.\n"
    )
    return blocked, intro + "\n" + "\n".join(sections)


# ----------------------------------------------------------------------- CLI
def _output(**values: str) -> None:
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with Path(out).open("a") as f:
            for k, v in values.items():
                f.write(f"{k}={v}\n")


def _summary(text: str) -> None:
    out = os.environ.get("GITHUB_STEP_SUMMARY")
    if out:
        with Path(out).open("a") as f:
            f.write(text + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--base", type=Path, required=True, help="base branch's configs/models.yaml")
    p.add_argument("--head", type=Path, default=REPO / "configs" / "models.yaml")
    p.add_argument("--out", type=Path, required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--plan", type=Path, required=True)
    f.add_argument("--models-dir", type=Path, default=REPO / "models")
    f.add_argument("--assets", type=Path, default=REPO / "configs" / "assets.yaml")
    v = sub.add_parser("verdict")
    v.add_argument("--plan", type=Path, required=True)
    v.add_argument("--baseline", type=Path, required=True)
    v.add_argument("--candidate", type=Path, required=True)
    v.add_argument("--report", type=Path, required=True)
    args = ap.parse_args()

    try:
        if args.cmd == "plan":
            plan = make_plan(
                yaml.safe_load(args.base.read_text()), yaml.safe_load(args.head.read_text())
            )
            args.out.write_text(json.dumps(plan, indent=1))
            report = plan_report(plan)
            print(report)
            _summary(report)
            _output(
                needed=str(bool(plan["pairs"])).lower(),
                baseline=" ".join(p["baseline"] for p in plan["pairs"]),
                candidate=" ".join(p["candidate"] for p in plan["pairs"]),
            )
            return 0
        if args.cmd == "fetch":
            plan = json.loads(args.plan.read_text())
            manifest = yaml.safe_load(args.assets.read_text()) or {}
            models = {**plan["baseline_models"], **plan["candidate_models"]}
            # every version's entry, so a file owned by another version (irra-v2 reuses
            # irra-v1's text tower) still has a hash
            head = yaml.safe_load((REPO / "configs" / "models.yaml").read_text())
            assets_repo = os.environ.get("PS_ASSETS__REPO") or manifest.get("repo")
            fetch(
                plan,
                {**head["models"], **models},
                manifest,
                args.models_dir,
                assets_repo,
                os.environ.get("HF_TOKEN", "").strip() or None,
            )
            return 0
        plan = json.loads(args.plan.read_text())
        blocked, report = verdict_report(
            plan, json.loads(args.baseline.read_text()), json.loads(args.candidate.read_text())
        )
        args.report.write_text(report)
        print(report)
        _summary(report)
        _output(blocked=str(blocked).lower())
        return 1 if blocked else 0
    except GateError as e:
        print(f"::error title=model gate::{e}", file=sys.stderr)
        _summary(f"{MARKER}\n## Model gate: ❌ cannot gate this PR\n\n{e}\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())
