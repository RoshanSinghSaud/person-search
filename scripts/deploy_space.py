"""Step 4: deploy the service to a Hugging Face Space and check that it answers.

CI runs this on every push to main, after the tests and the image build pass
(.github/workflows/ci.yml, job ``deploy``); ``make deploy-space`` runs it by hand.
Decision 0015.

1. Creates the Space ``<hf-user>/person-search`` (public, Docker SDK) if missing.
2. Gives the container what it reads at start: the variable PS_ASSETS__REPO and the
   secrets HF_TOKEN (read access to the private assets repo) and PS_API__API_KEY
   (uploads). Every change restarts the Space, so a variable is written only when its
   value changes and a secret only when it is missing (``--sync-secrets`` rewrites them,
   e.g. after a token rotation).
3. Uploads the files the Docker build needs (git-tracked only) as one commit, with the
   Space's settings prepended to README.md; files from earlier deploys that the build
   no longer needs are deleted.
4. Waits until the Space runs that commit, then checks over HTTPS that ``/health`` reports
   both models and a non-empty index, that a text search answers, and that an upload
   without the API key is refused. A failed build or start prints the Space's logs.

Environment (CI: repository secrets and variables)::

    HF_TOKEN          write token of the Space's owner (secret HF_TOKEN)
    HF_SPACE_TOKEN    read token the Space uses for the private assets repo
    PS_API_KEY        key for POST /ingest on the public Space (>= 16 characters)
    HF_SPACE          optional, default <token owner>/person-search
    PS_ASSETS__REPO   optional, default <token owner>/person-search-assets

Exit codes: 0 deployed and healthy, 1 the Space failed or never became healthy,
2 a setting is missing (nothing was changed).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

# What the Docker build reads (Dockerfile, .dockerignore) plus the notices that ship
# with the code. A public Space's files are the corresponding source (AGPL section 13).
BUILD_FILES = (
    "Dockerfile",
    ".dockerignore",
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "src/",
    "configs/",
)
APP_PORT = 8000  # the Dockerfile's PORT; HF routes https://<subdomain>.hf.space to it
FAILED = {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE", "PAUSED", "STOPPED"}
IN_PROGRESS = {"BUILDING", "RUNNING_BUILDING", "APP_STARTING", "RUNNING_APP_STARTING"}
SMOKE_TEXT = "a person wearing a dark jacket"

SPACE_HEADER = f"""\
---
title: Person Search
emoji: 🔎
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: {APP_PORT}
startup_duration_timeout: 1h
short_description: Find a person in video by photo, description or both
pinned: false
---
"""

SPACE_BODY = """\
# Person Search

Find a person in indexed CCTV video **by photo** (OSNet, ICCVBIC 2026 paper),
**by description** (IRRA) or **by both**. This Space runs the service's Docker
image on a free CPU: open **/docs** on the app's URL to try `POST /search`.

- The index holds the 4 test videos of the paper (vectors only: no footage is on
  this server, so results come without clips).
- `POST /ingest` needs an API key (`X-API-Key`).
- No persistent storage: a restart downloads the weights again (about a minute)
  and forgets videos ingested since the last deploy.

Deployed by CI from {source} (commit `{sha}`); every push to `main` redeploys.
"""


class SetupError(Exception):
    """A missing or invalid setting: exit 2, before anything changes."""


# ------------------------------------------------------------------ pure helpers
def build_paths(tracked: list[str]) -> list[str]:
    """The git-tracked files the Space needs, in a stable order."""
    keep = [
        p
        for p in tracked
        if any(p == f or (f.endswith("/") and p.startswith(f)) for f in BUILD_FILES)
    ]
    missing = [f for f in BUILD_FILES if not any(p == f or p.startswith(f) for p in keep)]
    if missing:
        raise SetupError(f"not tracked by git: {missing}")
    return sorted(keep)


def space_readme(source: str, sha: str) -> str:
    return SPACE_HEADER + "\n" + SPACE_BODY.format(source=source, sha=sha[:7])


def check_health(body: dict[str, Any]) -> list[str]:
    """What is wrong with a /health answer (empty = healthy)."""
    problems = []
    if body.get("status") != "ok":
        problems.append(f"status {body.get('status')!r}")
    models = body.get("models") or {}
    for name in ("osnet", "irra"):
        if not models.get(name):
            problems.append(f"no {name} model version")
    if not (body.get("index") or {}).get("points"):
        problems.append("empty index")
    return problems


def secrets_to_write(wanted: dict[str, str], existing: set[str], sync: bool) -> dict[str, str]:
    return {k: v for k, v in wanted.items() if sync or k not in existing}


def variables_to_write(wanted: dict[str, str], existing: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in wanted.items() if existing.get(k) != v}


# ------------------------------------------------------------------------ HTTP
def http(
    url: str, *, data: dict[str, str] | None = None, timeout: float = 30
) -> tuple[int, dict[str, Any]]:
    body = urllib.parse.urlencode(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except ValueError:
            return e.code, {}


def smoke(base: str, wait_s: float) -> dict[str, Any]:
    """/health until healthy (the proxy can lag the runtime), then a search and an upload."""
    deadline = time.monotonic() + wait_s
    while True:
        try:
            status, body = http(f"{base}/health")
            problems = [f"HTTP {status}"] if status != 200 else check_health(body)
        except (OSError, ValueError) as e:  # connection reset, HTML error page, ...
            problems, body = [f"{type(e).__name__}: {e}"], {}
        if not problems:
            break
        if time.monotonic() > deadline:
            raise RuntimeError(f"/health not healthy: {'; '.join(problems)}")
        print(f"  /health: {'; '.join(problems)}; retrying")
        time.sleep(10)
    print(f"  /health: ok, models {body['models']}, index {body['index']}")

    t0 = time.perf_counter()
    status, found = http(f"{base}/search", data={"description": SMOKE_TEXT, "top_k": "3"})
    if status != 200 or found.get("mode") != "text" or not found.get("tracks"):
        raise RuntimeError(f"text search: HTTP {status} {str(found)[:300]}")
    print(f"  /search (text): {len(found['tracks'])} tracks in {time.perf_counter() - t0:.2f} s")

    status, _ = http(f"{base}/ingest", data={})
    if status != 401:
        raise RuntimeError(f"/ingest without a key answered {status}, expected 401")
    print("  /ingest without the API key: 401")
    return body


# ------------------------------------------------------------------------- run
def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def print_logs(api: HfApi, space: str, build: bool, last: int = 150) -> None:
    try:
        lines = list(api.fetch_space_logs(space, build=build))
    except Exception as e:  # logs are a courtesy; the failure is reported anyway
        print(f"(could not fetch {'build' if build else 'run'} logs: {e})")
        return
    kind = "build" if build else "container"
    print(f"::group::last {min(last, len(lines))} lines of the {kind} log")
    for line in lines[-last:]:
        print(line.rstrip())
    print("::endgroup::")


def wait_for_commit(api: HfApi, space: str, sha: str, timeout_s: float) -> str:
    """Poll until the Space runs ``sha``; return its stage when it stops changing."""
    start = time.monotonic()
    seen_progress, last = False, None
    while time.monotonic() - start < timeout_s:
        raw = api.get_space_runtime(space).raw
        stage, running = str(raw.get("stage")), raw.get("sha")
        if (stage, running) != last:
            short = running[:7] if running else "?"
            print(f"  {time.monotonic() - start:5.0f} s  {stage}  (commit {short})")
            last = (stage, running)
        seen_progress |= stage in IN_PROGRESS
        if running == sha and stage == "RUNNING":
            return stage
        if stage in FAILED and (running == sha or seen_progress):
            return stage
        time.sleep(15)
    return "TIMEOUT"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--sync-secrets", action="store_true", help="rewrite the Space's secrets")
    ap.add_argument("--wait-minutes", type=float, default=45, help="build + start budget")
    ap.add_argument("--dry-run", action="store_true", help="stage the files, upload nothing")
    args = ap.parse_args()

    try:
        token = os.environ.get("HF_TOKEN") or None
        if not token and not args.dry_run:
            raise SetupError("HF_TOKEN (write token) is not set")
        tracked = git("ls-files").splitlines()
        paths = build_paths(tracked)
        dirty = git("status", "--porcelain", "--", *BUILD_FILES).strip()
        if dirty and not args.dry_run:
            raise SetupError(f"uncommitted changes in the build files:\n{dirty}")
        sha = git("rev-parse", "HEAD").strip()
        server, repo = os.environ.get("GITHUB_SERVER_URL"), os.environ.get("GITHUB_REPOSITORY")
        source = f"{server}/{repo}" if server and repo else "the person-search repository"

        api = HfApi(token=token)
        owner = api.whoami()["name"] if token else "<hf-user>"
        space = os.environ.get("HF_SPACE") or f"{owner}/person-search"
        assets_repo = os.environ.get("PS_ASSETS__REPO") or f"{owner}/person-search-assets"
        wanted_secrets = {
            "HF_TOKEN": os.environ.get("HF_SPACE_TOKEN", ""),
            "PS_API__API_KEY": os.environ.get("PS_API_KEY", ""),
        }
        if len(wanted_secrets["PS_API__API_KEY"]) < 16 and not args.dry_run:
            raise SetupError("PS_API_KEY must be set, >= 16 characters (openssl rand -hex 24)")
    except SetupError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    staging = Path(tempfile.mkdtemp(prefix="space-"))
    for p in paths:
        dst = staging / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    (staging / "README.md").write_text(space_readme(source, sha), encoding="utf-8")
    size = sum(f.stat().st_size for f in staging.rglob("*") if f.is_file())
    print(f"{len(paths)} files, {size / 1e6:.1f} MB, commit {sha[:7]} -> Space {space}")
    if args.dry_run:
        print(f"dry run: staged in {staging}")
        return 0

    # 1-2. the Space and its settings (only what changed: every change restarts it)
    url = api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
    print(f"Space: {url}")
    existing_secrets = set(api.get_space_secrets(space))
    if not wanted_secrets["HF_TOKEN"] and "HF_TOKEN" not in existing_secrets:
        print(
            "ERROR: HF_SPACE_TOKEN is not set and the Space has no HF_TOKEN secret: "
            f"it could not download the private assets from {assets_repo}",
            file=sys.stderr,
        )
        return 2
    wanted_secrets = {k: v for k, v in wanted_secrets.items() if v}
    for key, value in secrets_to_write(wanted_secrets, existing_secrets, args.sync_secrets).items():
        api.add_space_secret(space, key, value)
        print(f"  secret {key}: written")
    current = {k: v.value for k, v in api.get_space_variables(space).items()}
    for key, value in variables_to_write({"PS_ASSETS__REPO": assets_repo}, current).items():
        api.add_space_variable(space, key, value)
        print(f"  variable {key}={value}")

    # 3. one commit with exactly the build files
    info = api.upload_folder(
        repo_id=space,
        repo_type="space",
        folder_path=staging,
        commit_message=f"Deploy {sha[:7]}",
        commit_description=f"From {source}/commit/{sha}" if repo else None,
        delete_patterns="*",  # files earlier deploys needed and this one does not
    )
    commit = info.oid
    print(f"uploaded: {info.commit_url}")

    # 4. the build, the start, then the service itself
    stage = wait_for_commit(api, space, commit, args.wait_minutes * 60)
    if stage != "RUNNING":
        print(f"ERROR: the Space ended in {stage}", file=sys.stderr)
        print_logs(api, space, build=True)
        print_logs(api, space, build=False)
        return 1
    host = api.space_info(space).host or f"https://{space.replace('/', '-').lower()}.hf.space"
    try:
        smoke(host, wait_s=300)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print_logs(api, space, build=False)
        return 1
    print(f"live: {host}/docs")
    if out := os.environ.get("GITHUB_OUTPUT"):
        with Path(out).open("a") as f:
            f.write(f"url={host}/docs\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
