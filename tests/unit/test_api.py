"""Contract tests for the HTTP service (Step 2), with fake models.

The routes, the job queue, ``SearchService``, ``IndexedSearch`` and both index
implementations are real; only the encoders, detector and tracker are fakes.
Index fixture: two videos; track "3" of video A is the planted person, close to
the fake photo encoder's vector for a bright photo and to the fake text
encoder's vector for TARGET_TEXT.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable, Iterator, Sequence

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from person_search.api.app import create_app
from person_search.api.service import SearchService, open_index
from person_search.config import ApiSettings, Settings
from person_search.errors import ConfigurationError
from person_search.index import InMemoryIndex, SearchFilter, points_from_ingest
from person_search.index.qdrant import QdrantIndex
from person_search.ingest.pipeline import IngestPipeline, IngestResult
from person_search.types import IRRA, OSNET, BBox, Detection, Observation, TrackState, VideoInfo

DIM = 32
VERSIONS = {OSNET: "osnet-test", IRRA: "irra-test"}
VIDEO_A, VIDEO_B = "a" * 16, "b" * 16
TARGET_TRACK = "3"
TARGET_TEXT = "a woman in a red coat with a black backpack"
TARGET_FRAMES = range(5, 26)


# ---------------------------------------------------------------------- fakes
def _unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


def key_vector(key: str) -> np.ndarray:
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    return _unit(np.random.default_rng(seed).normal(size=DIM))


class FakeEncoder:
    """Images -> a vector per brightness bucket (mean // 16); texts -> a vector per string."""

    def __init__(self, name: str) -> None:
        self.name, self.dim, self.version = name, DIM, VERSIONS[name]

    def encode_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        vecs = [key_vector(f"{self.name}:img:{int(np.mean(im)) // 16}") for im in images]
        return np.stack(vecs) if vecs else np.zeros((0, DIM), np.float32)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        return np.stack([key_vector(f"{self.name}:text:{t}") for t in texts])


class FakeDetector:
    def __init__(self, before: Callable[[], None] | None = None) -> None:
        self.before = before

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self.before is not None:
            self.before()
        return [Detection(4, 4, 40, 40, 0.9)]


class FakeTracker:
    """One confirmed track covering most of the frame."""

    def reset(self) -> None:
        pass

    def update(self, detections: Sequence[Detection], frame: np.ndarray) -> list[TrackState]:
        return [TrackState("1", (4.0, 4.0, 44.0, 44.0))]


BRIGHT = 200  # photo brightness whose OSNet vector the target is planted near
PHOTO_VEC = key_vector(f"{OSNET}:img:{BRIGHT // 16}")
TEXT_VEC = key_vector(f"{IRRA}:text:{TARGET_TEXT}")


def make_ingest(seed: int, plant: bool) -> IngestResult:
    rng = np.random.default_rng(seed)
    obs = []
    for t in range(1, 7):
        tid = str(t)
        target = plant and tid == TARGET_TRACK
        centre = {
            OSNET: PHOTO_VEC if target else _unit(rng.normal(size=DIM)),
            IRRA: TEXT_VEC if target else _unit(rng.normal(size=DIM)),
        }
        frames = TARGET_FRAMES if target else range(1 + t, 21 + t)
        for f in frames:
            o = Observation(tid, f, f - 1, BBox(t, t, t + 20, t + 40))
            if f % 3 == 0:
                for name in (OSNET, IRRA):
                    o.embeddings[name] = _unit(centre[name] + 0.05 * rng.normal(size=DIM))
            obs.append(o)
    return IngestResult(VideoInfo("v.mp4", 12.0, 40, 64, 48), obs, 40, dict(VERSIONS))


def build_service(
    index_kind: str,
    tmp_path,
    *,
    before_detect: Callable[[], None] | None = None,
    **api: object,
) -> SearchService:
    index = (
        InMemoryIndex(VERSIONS)
        if index_kind == "memory"
        else QdrantIndex.in_memory(VERSIONS, {OSNET: DIM, IRRA: DIM})
    )
    index.upsert(points_from_ingest(make_ingest(1, True), VIDEO_A, video_name="a.mp4", camera="A"))
    index.upsert(points_from_ingest(make_ingest(2, False), VIDEO_B, video_name="b.mp4", camera="B"))
    settings = Settings(_env_file=None, api=ApiSettings(upload_dir=tmp_path / "up", **api))  # type: ignore[call-arg]
    osnet, irra = FakeEncoder(OSNET), FakeEncoder(IRRA)

    def new_ingest() -> IngestPipeline:
        return IngestPipeline(
            FakeDetector(before_detect),
            FakeTracker(),
            {OSNET: osnet, IRRA: irra},
            sampling=settings.sampling,
        )

    return SearchService(settings, osnet, irra, index, new_ingest)


@pytest.fixture(params=["memory", "qdrant"])
def service(request, tmp_path) -> SearchService:
    return build_service(request.param, tmp_path)


@pytest.fixture
def client(service) -> Iterator[TestClient]:
    with TestClient(create_app(service)) as c:
        yield c


def jpeg(value: int, size: tuple[int, int] = (64, 32)) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((*size, 3), value, np.uint8))
    assert ok
    return buf.tobytes()


def video_bytes(tmp_path, value: int, n_frames: int = 9) -> bytes:
    path = tmp_path / f"src_{value}_{n_frames}.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 12.0, (64, 48))
    for _ in range(n_frames):
        w.write(np.full((48, 64, 3), value, np.uint8))
    w.release()
    return path.read_bytes()


def wait_job(client: TestClient, job_id: str, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/jobs/{job_id}").json()
        if body["status"] in ("succeeded", "failed"):
            return body
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} still {body['status']} after {timeout}s")


# ------------------------------------------------------------------- contract
def test_openapi_contract(client):
    schema = client.get("/openapi.json").json()
    assert {p: sorted(m) for p, m in schema["paths"].items()} == {
        "/health": ["get"],
        "/videos": ["get"],
        "/search": ["post"],
        "/ingest": ["post"],
        "/jobs/{job_id}": ["get"],
    }
    comps = schema["components"]["schemas"]

    def form(path: str) -> dict:
        ref = schema["paths"][path]["post"]["requestBody"]["content"]["multipart/form-data"]
        return comps[ref["schema"]["$ref"].rsplit("/", 1)[-1]]

    search_form = form("/search")
    assert set(search_form["properties"]) == {
        "photo",
        "description",
        "top_k",
        "alpha",
        "video_ids",
        "cameras",
    }
    assert not search_form.get("required")  # photo, description or both: checked in code
    # What /docs pre-fills. Without examples it fills "string" into the description and
    # the lists, which silently turns a photo search into photo + "string" over no camera.
    props = search_form["properties"]
    assert props["description"]["examples"] == [""]
    assert props["alpha"]["examples"] == [0.5]
    assert props["video_ids"]["examples"] == props["cameras"]["examples"] == [[]]
    ingest_form = form("/ingest")
    assert set(ingest_form["properties"]) == {"video", "camera"}
    assert ingest_form["required"] == ["video"]
    assert schema["paths"]["/ingest"]["post"]["responses"].keys() >= {"202", "413", "429"}
    assert set(comps["SearchResponse"]["required"]) == {
        "mode",
        "model_versions",
        "tracks",
        "timing_ms",
    }
    assert set(comps["JobOut"]["properties"]["status"]["$ref"].split("/")[-1:]) == {"JobStatus"}
    assert comps["JobStatus"]["enum"] == ["queued", "running", "succeeded", "failed"]


def test_health_reports_model_versions_and_index(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["models"] == VERSIONS
    assert body["index"]["videos"] == 2 and body["index"]["points"] > 0
    assert body["jobs"] == {"queued": 0, "running": 0, "succeeded": 0, "failed": 0}
    videos = client.get("/videos").json()
    assert [(v["video_id"], v["camera"]) for v in videos] == [(VIDEO_A, "A"), (VIDEO_B, "B")]
    assert client.get("/", follow_redirects=False).headers["location"] == "/docs"


def test_service_refuses_an_index_built_with_other_models(tmp_path):
    osnet, irra = FakeEncoder(OSNET), FakeEncoder(IRRA)
    index = InMemoryIndex({OSNET: "osnet-v1", IRRA: "irra-test"})
    with pytest.raises(ConfigurationError, match="rebuild the index"):
        SearchService(Settings(_env_file=None), osnet, irra, index, lambda: None)  # type: ignore[call-arg, arg-type, return-value]


def test_open_index_creates_reopens_and_refuses_other_versions(tmp_path):
    folder = tmp_path / "index"
    dims = {OSNET: DIM, IRRA: DIM}
    idx = open_index(folder, VERSIONS, dims)  # empty folder -> new collection
    idx.upsert(points_from_ingest(make_ingest(1, True), VIDEO_A, video_name="a.mp4"))
    n = idx.count()
    idx.close()  # type: ignore[attr-defined]
    idx = open_index(folder, VERSIONS, dims)  # same versions -> same points
    assert idx.count() == n
    idx.close()  # type: ignore[attr-defined]
    with pytest.raises(ConfigurationError, match="rebuild the index"):
        open_index(folder, {OSNET: "osnet-test", IRRA: "irra-v9"}, dims)


# --------------------------------------------------------------------- search
def test_photo_search_matches_the_planted_track(client, service):
    r = client.post("/search", files={"photo": ("q.jpg", jpeg(BRIGHT), "image/jpeg")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "photo" and body["matched"] is True
    assert body["model_versions"] == VERSIONS
    top = body["videos"][0]
    assert top["video_id"] == VIDEO_A and top["camera"] == "A"
    assert top["accepted_tracks"] == [TARGET_TRACK]
    assert top["selected_frames"] == len(TARGET_FRAMES)
    assert [(s["start_frame"], s["end_frame"]) for s in top["segments"]] == [(5, 25)]
    assert body["tracks"][0]["track_id"] == TARGET_TRACK and body["tracks"][0]["accepted"]
    assert not any(t["accepted"] for t in body["tracks"][1:])
    assert {"encode_photo", "retrieve", "fetch", "score", "total"} <= body["timing_ms"].keys()
    # the API answers exactly what IndexedSearch answers
    direct = service.searcher.photo(PHOTO_VEC)
    assert {v["video_id"]: v["accepted_tracks"] for v in body["videos"]} == {
        vid: list(res.decision.accepted) for vid, res in direct.per_video.items()
    }


def test_photo_search_no_match_still_lists_the_closest_tracks(client):
    body = client.post("/search", files={"photo": ("q.jpg", jpeg(10), "image/jpeg")}).json()
    assert body["matched"] is False
    assert body["tracks"] and not any(t["accepted"] for t in body["tracks"])
    assert all(not v["matched"] and v["segments"] == [] for v in body["videos"])


def test_text_search_ranks_the_planted_track_first(client):
    r = client.post("/search", data={"description": TARGET_TEXT, "top_k": "3"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["mode"] == "text" and body["matched"] is None and body["videos"] == []
    assert len(body["tracks"]) == 3
    first = body["tracks"][0]
    assert (first["video_id"], first["track_id"]) == (VIDEO_A, TARGET_TRACK)
    assert first["accepted"] is None
    scores = [t["score"] for t in body["tracks"]]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_search_uses_photo_and_description(client):
    files = {"photo": ("q.jpg", jpeg(BRIGHT), "image/jpeg")}
    body = client.post(
        "/search", files=files, data={"description": TARGET_TEXT, "alpha": "0.7"}
    ).json()
    assert body["mode"] == "hybrid" and body["alpha"] == 0.7
    assert body["tracks"][0]["track_id"] == TARGET_TRACK
    assert {"encode_photo", "encode_text"} <= body["timing_ms"].keys()
    default = client.post("/search", files=files, data={"description": TARGET_TEXT}).json()
    assert default["alpha"] == 0.5  # settings.fusion.alpha


def test_filters_restrict_the_search(client):
    data = {"description": TARGET_TEXT, "top_k": "10"}
    only_b = client.post("/search", data={**data, "video_ids": VIDEO_B}).json()
    assert only_b["tracks"] and {t["video_id"] for t in only_b["tracks"]} == {VIDEO_B}
    by_camera = client.post("/search", data={**data, "cameras": "B"}).json()
    assert {t["camera"] for t in by_camera["tracks"]} == {"B"}
    both = client.post("/search", data={**data, "cameras": ["A", "B"]}).json()
    assert {t["camera"] for t in both["tracks"]} == {"A", "B"}
    photo = client.post(
        "/search",
        files={"photo": ("q.jpg", jpeg(BRIGHT), "image/jpeg")},
        data={"video_ids": VIDEO_B},
    ).json()
    assert photo["matched"] is False and {v["video_id"] for v in photo["videos"]} <= {VIDEO_B}


@pytest.mark.parametrize(
    ("data", "names"),
    [
        ({"cameras": "string"}, ["'string'", "A, B", "pre-fills"]),  # the /docs placeholder
        ({"cameras": "B,C"}, ["unknown cameras ['C']", "A, B"]),
        ({"video_ids": [VIDEO_B, "nope"]}, ["unknown video_ids ['nope']", VIDEO_A, "GET /videos"]),
        ({"video_ids": "nope", "cameras": "Z"}, ["video_ids ['nope']", "cameras ['Z']"]),
    ],
)
def test_unknown_filter_values_are_a_422_naming_the_valid_ones(client, data, names):
    """A typo in a filter must not look like "nobody found" (empty 200)."""
    r = client.post("/search", data={"description": TARGET_TEXT, **data})
    assert r.status_code == 422, r.text
    detail = r.json()["detail"]
    for name in names:
        assert name in detail, detail


def test_search_flt_equals_the_service_filter(service):
    a = service.search(description=TARGET_TEXT, top_k=10, flt=SearchFilter(cameras=("A",)))
    assert {t.video_id for t in a.tracks} == {VIDEO_A}


@pytest.mark.parametrize(
    ("data", "files", "status"),
    [
        ({}, None, 422),  # neither photo nor description
        ({"description": "   "}, None, 422),  # blank description counts as none
        ({"description": "x" * 501}, None, 422),
        ({"description": "red", "top_k": "0"}, None, 422),
        ({"description": "red", "top_k": "51"}, None, 422),
        ({"description": "red", "alpha": "1.5"}, None, 422),
        ({}, {"photo": ("q.jpg", b"not an image", "image/jpeg")}, 400),
        ({}, {"photo": ("q.jpg", b"", "image/jpeg")}, 422),  # empty file field = no photo
    ],
)
def test_bad_search_requests(client, data, files, status):
    r = client.post("/search", data=data, files=files)
    assert r.status_code == status, r.text
    assert "detail" in r.json()


def test_empty_photo_field_with_a_description_is_a_text_query(client):
    r = client.post(
        "/search",
        files={"photo": ("", b"", "application/octet-stream")},
        data={"description": TARGET_TEXT},
    )
    assert r.status_code == 200 and r.json()["mode"] == "text"


def test_photo_size_limit(tmp_path):
    svc = build_service("memory", tmp_path, max_photo_mb=0.001)
    with TestClient(create_app(svc)) as c:
        big = c.post("/search", files={"photo": ("q.jpg", b"\xff" * 4096, "image/jpeg")})
        assert big.status_code == 413


# --------------------------------------------------------------------- ingest
def test_ingest_returns_a_job_and_the_video_becomes_searchable(client, tmp_path):
    before = client.get("/health").json()["index"]
    data = video_bytes(tmp_path, BRIGHT)
    r = client.post(
        "/ingest", files={"video": ("cam7.avi", data, "video/x-msvideo")}, data={"camera": "gate 7"}
    )
    assert r.status_code == 202, r.text
    job = r.json()
    assert r.headers["location"] == f"/jobs/{job['job_id']}"
    assert job["status"] in ("queued", "running", "succeeded")
    assert job["video"] == "cam7.avi" and job["camera"] == "gate 7"
    done = wait_job(client, job["job_id"])
    assert done["status"] == "succeeded", done
    assert done["progress"] == 1.0
    res = done["result"]
    assert res["video_id"] == job["video_id"] and res["skipped"] is False
    assert res["samples"] == 9 and res["tracks"] == 1 and res["points"] == 9
    assert res["embedded_points"] == 3

    after = client.get("/health").json()
    assert after["index"]["videos"] == before["videos"] + 1
    assert after["index"]["points"] == before["points"] + 9
    assert after["jobs"]["succeeded"] == 1
    # the new video is searchable; its only track looks exactly like the bright photo
    found = client.post(
        "/search",
        files={"photo": ("q.jpg", jpeg(BRIGHT), "image/jpeg")},
        data={"cameras": "gate 7"},
    ).json()
    assert found["matched"] and found["videos"][0]["video_id"] == job["video_id"]
    assert found["videos"][0]["video"] == "cam7.avi"

    # the same video again: no new work, no new points
    again = client.post("/ingest", files={"video": ("copy.avi", data, "video/x-msvideo")})
    assert again.status_code == 202
    assert again.json()["status"] == "succeeded" and again.json()["result"]["skipped"] is True
    assert client.get("/health").json()["index"]["points"] == after["index"]["points"]
    assert not list((tmp_path / "up").iterdir())  # uploads are deleted once handled


def test_ingest_queue_returns_at_once_and_refuses_when_full(tmp_path):
    gate = threading.Event()
    svc = build_service("qdrant", tmp_path, before_detect=lambda: gate.wait(10), max_pending_jobs=2)
    with TestClient(create_app(svc)) as c:

        def upload(value: int):  # one distinct video per brightness value
            files = {"video": (f"{value}.avi", video_bytes(tmp_path, value), "video/avi")}
            return c.post("/ingest", files=files)

        ids = []
        for value in (60, 90):
            t0 = time.monotonic()
            r = upload(value)
            assert r.status_code == 202 and time.monotonic() - t0 < 5  # the worker is blocked
            assert r.json()["status"] in ("queued", "running")
            ids.append(r.json()["job_id"])
        # re-uploading a queued video returns its job instead of queueing it twice
        dup = upload(90)
        assert dup.status_code == 202 and dup.json()["job_id"] == ids[1]
        full = upload(120)
        assert full.status_code == 429 and full.headers["retry-after"] == "60"
        jobs = c.get("/health").json()["jobs"]
        assert jobs["running"] + jobs["queued"] == 2
        gate.set()
        assert [wait_job(c, i)["status"] for i in ids] == ["succeeded", "succeeded"]
        assert c.get("/health").json()["index"]["videos"] == 4


def test_failed_job_reports_its_error(tmp_path):
    def boom() -> None:
        raise RuntimeError("detector exploded")

    svc = build_service("memory", tmp_path, before_detect=boom)
    with TestClient(create_app(svc)) as c:
        r = c.post("/ingest", files={"video": ("v.avi", video_bytes(tmp_path, 50), "video/avi")})
        done = wait_job(c, r.json()["job_id"])
        assert done["status"] == "failed"
        assert done["error"] == "RuntimeError: detector exploded"
        assert c.get("/health").json()["index"]["videos"] == 2
        assert c.get("/health").json()["status"] == "ok"  # the worker survives a failed job


@pytest.mark.parametrize(
    ("name", "payload", "data", "status"),
    [
        ("v.txt", b"hello", {}, 415),
        ("v", b"hello", {}, 415),
        ("v.mp4", b"not a video at all", {}, 400),
        ("v.mp4", b"", {}, 400),
        ("v.avi", None, {"camera": "../../etc"}, 422),
    ],
)
def test_bad_ingest_requests(client, tmp_path, name, payload, data, status):
    body = video_bytes(tmp_path, 30) if payload is None else payload
    r = client.post("/ingest", files={"video": (name, body, "video/mp4")}, data=data)
    assert r.status_code == status, r.text
    assert client.get("/health").json()["jobs"]["queued"] == 0


def test_ingest_size_limits(tmp_path):
    svc = build_service("memory", tmp_path, max_video_mb=0.001)
    with TestClient(create_app(svc)) as c:
        # over the limit but under the Content-Length guard: the route counts the bytes
        r = c.post("/ingest", files={"video": ("v.avi", video_bytes(tmp_path, 30), "video/avi")})
        assert r.status_code == 413
        # Content-Length above limit + 1 MB: refused before the body is read
        r = c.post("/ingest", files={"video": ("v.avi", b"\0" * (3 << 20), "video/avi")})
        assert r.status_code == 413 and "larger than" in r.json()["detail"]


def test_unknown_job_is_404(client):
    assert client.get("/jobs/does-not-exist").status_code == 404
