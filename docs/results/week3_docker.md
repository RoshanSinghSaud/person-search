# Step 3: the Docker image, built and run

Machine: Linux x86_64, 2 vCPU, 7.8 GB RAM, Docker 29.9 (BuildKit). Decision 0013.
The sandbox reaches the internet through an egress proxy, so for these runs the
build added its CA certificate and used `apt` over https; the Dockerfile in the
repo is otherwise exactly what was built.

## Image

| | |
|---|---|
| Size | 603 MB compressed, 2.66 GB unpacked (venv 1.6 GB, of which torch 684 MB) |
| torch | `2.5.0+cpu`, `torch.version.cuda = None`, no NVIDIA or triton packages |
| Build, empty cache | 2 min 40 s; with the dependency layer cached: 1 min |
| linux/arm64 (Apple Silicon) | builds from the same lockfile; run under QEMU emulation, it gave the same answers as amd64 (text tc1-short track 3 at 0.1588, tc1-medium track 8 at 0.1957; case 4 photo tracks 7, 42, 62 frames), slowly (36-44 s per query emulated; native arm64 is fast) |
| User | uid 1000, data in `/data`, port 8000 |

## Starts

| Start | Time to `/health` | What happened |
|---|---:|---|
| Cold, empty `/data` volume | 37 s | 5 files (1.1 GB) downloaded, size + SHA-256 checked, moved into place in 29 s; models loaded in 6 s |
| Warm restart, same volume | 9 s | "all assets present (4)", no request to the Hub |
| `./models` and the index mounted (`make docker-run-local`) | 12 s | nothing to download |
| No `HF_TOKEN` for the private repo | exits (code 2) | one log line: 401 on `weights/yolo11n_person.pt`; `/data` left empty |
| No `PS_ASSETS__REPO` and no files | exits (code 2) | lists each missing file and the setting to fix |

For the cold start, the private files (YOLO, OSNet, the Week 2 index) came from
a local stand-in for the Hub's file endpoint (`HF_ENDPOINT`), which answers 401
without the token like a private repo; the download code is the real
`huggingface_hub` path. The IRRA weights were also fetched from the real
huggingface.co inside the container, at the pinned commit: 1.04 GB in 9.8 s,
SHA-256 equal to `configs/models.yaml`.

## Same answers inside the container

`scripts/smoke_api.py` against the container (each case searched in its own video):

| Check | Result |
|---|---|
| Photo: accepted tracks equal Week 2 | 4/4 |
| Text: top track and score equal Week 2 | 12/12 |
| Photo + text (alpha 0.5): top track is a photo-accepted track | 12/12 |
| p50 latency photo / text / both | 264 / 276 / 956 ms (natively: 260 / 275 / 932) |
| `/ingest` of the 10 s clip | done in 66 s, target found; still indexed after a restart (5 videos, 17,009 points) |

## Memory

2.3 GB idle after loading YOLO, DeepSORT, OSNet and IRRA (FP32); 2.8 GB after
an ingest. A free Hugging Face CPU Space (16 GB) has room.
