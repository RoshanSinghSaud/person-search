# Person Search - developer entry points. Run `make help`.
SHELL := /bin/bash
.DEFAULT_GOAL := help

LEGACY_DIR ?= ../legacy
CASES_DIR  ?= $(LEGACY_DIR)/amrit-pbvs
CASES      := test_case_001 test_case_002 test_case_003 test_case_004
INGEST_DIR ?= $(LEGACY_DIR)/runs/ingest_v1
WEEK2_DIR  ?= $(LEGACY_DIR)/runs/week2
INDEX_DIR  ?= $(WEEK2_DIR)/index
GATE_DIR   ?= $(LEGACY_DIR)/gate_set
CAMPUS_DIR ?= ../custom_reid_dataset
# e.g. linux-x86_64, darwin-arm64 (same tag as the tests compute from Python)
PLATFORM   := $(shell uname -s | tr '[:upper:]' '[:lower:]')-$(shell uname -m)

# Every target imports the package straight from src/, so nothing here depends
# on the editable-install .pth file (macOS can flag it hidden; Python then skips
# it). Outside make, `make fix-pth` repairs the .pth for `uv run person-search`.
export PYTHONPATH := $(CURDIR)/src$(if $(PYTHONPATH),:$(PYTHONPATH))
PY         ?= uv run python
PYTEST     ?= uv run pytest

.PHONY: help
help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

.PHONY: setup
setup: ## create the env (CPU torch) with dev + eval + API deps, install pre-commit hooks
	uv sync --group dev --group eval --group serve
	@$(MAKE) --no-print-directory fix-pth
	uv run --no-sync python -c "import person_search; print('person_search importable from', person_search.__file__)"
	uv run --no-sync pre-commit install

.PHONY: fix-pth
fix-pth: ## macOS: un-hide .pth files in .venv (Python skips hidden .pth files)
	@if [ "$$(uname)" = "Darwin" ]; then \
		chflags nohidden .venv/lib/python*/site-packages/*.pth && echo "cleared hidden flag on .venv .pth files"; \
	fi

.PHONY: legacy
legacy: ## clone both legacy repos into $(LEGACY_DIR) (read-only reference)
	bash scripts/bootstrap_legacy.sh $(LEGACY_DIR)

.PHONY: weights
weights: ## download YOLO/OSNet (team Drive) and IRRA (Hugging Face) into models/
	$(PY) scripts/fetch_weights.py --out models

.PHONY: lint
lint: ## ruff lint + format check
	uv run ruff check src tests tools scripts
	uv run ruff format --check src tests tools scripts

.PHONY: fmt
fmt: ## auto-fix lint and format
	uv run ruff check --fix src tests tools scripts
	uv run ruff format src tests tools scripts

.PHONY: typecheck
typecheck: ## mypy on the package
	uv run mypy

.PHONY: test
test: ## unit tests (no weights needed; what CI runs)
	$(PYTEST) tests/unit --cov --cov-report=term-missing:skip-covered

.PHONY: parity
parity: ## OSNet/IRRA vs legacy code on 20 crops + 5 descriptions (needs legacy + weights)
	PS_LEGACY_ROOT=$(LEGACY_DIR) $(PYTEST) tests/parity -m parity -v

.PHONY: golden
golden: ## pipeline.run == repo 2 frames on this platform's golden data (~15-20 min CPU)
	PS_LEGACY_ROOT=$(LEGACY_DIR) PS_CASES_DIR=$(CASES_DIR) $(PYTEST) tests/golden -m golden -v

.PHONY: reproduce
reproduce: ## Day 2: run repo 2 untouched on the 4 test cases, on this machine
	$(PY) tools/legacy/reproduce_repo2.py --legacy-root $(CASES_DIR) \
		--yolo models/yolo11n_person.pt --osnet models/osnet_campus_v1.pth.tar \
		--out $(LEGACY_DIR)/runs/repo2_$(PLATFORM)

.PHONY: golden-baseline
golden-baseline: reproduce ## record golden data for this platform from repo 2 untouched (~15-20 min)
	$(PY) tools/legacy/make_golden.py --run $(LEGACY_DIR)/runs/repo2_$(PLATFORM) \
		--out tests/golden/data/$(PLATFORM)

.PHONY: fingerprint
fingerprint: ## compare decode / YOLO / OSNet outputs on this machine with the linux-x86_64 reference
	$(PY) tools/diagnostics/fingerprint.py --cases-dir $(CASES_DIR) \
		--out $(LEGACY_DIR)/runs/fingerprint-$(PLATFORM).json > /dev/null
	$(PY) tools/diagnostics/fingerprint.py --compare tests/golden/data/linux-x86_64/fingerprint.json \
		$(LEGACY_DIR)/runs/fingerprint-$(PLATFORM).json

.PHONY: eval-frames
eval-frames: ## frame-level metrics from the committed evaluation.json labels (89.05% F1)
	uv run person-search eval-frames --cases-dir tests/golden/data/labels

# ---------------------------------------------------------------- Week 2
.PHONY: ingest-cases
ingest-cases: ## ingest the 4 test videos with OSNet + IRRA vectors (~25 min on 2 CPUs)
	@mkdir -p $(INGEST_DIR)
	for c in $(CASES); do \
		uv run person-search --log-level WARNING ingest --video $(CASES_DIR)/$$c/test_video.mp4 \
			--out $(INGEST_DIR)/$$c.npz --encoders osnet irra > $(INGEST_DIR)/$$c.json || exit 1; \
	done

.PHONY: index
index: ## Day 10: write the ingested cases into the local Qdrant index (re-run adds 0 points)
	$(PY) scripts/build_index.py --ingest-dir $(INGEST_DIR) --cases-dir $(CASES_DIR) \
		--index $(INDEX_DIR) --out $(WEEK2_DIR)/index.json

.PHONY: clipreid-queries
clipreid-queries: ## Day 12: repo 1's CLIP-ReID embedding of each query photo (for its hybrid)
	$(PY) tools/legacy/clipreid_query_embeddings.py --repo1 $(LEGACY_DIR)/person-reid-clip \
		--cases-dir $(CASES_DIR) --ckpt models/clipreid_image_legacy.pth \
		--out $(WEEK2_DIR)/clipreid_query_embeddings.npz

.PHONY: eval-search
eval-search: ## Days 11-12: indexed photo, text and hybrid search on the 4 cases
	$(PY) scripts/eval_search.py --ingest-dir $(INGEST_DIR) --cases-dir $(CASES_DIR) \
		--index $(INDEX_DIR) --clipreid $(WEEK2_DIR)/clipreid_query_embeddings.npz --out-dir $(WEEK2_DIR)

.PHONY: gate-set
gate-set: ## Day 8: build the gate set (campus split, cached crops, text set) in ../legacy/gate_set
	$(PY) scripts/gate_set.py build --ingest-dir $(INGEST_DIR) --cases-dir $(CASES_DIR) \
		--campus $(CAMPUS_DIR) --out $(GATE_DIR)

.PHONY: eval-models
eval-models: ## Day 8: score osnet v0/v1 and irra v0/v1 on the gate set (~40 min on 2 CPUs)
	$(PY) scripts/eval_models.py evaluate --gate-dir $(GATE_DIR) --out $(WEEK2_DIR)/model_eval.json

.PHONY: register
register: ## Day 8: record the scored versions in MLflow (MLFLOW_TRACKING_URI, else local SQLite)
	$(PY) scripts/eval_models.py register --results $(WEEK2_DIR)/model_eval.json

.PHONY: profile
profile: ## Day 13: time per ingest stage on test case 2, batched vs one crop per pass
	$(PY) scripts/profile_ingest.py --cases-dir $(CASES_DIR) --case test_case_002 \
		--out $(WEEK2_DIR)/profile_case2_batched.json
	$(PY) scripts/profile_ingest.py --cases-dir $(CASES_DIR) --case test_case_002 --max-samples 240 \
		--batch-size 1 --out $(WEEK2_DIR)/profile_case2_b1_240.json
	$(PY) scripts/profile_ingest.py --cases-dir $(CASES_DIR) --case test_case_002 --max-samples 240 \
		--out $(WEEK2_DIR)/profile_case2_batched_240.json

# ---------------------------------------------------------------- Week 3
.PHONY: onnx
onnx: ## export IRRA's image tower to ONNX FP32 and INT8 (models/irra_v*_image_*.onnx)
	$(PY) scripts/export_onnx.py --weights models/irra_v1.pth --out-dir models

.PHONY: benchmark
benchmark: ## PyTorch FP32 vs ONNX FP32 vs ONNX INT8 speed of IRRA's image tower
	$(PY) scripts/benchmark_irra.py --gate-dir $(GATE_DIR) --out $(LEGACY_DIR)/runs/week3/benchmark_irra.json

.PHONY: gate-compare
gate-compare: ## gate verdict for CANDIDATE=<version> against production (exit 1 = blocked)
	$(PY) scripts/eval_models.py compare --results $(WEEK2_DIR)/model_eval.json \
		$(wildcard $(LEGACY_DIR)/runs/week3/model_eval_*.json) --candidate $(CANDIDATE)

# ---------------------------------------------------------------- Step 2: API
API_INDEX ?= data/index
PORT      ?= 8000

.PHONY: api-index
api-index: ## copy the Week 2 index (the 4 test videos) to $(API_INDEX), for the API to serve and extend
	@test ! -e $(API_INDEX) || { echo "$(API_INDEX) already exists"; exit 1; }
	mkdir -p $(dir $(API_INDEX)) && cp -R $(INDEX_DIR) $(API_INDEX)

.PHONY: serve
serve: ## run the API on :$(PORT); models load first (~15 s), docs at /docs
	PS_API__INDEX_PATH=$(API_INDEX) uv run --group serve uvicorn person_search.api.app:create_app \
		--factory --host 0.0.0.0 --port $(PORT)

.PHONY: smoke-api
smoke-api: ## check a running API (make serve) over HTTP against Week 2's numbers (CLIP=video to ingest)
	$(PY) scripts/smoke_api.py --cases-dir $(CASES_DIR) $(if $(CLIP),--ingest $(CLIP)) \
		--out $(LEGACY_DIR)/runs/week3/api_smoke.json

.PHONY: clean
clean: ## remove caches and outputs
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov outputs dist build
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
