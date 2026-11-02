# Person Search - developer entry points. Run `make help`.
SHELL := /bin/bash
.DEFAULT_GOAL := help

LEGACY_DIR ?= ../legacy
CASES_DIR  ?= $(LEGACY_DIR)/amrit-pbvs
# e.g. linux-x86_64, darwin-arm64 (same tag as the tests compute from Python)
PLATFORM   := $(shell uname -s | tr '[:upper:]' '[:lower:]')-$(shell uname -m)
PY         ?= uv run python
PYTEST     ?= uv run pytest

.PHONY: help
help: ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

.PHONY: setup
setup: ## create the env (CPU torch) with dev tools, install pre-commit hooks
	uv sync --group dev
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

.PHONY: clean
clean: ## remove caches and outputs
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov outputs dist build
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
