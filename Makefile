# WISE (P07) — developer entry points
.DEFAULT_GOAL := help
PY ?= python

.PHONY: help install test lint fmt \
        fiber phase methods physical \
        reproduce reproduce-fast paper clean

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Editable install with dev extras
	$(PY) -m pip install -e ".[dev,viz]"

test: ## Run the test suite
	$(PY) -m pytest

lint: ## Lint with ruff
	ruff check src experiments tests

fmt: ## Auto-format with ruff
	ruff check --fix src experiments tests

fiber: ## E-fiber: constant V, varying lambda2 (Example 1)
	$(PY) experiments/exp_fiber.py

phase: ## Phase diagram with the exact SDP boundary
	$(PY) experiments/exp_phase.py

methods: ## 7-method comparison, 30 seeds, bootstrap CI
	$(PY) experiments/exp_methods.py

physical: ## Closed-loop rigid-load transport
	$(PY) experiments/exp_physical.py

reproduce: ## Full pipeline: tests -> certificates -> experiments -> paper -> checks
	$(PY) experiments/reproduce.py

reproduce-fast: ## Quick smoke test of the pipeline (reduced sweeps)
	$(PY) experiments/reproduce.py --fast

paper: ## Build the IEEE manuscript
	cd paper && latexmk -pdf main.tex

clean: ## Remove build/latex artifacts
	cd paper && latexmk -C || true
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
