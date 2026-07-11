# WISE (P07) — developer entry points
.DEFAULT_GOAL := help
PY ?= python

.PHONY: help install test lint fmt \
        exp01 exp02 exp03 exp04 experiments \
        paper figures clean

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

exp01: ## Phase-transition sweep
	$(PY) experiments/exp01_phase_transition.py --config configs/phase_transition.yaml

exp02: ## Constraint ablation
	$(PY) experiments/exp02_constraint_ablation.py --config configs/ablation.yaml

exp03: ## Role emergence
	$(PY) experiments/exp03_role_emergence.py --config configs/roles.yaml

exp04: ## Self-defeat threshold validation
	$(PY) experiments/exp04_threshold_validation.py --config configs/ablation.yaml

experiments: exp01 exp02 exp03 exp04 ## Run the full campaign

paper: ## Build the IEEE manuscript
	cd paper && latexmk -pdf main.tex

figures: ## Regenerate paper figures from results/
	$(PY) -m wise_mr.metrics --render-figures

clean: ## Remove build/latex artifacts
	cd paper && latexmk -C || true
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
