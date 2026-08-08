# WISE (P07) — developer entry points
.DEFAULT_GOAL := help
PY ?= python

.PHONY: help install test lint fmt \
        fiber spatial central phase methods physical epsilon \
        robot robot-mc robot-sweep robot-fig robot-video robot-test \
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

fiber: ## E-fiber: certified fiber direction, V/aggregate neutral, lambda2 crossing
	$(PY) experiments/exp_fiber.py

spatial: ## Same-fiber unsafe vs. WISE compositions in the x-y plane
	$(PY) experiments/exp_spatial.py

central: ## Compose the central paper figure (needs fiber + spatial first)
	$(PY) experiments/make_central_fig.py

phase: ## Phase diagram with the exact SDP boundary
	$(PY) experiments/exp_phase.py

methods: ## 7-method comparison, 30 seeds, Wilson CI
	$(PY) experiments/exp_methods.py

physical: ## Closed-loop rigid-load transport
	$(PY) experiments/exp_physical.py

epsilon: ## Lexicographic WISE vs. weighted-sum scalarization (Tikhonov bounds)
	$(PY) experiments/exp_epsilon.py

robot: ## E-Robot: PHASE-R0 audit + one deterministic closed-loop flagship run
	$(PY) -m experiments.robot_closed_loop.run_flagship

robot-mc: ## E-Robot: paired Monte-Carlo campaign (default 30 seeds)
	$(PY) -m experiments.robot_closed_loop.run_monte_carlo $(SEEDS)

robot-sweep: ## E-Robot: predeclared relay-attenuation robustness sweep
	$(PY) -m experiments.robot_closed_loop.run_margin_sweep

robot-fig: ## E-Robot: paper hero figure + supplementary four-panel figure
	$(PY) -m experiments.robot_closed_loop.make_hero
	$(PY) -m experiments.robot_closed_loop.make_figure

robot-video: ## E-Robot: PROD | HARD | WISE side-by-side animation
	$(PY) -m experiments.robot_closed_loop.render

robot-test: ## E-Robot: the 12 invariant tests
	$(PY) -m pytest experiments/robot_closed_loop/tests -q

reproduce: ## Full pipeline: tests -> certificates -> experiments -> paper -> checks
	$(PY) experiments/reproduce.py

reproduce-fast: ## Quick smoke test of the pipeline (reduced sweeps)
	$(PY) experiments/reproduce.py --fast

paper: ## Build the IEEE manuscript
	cd paper && latexmk -pdf main.tex

clean: ## Remove build/latex artifacts
	cd paper && latexmk -C || true
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
