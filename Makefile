# =============================================================================
# Makefile — Adaptive Multi-Tier KV Cache Orchestrator
# =============================================================================
# Usage: make <target>
# Run `make help` to list all available targets.
# =============================================================================

SHELL           := /bin/bash
.DEFAULT_GOAL   := help

# ---------------------------------------------------------------------------
# Configurable variables (override via CLI: make serve PORT=9000)
# ---------------------------------------------------------------------------
SRC_DIR         := src
TEST_DIR        := tests
SCRIPTS_DIR     := scripts
CONFIG_DIR      := configs
DOCKER_DIR      := docker

HOST            := 0.0.0.0
PORT            := 8000
CONFIG          := $(CONFIG_DIR)/default.yaml

PYTHON          := python
PIP             := pip
PYTEST          := pytest
RUFF            := ruff
MYPY            := mypy
UVICORN         := uvicorn

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
BOLD  := $(shell tput -T xterm bold   2>/dev/null || true)
RESET := $(shell tput -T xterm sgr0   2>/dev/null || true)
GREEN := $(shell tput -T xterm setaf 2 2>/dev/null || true)
CYAN  := $(shell tput -T xterm setaf 6 2>/dev/null || true)

# ---------------------------------------------------------------------------
# Phony targets
# ---------------------------------------------------------------------------
.PHONY: install install-dev test test-fast test-cov lint format typecheck \
        serve benchmark train-policy docker-up docker-down docker-logs \
        clean results-clean help all

# =============================================================================
# Installation
# =============================================================================

## install: Install the package in editable mode (runtime deps only)
install:
	@echo "$(CYAN)Installing kv_orchestrator (editable)...$(RESET)"
	$(PIP) install -e "."

## install-dev: Install the package + all development dependencies
install-dev:
	@echo "$(CYAN)Installing kv_orchestrator with dev extras...$(RESET)"
	$(PIP) install -e ".[dev]"
	@echo "$(CYAN)Installing pre-commit hooks...$(RESET)"
	pre-commit install --install-hooks || true
	@echo "$(GREEN)Dev environment ready.$(RESET)"

# =============================================================================
# Testing
# =============================================================================

## test: Run full test suite with short tracebacks
test:
	@echo "$(CYAN)Running test suite...$(RESET)"
	$(PYTEST) $(TEST_DIR)/ -v --tb=short

## test-fast: Run fast tests only (excludes @pytest.mark.slow)
test-fast:
	@echo "$(CYAN)Running fast tests (skipping slow markers)...$(RESET)"
	$(PYTEST) $(TEST_DIR)/ -v -m "not slow"

## test-cov: Run tests with HTML coverage report
test-cov:
	@echo "$(CYAN)Running tests with coverage...$(RESET)"
	$(PYTEST) $(TEST_DIR)/ -v --tb=short \
	  --cov=$(SRC_DIR) \
	  --cov-report=term-missing \
	  --cov-report=html:htmlcov \
	  --cov-fail-under=70
	@echo "$(GREEN)Coverage report written to htmlcov/index.html$(RESET)"

# =============================================================================
# Code Quality
# =============================================================================

## lint: Run ruff linter on src/ and tests/
lint:
	@echo "$(CYAN)Linting with ruff...$(RESET)"
	$(RUFF) check $(SRC_DIR)/ $(TEST_DIR)/

## format: Auto-format src/ and tests/ with ruff
format:
	@echo "$(CYAN)Formatting with ruff...$(RESET)"
	$(RUFF) format $(SRC_DIR)/ $(TEST_DIR)/

## lint-fix: Lint and auto-fix safe violations
lint-fix:
	@echo "$(CYAN)Linting and fixing with ruff...$(RESET)"
	$(RUFF) check --fix $(SRC_DIR)/ $(TEST_DIR)/

## typecheck: Run mypy static type analysis on src/
typecheck:
	@echo "$(CYAN)Type-checking with mypy...$(RESET)"
	$(MYPY) $(SRC_DIR)/

## check: Run lint + typecheck (non-destructive CI gate)
check: lint typecheck

# =============================================================================
# Serving
# =============================================================================

## serve: Launch the FastAPI inference server with hot-reload
serve:
	@echo "$(CYAN)Starting API server on $(HOST):$(PORT)...$(RESET)"
	$(UVICORN) src.serving.api_server:app \
	  --host $(HOST) \
	  --port $(PORT) \
	  --reload \
	  --log-level info

## serve-prod: Launch production server (no reload, multiple workers)
serve-prod:
	@echo "$(CYAN)Starting production API server on $(HOST):$(PORT)...$(RESET)"
	$(UVICORN) src.serving.api_server:app \
	  --host $(HOST) \
	  --port $(PORT) \
	  --workers 4 \
	  --log-level warning \
	  --access-log

# =============================================================================
# ML Workflows
# =============================================================================

## benchmark: Run the standard benchmark suite against default config
benchmark:
	@echo "$(CYAN)Running benchmark suite (config: $(CONFIG))...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/run_benchmark.py --config $(CONFIG)

## train-policy: Train the learned eviction policy model
train-policy:
	@echo "$(CYAN)Training eviction policy (config: $(CONFIG))...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/train_policy.py --config $(CONFIG)

## benchmark-longbench: Run LongBench-specific evaluation
benchmark-longbench:
	@echo "$(CYAN)Running LongBench evaluation...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/run_benchmark.py \
	  --config $(CONFIG) \
	  --override benchmark=$(CONFIG_DIR)/benchmarks/longbench.yaml

## benchmark-stress: Run stress / load test
benchmark-stress:
	@echo "$(CYAN)Running stress benchmark...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/run_benchmark.py \
	  --config $(CONFIG) \
	  --override benchmark=$(CONFIG_DIR)/benchmarks/stress.yaml

## collect-traces: Collect KV-cache telemetry traces for policy training
collect-traces:
	@echo "$(CYAN)Collecting telemetry traces...$(RESET)"
	$(PYTHON) $(SCRIPTS_DIR)/collect_traces.py --config $(CONFIG)

# =============================================================================
# Docker
# =============================================================================

## docker-up: Start all services (API, Prometheus, Grafana, Redis) in background
docker-up:
	@echo "$(CYAN)Starting Docker services...$(RESET)"
	docker-compose -f $(DOCKER_DIR)/docker-compose.yml up -d
	@echo "$(GREEN)Services started. API: http://localhost:8000  Grafana: http://localhost:3000$(RESET)"

## docker-down: Stop and remove all Docker services
docker-down:
	@echo "$(CYAN)Stopping Docker services...$(RESET)"
	docker-compose -f $(DOCKER_DIR)/docker-compose.yml down

## docker-logs: Tail logs for all running services
docker-logs:
	docker-compose -f $(DOCKER_DIR)/docker-compose.yml logs -f

## docker-build: (Re)build the API server Docker image
docker-build:
	@echo "$(CYAN)Building Docker image...$(RESET)"
	docker-compose -f $(DOCKER_DIR)/docker-compose.yml build --no-cache api

## docker-restart: Restart a specific service (usage: make docker-restart SERVICE=api)
docker-restart:
	docker-compose -f $(DOCKER_DIR)/docker-compose.yml restart $(SERVICE)

# =============================================================================
# Cleanup
# =============================================================================

## clean: Remove Python cache files, build artifacts, and test cache
clean:
	@echo "$(CYAN)Cleaning Python artifacts...$(RESET)"
	find . -type d -name "__pycache__"   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info"    -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache"   -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc"         -delete                        2>/dev/null || true
	find . -type f -name "*.pyo"         -delete                        2>/dev/null || true
	find . -type d -name "htmlcov"       -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage"     -delete                        2>/dev/null || true
	@echo "$(GREEN)Clean complete.$(RESET)"

## results-clean: Remove all generated result files (tables, plots, logs)
results-clean:
	@echo "$(CYAN)Cleaning results directory...$(RESET)"
	rm -f results/tables/*.csv
	rm -f results/plots/*.png
	rm -f results/plots/*.html
	rm -f results/plots/*.pdf
	rm -rf results/logs/
	@echo "$(GREEN)Results cleaned.$(RESET)"

## clean-all: Full clean including results and Hydra outputs
clean-all: clean results-clean
	@echo "$(CYAN)Removing Hydra/experiment outputs...$(RESET)"
	rm -rf outputs/ multirun/ .hydra/
	@echo "$(GREEN)Full clean complete.$(RESET)"

# =============================================================================
# Help
# =============================================================================

## help: Print this help message (lists all targets with descriptions)
help:
	@echo ""
	@echo "$(BOLD)Adaptive Multi-Tier KV Cache Orchestrator$(RESET)"
	@echo "$(BOLD)===========================================$(RESET)"
	@echo ""
	@echo "$(BOLD)Available targets:$(RESET)"
	@echo ""
	@grep -E '^## [a-zA-Z_-]+:.*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; /^## / {split($$0, a, ": "); printf "  $(GREEN)%-22s$(RESET) %s\n", a[2], substr($$0, index($$0, a[2]) + length(a[2]) + 2)}' \
	  | sed 's/## //'
	@echo ""
	@echo "$(BOLD)Overridable variables:$(RESET)"
	@echo "  HOST=$(HOST)  PORT=$(PORT)  CONFIG=$(CONFIG)"
	@echo ""

## all: install-dev + check + test
all: install-dev check test
