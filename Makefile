# =========================================================
#  GROOVE EXPERIMENT SYSTEM — Makefile
#  Usage: make help
# =========================================================

PYTHON   := python
SEED     := 42
REPEATS  := 8
MODE     := full
STEPS    := embeddings projection clustering metrics_view viz export

# Colours (works in bash, zsh, fish)
BOLD  := \033[1m
GREEN := \033[32m
CYAN  := \033[36m
DIM   := \033[2m
RESET := \033[0m

# =========================================================
# HELP  (default target)
# =========================================================

.DEFAULT_GOAL := help

help:
	@echo ""
	@echo "$(BOLD)🎧  GROOVE EXPERIMENT SYSTEM$(RESET)"
	@echo "$(DIM)────────────────────────────────────────────$(RESET)"
	@echo ""
	@echo "$(BOLD)$(CYAN)📦  DATA PIPELINE$(RESET)"
	@echo "  make generate          full generation  (MIDI + audio + dataset)"
	@echo "  make fast              generation without audio rendering"
	@echo "  make preview           quick stimulus preview"
	@echo ""
	@echo "$(BOLD)$(CYAN)🧠  ANALYSIS$(RESET)"
	@echo "  make analysis          full analysis pipeline"
	@echo "  make analysis-audio    perceptual-focused pipeline"
	@echo "  make analysis-groove   groove-focused pipeline"
	@echo "  make analysis-custom   custom steps  (STEPS='...')"
	@echo ""
	@echo "$(BOLD)$(CYAN)📊  MODELLING$(RESET)"
	@echo "  make regression        train regression model"
	@echo "  make perception        perceptual alignment"
	@echo ""
	@echo "$(BOLD)$(CYAN)☁️   INFRA$(RESET)"
	@echo "  make sync              sync dataset → Supabase"
	@echo "  make serve             start FastAPI backend"
	@echo "  make ui                open Streamlit explorer"
	@echo ""
	@echo "$(BOLD)$(CYAN)🚀  FULL PIPELINES$(RESET)"
	@echo "  make all               generate + analysis + sync + perception"
	@echo "  make paper             analysis + perception + regression"
	@echo "  make repro             clean + all  (fully reproducible)"
	@echo ""
	@echo "$(BOLD)$(CYAN)🧹  CLEAN$(RESET)"
	@echo "  make clean             full clean  (all targets)"
	@echo "  make clean-outputs     MIDI / WAV / MP3 / PREVIEW"
	@echo "  make clean-analysis    analysis outputs"
	@echo "  make clean-metadata    metadata CSV"
	@echo "  make clean-cache       __pycache__"
	@echo ""
	@echo "$(BOLD)$(CYAN)⚙️   UTILS$(RESET)"
	@echo "  make status            system status + dependency check"
	@echo "  make validate          dry-run: show what would execute"
	@echo "  make setup             install dependencies"
	@echo "  make dev               start backend + UI concurrently"
	@echo ""
	@echo "$(DIM)Config: SEED=$(SEED)  REPEATS=$(REPEATS)  MODE=$(MODE)$(RESET)"
	@echo ""


# =========================================================
# AUTO __pycache__ WIPE
# Runs silently after every real target.
# Skipped for: help, status, validate, clean-*, serve, ui, dev
# =========================================================

# One-liner: no Python process needed
_wipe-cache:
	@find . -type d -name __pycache__ -not -path './.git/*' \
	  | xargs rm -rf 2>/dev/null || true
	@find . -name '*.pyc' -not -path './.git/*' -delete 2>/dev/null || true


# =========================================================
# SETUP
# =========================================================

setup:
	$(PYTHON) setup.py

install:
	pip install -r requirements.txt


# =========================================================
# STATUS & VALIDATION
# =========================================================

status:
	$(PYTHON) cli.py --status

validate:
	@echo "$(BOLD)$(CYAN)🔍  Dry-run validation$(RESET)"
	$(PYTHON) cli.py --generate --seed $(SEED) --repeats $(REPEATS) --dry-run
	$(PYTHON) cli.py --analysis --analysis-mode $(MODE) --dry-run


# =========================================================
# DATA PIPELINE
# =========================================================

generate: _wipe-cache
	$(PYTHON) cli.py --generate --seed $(SEED) --repeats $(REPEATS)
	@$(MAKE) --no-print-directory _wipe-cache

fast: _wipe-cache
	$(PYTHON) cli.py --generate --seed $(SEED) --repeats $(REPEATS) --skip-audio
	@$(MAKE) --no-print-directory _wipe-cache

preview: _wipe-cache
	$(PYTHON) cli.py --preview --seed $(SEED)
	@$(MAKE) --no-print-directory _wipe-cache


# =========================================================
# CLEAN  (multi-level)
# =========================================================

clean:
	$(PYTHON) cli.py --clean all

clean-outputs:
	$(PYTHON) cli.py --clean outputs

clean-analysis:
	$(PYTHON) cli.py --clean analysis

clean-metadata:
	$(PYTHON) cli.py --clean metadata

clean-cache:
	$(PYTHON) cli.py --clean cache


# =========================================================
# ANALYSIS ENGINE
# =========================================================

analysis: _wipe-cache
	$(PYTHON) cli.py --analysis --analysis-mode full
	@$(MAKE) --no-print-directory _wipe-cache

analysis-audio: _wipe-cache
	$(PYTHON) cli.py --analysis --analysis-mode audio
	@$(MAKE) --no-print-directory _wipe-cache

analysis-groove: _wipe-cache
	$(PYTHON) cli.py --analysis --analysis-mode groove
	@$(MAKE) --no-print-directory _wipe-cache

analysis-custom: _wipe-cache
	$(PYTHON) cli.py --analysis --steps $(STEPS)
	@$(MAKE) --no-print-directory _wipe-cache


# =========================================================
# MODELLING
# =========================================================

regression: _wipe-cache
	$(PYTHON) cli.py --regression
	@$(MAKE) --no-print-directory _wipe-cache

perception: _wipe-cache
	$(PYTHON) cli.py --perception
	@$(MAKE) --no-print-directory _wipe-cache


# =========================================================
# INFRA
# =========================================================

sync: _wipe-cache
	$(PYTHON) cli.py --sync
	@$(MAKE) --no-print-directory _wipe-cache

serve:
	$(PYTHON) run_server.py

ui:
	streamlit run analysis/explorer/app.py


# =========================================================
# FULL PIPELINES
# =========================================================

all: generate analysis sync perception
	@echo ""
	@echo "$(BOLD)$(GREEN)✔  Full pipeline complete$(RESET)"

paper: analysis perception regression
	@echo ""
	@echo "$(BOLD)$(GREEN)✔  Paper pipeline complete$(RESET)"

repro: clean all
	@echo ""
	@echo "$(BOLD)$(GREEN)✔  Reproducible run complete$(RESET)"


# =========================================================
# DEV
# =========================================================

dev:
	@echo "$(BOLD)$(CYAN)🚀  Starting backend + Streamlit UI$(RESET)"
	@$(PYTHON) run_server.py & \
	  sleep 2 && \
	  streamlit run analysis/explorer/app.py


# =========================================================
# PHONY
# =========================================================

.PHONY: \
  help setup install \
  status validate \
  generate fast preview \
  clean clean-outputs clean-analysis clean-metadata clean-cache _wipe-cache \
  analysis analysis-audio analysis-groove analysis-custom \
  regression perception \
  sync serve ui \
  all paper repro \
  dev