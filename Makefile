# ===========================================================
#  GROOVE EXPERIMENT SYSTEM — Makefile
#
#  Usage    : make help
#  Requires : GNU Make ≥ 3.81, Python ≥ 3.11
# ===========================================================

# ── Tooling ─────────────────────────────────────────────────
PYTHON      ?= python
PIP         ?= pip

# ── Experiment defaults (overrideable on CLI) ───────────────
SEED        ?= 42
REPEATS     ?=
MODE        ?= full
STEPS       ?= embeddings projection clustering metrics_view viz export
FEATURE_SET ?= all

# ── Flags (set to 1 to enable) ──────────────────────────────
NO_CHECK_DB ?= 0      # 1 → skip Supabase check in regression
SKIP_AUDIO  ?= 0      # 1 → skip WAV/MP3 rendering in generate

# ── Internal helpers ────────────────────────────────────────
_NO_DB_FLAG  = $(if $(filter 1,$(NO_CHECK_DB)),--no-check-db,)
_AUDIO_FLAG  = $(if $(filter 1,$(SKIP_AUDIO)),--skip-audio,)

# ── Colours ─────────────────────────────────────────────────
BOLD  := \033[1m
GREEN := \033[32m
CYAN  := \033[36m
YELLOW:= \033[33m
DIM   := \033[2m
RESET := \033[0m

.DEFAULT_GOAL := help

# ===========================================================
# HELP
# ===========================================================

.PHONY: help
help:
	@printf "\n$(BOLD)🎧  GROOVE EXPERIMENT SYSTEM$(RESET)\n"
	@printf "$(DIM)──────────────────────────────────────────────$(RESET)\n\n"

	@printf "$(BOLD)$(CYAN)📦  DATA PIPELINE$(RESET)\n"
	@printf "  %-28s %s\n" "make generate"        "Full generation (MIDI + audio + metadata.csv)"
	@printf "  %-28s %s\n" "make fast"             "Generation without audio  (SKIP_AUDIO=1)"
	@printf "  %-28s %s\n" "make preview"          "3 preview stimuli (baseline / swing / syncopated)"
	@printf "\n"

	@printf "$(BOLD)$(CYAN)🧠  ANALYSIS$(RESET)\n"
	@printf "  %-28s %s\n" "make new-run"          "Crée un nouveau dossier de run (requis avant analysis)"
	@printf "  %-28s %s\n" "make analysis"         "new-run + full analysis pipeline  (MODE=full)"
	@printf "  %-28s %s\n" "make analysis-audio"   "new-run + perceptual-focused pipeline"
	@printf "  %-28s %s\n" "make analysis-groove"  "new-run + groove-focused pipeline"
	@printf "  %-28s %s\n" "make analysis-custom"  "new-run + custom steps  (STEPS='...')"
	@printf "\n"

	@printf "$(BOLD)$(CYAN)📊  MODELLING$(RESET)\n"
	@printf "  %-28s %s\n" "make regression"          "One feature set  (FEATURE_SET=design|acoustic|all)"
	@printf "  %-28s %s\n" "make regression-design"   "Design params only  (S_mv, D_mv, E)"
	@printf "  %-28s %s\n" "make regression-acoustic" "Acoustic metrics only  (D, I, V, S_real, E_real)"
	@printf "  %-28s %s\n" "make regression-all"      "All 3 feature sets — thesis mode"
	@printf "  %-28s %s\n" "make perception"           "Perceptual alignment  (latent → ratings)"
	@printf "  %-28s %s\n" "make perception-space"     "Geometric analysis of groove in UMAP space"
	@printf "  %-28s %s\n" "make refresh"              "Re-fetch Supabase + rerun regression + perception"
	@printf "  %-28s %s\n" "make regression-fast"      "Regression without Supabase check  (offline)"
	@printf "\n"

	@printf "$(BOLD)$(CYAN)☁️   INFRA$(RESET)\n"
	@printf "  %-28s %s\n" "make sync"    "Fetch Supabase → data/responses.csv  (read-only)"
	@printf "  %-28s %s\n" "make serve"   "Start FastAPI backend"
	@printf "  %-28s %s\n" "make ui"      "Open Streamlit explorer"
	@printf "\n"

	@printf "$(BOLD)$(CYAN)🚀  FULL PIPELINES$(RESET)\n"
	@printf "  %-28s %s\n" "make all"     "generate + new-run + analysis + sync + regression-all + perception"
	@printf "  %-28s %s\n" "make paper"   "new-run + sync + regression-all + perception + perception-space"
	@printf "  %-28s %s\n" "make repro"   "clean + all  (fully reproducible from scratch)"
	@printf "\n"

	@printf "$(BOLD)$(CYAN)🧹  CLEAN$(RESET)\n"
	@printf "  %-28s %s\n" "make clean"            "Full clean  (all targets)"
	@printf "  %-28s %s\n" "make clean-outputs"    "MIDI / WAV / MP3 / PREVIEW"
	@printf "  %-28s %s\n" "make clean-analysis"   "data/analysis/ + .current_run"
	@printf "  %-28s %s\n" "make clean-metadata"   "data/metadata.csv"
	@printf "  %-28s %s\n" "make clean-responses"  "data/responses.csv  (local Supabase cache)"
	@printf "  %-28s %s\n" "make clean-cache"      "__pycache__ + *.pyc"
	@printf "\n"

	@printf "$(BOLD)$(CYAN)⚙️   UTILS$(RESET)\n"
	@printf "  %-28s %s\n" "make status"   "System status  (dirs, deps, cache)"
	@printf "  %-28s %s\n" "make doctor"   "Supabase + env diagnostic"
	@printf "  %-28s %s\n" "make validate" "Dry-run all major commands"
	@printf "  %-28s %s\n" "make setup"    "Install Python dependencies"
	@printf "  %-28s %s\n" "make dev"      "Start backend + Streamlit concurrently"
	@printf "\n"

	@printf "$(DIM)Config: SEED=$(SEED)  REPEATS=$(REPEATS)  MODE=$(MODE)  FEATURE_SET=$(FEATURE_SET)  NO_CHECK_DB=$(NO_CHECK_DB)  SKIP_AUDIO=$(SKIP_AUDIO)$(RESET)\n\n"

# ===========================================================
# INTERNAL: AUTO PYCACHE WIPE
# ===========================================================

.PHONY: _wipe
_wipe:
	@find . -type d -name __pycache__ -not -path './.git/*' \
	    | xargs rm -rf 2>/dev/null || true
	@find . -name '*.pyc' -not -path './.git/*' \
	    -delete 2>/dev/null || true

# ===========================================================
# SETUP
# ===========================================================

.PHONY: setup install
setup:
	$(PYTHON) setup.py

install:
	$(PIP) install -r requirements.txt

# ===========================================================
# STATUS / DIAGNOSTIC
# ===========================================================

.PHONY: status doctor validate
status:
	$(PYTHON) cli.py --status

doctor:
	$(PYTHON) cli.py --doctor

validate:
	@printf "$(BOLD)$(CYAN)🔍  Dry-run validation$(RESET)\n"
	$(PYTHON) cli.py --generate --seed $(SEED) --repeats $(REPEATS) $(_AUDIO_FLAG) --dry-run
	$(PYTHON) cli.py --analysis --analysis-mode $(MODE) --dry-run
	$(PYTHON) cli.py --sync --dry-run
	$(PYTHON) cli.py --regression --feature-set $(FEATURE_SET) $(_NO_DB_FLAG) --dry-run
	$(PYTHON) cli.py --regression-all $(_NO_DB_FLAG) --dry-run
	$(PYTHON) cli.py --perception --dry-run
	$(PYTHON) cli.py --perception-space --dry-run

# ===========================================================
# DATA PIPELINE
# ===========================================================

.PHONY: generate fast preview
generate: _wipe
	$(PYTHON) cli.py --generate --seed $(SEED) $(if $(REPEATS),--repeats $(REPEATS),)
	@$(MAKE) --no-print-directory _wipe

fast: _wipe
	$(PYTHON) cli.py --generate --seed $(SEED) --repeats $(REPEATS) --skip-audio
	@$(MAKE) --no-print-directory _wipe

preview: _wipe
	$(PYTHON) cli.py --preview --seed $(SEED)
	@$(MAKE) --no-print-directory _wipe

# ===========================================================
# CLEAN
# ===========================================================

.PHONY: clean clean-outputs clean-analysis clean-metadata clean-responses clean-cache
clean:
	$(PYTHON) cli.py --clean all

clean-outputs:
	$(PYTHON) cli.py --clean outputs

clean-analysis:
	$(PYTHON) cli.py --clean analysis

clean-metadata:
	$(PYTHON) cli.py --clean metadata

clean-responses:
	$(PYTHON) cli.py --clean responses

clean-cache:
	$(PYTHON) cli.py --clean cache

# ===========================================================
# DOSSIER DE RUN D'ANALYSE
# ===========================================================

.PHONY: new-run
new-run:
	$(PYTHON) cli.py --new-run

# ===========================================================
# ANALYSIS ENGINE
# ===========================================================

.PHONY: analysis analysis-audio analysis-groove analysis-custom
analysis: _wipe
	$(PYTHON) cli.py --analysis --analysis-mode full
	@$(MAKE) --no-print-directory _wipe

analysis-audio: _wipe
	$(PYTHON) cli.py --analysis --analysis-mode audio
	@$(MAKE) --no-print-directory _wipe

analysis-groove: _wipe
	$(PYTHON) cli.py --analysis --analysis-mode groove
	@$(MAKE) --no-print-directory _wipe

analysis-custom: _wipe
	$(PYTHON) cli.py --analysis --steps $(STEPS)
	@$(MAKE) --no-print-directory _wipe

# ===========================================================
# MODELLING
# ===========================================================

.PHONY: regression regression-design regression-acoustic regression-all
.PHONY: regression-fast perception perception-space refresh

regression: _wipe
	$(PYTHON) cli.py --regression --feature-set $(FEATURE_SET) $(_NO_DB_FLAG)
	@$(MAKE) --no-print-directory _wipe

regression-design: _wipe
	$(PYTHON) cli.py --regression --feature-set design $(_NO_DB_FLAG)
	@$(MAKE) --no-print-directory _wipe

regression-acoustic: _wipe
	$(PYTHON) cli.py --regression --feature-set acoustic $(_NO_DB_FLAG)
	@$(MAKE) --no-print-directory _wipe

regression-all: _wipe
	$(PYTHON) cli.py --regression-all $(_NO_DB_FLAG)
	@$(MAKE) --no-print-directory _wipe

regression-fast: _wipe
	$(PYTHON) cli.py --regression --feature-set $(FEATURE_SET) --no-check-db
	@$(MAKE) --no-print-directory _wipe

perception: _wipe
	$(PYTHON) cli.py --perception
	@$(MAKE) --no-print-directory _wipe

perception-space: _wipe
	$(PYTHON) cli.py --perception-space
	@$(MAKE) --no-print-directory _wipe

refresh: _wipe
	$(PYTHON) cli.py --regression-all --perception --refresh $(_NO_DB_FLAG)
	@$(MAKE) --no-print-directory _wipe

# ===========================================================
# INFRA
# ===========================================================

.PHONY: sync serve ui
sync: _wipe
	$(PYTHON) cli.py --sync
	@$(MAKE) --no-print-directory _wipe

serve:
	$(PYTHON) run_server.py

ui:
	streamlit run analysis/explorer/app.py

# ===========================================================
# FULL PIPELINES
# ===========================================================

.PHONY: all paper repro

# Option propre — new-run seulement dans all, pas dans analysis
all: generate new-run _wipe
	$(PYTHON) cli.py --analysis --analysis-mode full
	@$(MAKE) --no-print-directory _wipe
	$(PYTHON) cli.py --sync
	$(PYTHON) cli.py --regression-all $(_NO_DB_FLAG)
	$(PYTHON) cli.py --perception
	@printf "\n$(BOLD)$(GREEN)✔  Full pipeline complete$(RESET)\n\n"

paper: new-run _wipe
	@printf "$(BOLD)$(CYAN)📄  Paper pipeline$(RESET)\n"
	$(PYTHON) cli.py --sync
	$(PYTHON) cli.py --regression-all --refresh --perception --perception-space $(_NO_DB_FLAG)
	@$(MAKE) --no-print-directory _wipe
	@printf "\n$(BOLD)$(GREEN)✔  Paper pipeline complete$(RESET)\n\n"

repro: clean all
	@printf "\n$(BOLD)$(GREEN)✔  Reproducible run complete$(RESET)\n\n"

# ===========================================================
# DEV
# ===========================================================

.PHONY: dev
dev:
	@printf "$(BOLD)$(CYAN)🚀  Starting backend + Streamlit$(RESET)\n"
	@$(PYTHON) run_server.py & SERVER_PID=$$!; \
	 sleep 1 && streamlit run analysis/explorer/app.py & UI_PID=$$!; \
	 trap "kill $$SERVER_PID $$UI_PID 2>/dev/null" INT TERM; \
	 wait $$SERVER_PID $$UI_PID