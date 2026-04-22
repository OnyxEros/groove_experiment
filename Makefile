# =========================================================
# CONFIG
# =========================================================

PYTHON=python
SEED=42
REPEATS=8

# default analysis pipeline
STEPS=embeddings projection clustering viz export


# =========================================================
# HELP
# =========================================================

help:
	@echo ""
	@echo "🎧 GROOVE EXPERIMENT SYSTEM"
	@echo "==========================="
	@echo ""
	@echo "📦 DATA PIPELINE"
	@echo "----------------"
	@echo "make generate              → full generation (MIDI + audio + dataset)"
	@echo "make fast                  → generation without audio rendering"
	@echo "make preview               → stimulus preview generator"
	@echo "make clean                 → full clean"
	@echo "make clean-outputs         → remove audio/MIDI outputs"
	@echo "make clean-analysis       → remove analysis outputs"
	@echo "make clean-cache          → remove __pycache__"
	@echo ""
	@echo "🧠 ANALYSIS ENGINE"
	@echo "-------------------"
	@echo "make analysis              → full pipeline (default)"
	@echo "make analysis-audio        → perceptual reduced pipeline"
	@echo "make analysis-groove       → groove-focused pipeline"
	@echo "make analysis-custom       → custom steps pipeline"
	@echo ""
	@echo "AVAILABLE STEPS:"
	@echo "  embeddings projection clustering metrics_view interpretation viz export"
	@echo ""
	@echo "📊 MODELING"
	@echo "----------"
	@echo "make regression            → train regression model"
	@echo "make perception            → perceptual alignment"
	@echo ""
	@echo "☁️ INFRA"
	@echo "--------"
	@echo "make sync                  → sync dataset to Supabase"
	@echo "make serve                 → backend API"
	@echo "make ui                    → Streamlit explorer"
	@echo ""
	@echo "🚀 FULL PIPELINE"
	@echo "----------------"
	@echo "make all                   → generate + analysis + sync + perception"
	@echo "make paper                 → research pipeline (analysis + perception + regression)"
	@echo "make repro                 → full reproducible run (clean + all)"
	@echo ""
	@echo "⚙️ CONFIG"
	@echo "--------"
	@echo "SEED=$(SEED) REPEATS=$(REPEATS)"
	@echo ""


# =========================================================
# SETUP
# =========================================================

setup:
	$(PYTHON) setup.py


# =========================================================
# DATA PIPELINE
# =========================================================

generate:
	$(PYTHON) cli.py --generate --seed $(SEED) --repeats $(REPEATS)

fast:
	$(PYTHON) cli.py --generate --seed $(SEED) --skip-audio

preview:
	$(PYTHON) cli.py --preview


# =========================================================
# CLEAN SYSTEM (MULTI-LEVEL)
# =========================================================

clean:
	$(PYTHON) cli.py --clean

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

analysis:
	$(PYTHON) cli.py --analysis --analysis-mode full

analysis-audio:
	$(PYTHON) cli.py --analysis --analysis-mode audio

analysis-groove:
	$(PYTHON) cli.py --analysis --analysis-mode groove

analysis-custom:
	$(PYTHON) cli.py --analysis --steps $(STEPS)


# =========================================================
# MODELING
# =========================================================

regression:
	$(PYTHON) cli.py --regression

perception:
	$(PYTHON) cli.py --perception


# =========================================================
# INFRA
# =========================================================

sync:
	$(PYTHON) cli.py --sync

serve:
	$(PYTHON) run_server.py

ui:
	streamlit run analysis/explorer/app.py


# =========================================================
# FULL PIPELINE
# =========================================================

all:
	$(PYTHON) cli.py --clean
	$(PYTHON) cli.py --generate --seed $(SEED) --repeats $(REPEATS)
	$(PYTHON) cli.py --analysis --analysis-mode full
	$(PYTHON) cli.py --sync
	$(PYTHON) cli.py --perception


paper:
	$(PYTHON) cli.py --analysis --analysis-mode full
	$(PYTHON) cli.py --perception
	$(PYTHON) cli.py --regression


repro:
	$(MAKE) clean
	$(MAKE) all


# =========================================================
# DEV MODE
# =========================================================

experiment:
	@echo "🚀 Starting full dev system..."
	@$(PYTHON) run_server.py &
	@sleep 2
	@streamlit run analysis/explorer/app.py


# =========================================================
# PHONY
# =========================================================

.PHONY: help setup generate fast preview clean \
        clean-outputs clean-analysis clean-metadata clean-cache \
        analysis analysis-audio analysis-groove analysis-custom \
        regression perception sync serve ui all paper repro experiment