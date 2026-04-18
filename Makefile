# =========================================================
# CONFIG
# =========================================================

PYTHON=python
SEED=42
REPEATS=8

# analysis steps (safe default)
STEPS=embedding clustering viz


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
	@echo "make generate        → full generation pipeline (MIDI + audio + dataset)"
	@echo "make fast            → generation without audio rendering"
	@echo "make clean           → delete generated data"
	@echo "make preview         → test stimulus generator"
	@echo ""
	@echo "🧠 ANALYSIS (READ ONLY)"
	@echo "----------------------"
	@echo "make analysis        → full analysis pipeline"
	@echo "make analysis-light  → embeddings only"
	@echo "make analysis-cluster→ embeddings + clustering"
	@echo "make analysis-viz    → embeddings + visualization"
	@echo ""
	@echo "📊 MODELING"
	@echo "----------"
	@echo "make regression      → train regression model"
	@echo "make perception      → perceptual alignment"
	@echo ""
	@echo "☁️ INFRA"
	@echo "--------"
	@echo "make sync            → sync dataset to Supabase"
	@echo "make serve           → backend API"
	@echo "make ui              → Streamlit explorer"
	@echo ""
	@echo "🚀 FULL PIPELINE"
	@echo "--------------"
	@echo "make all             → full rebuild pipeline"
	@echo "make paper           → research pipeline"
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
# DATA PIPELINE (GENERATION ONLY)
# =========================================================

generate:
	$(PYTHON) cli.py --generate --seed $(SEED) --repeats $(REPEATS)

fast:
	$(PYTHON) cli.py --generate --seed $(SEED) --skip-audio

clean:
	$(PYTHON) cli.py --clean

preview:
	$(PYTHON) cli.py --preview


# =========================================================
# ANALYSIS (NO GENERATION)
# =========================================================

analysis:
	$(PYTHON) cli.py --analysis --analysis-mode audio --steps $(STEPS)

analysis-light:
	$(PYTHON) cli.py --analysis --steps embedding

analysis-cluster:
	$(PYTHON) cli.py --analysis --steps embedding clustering

analysis-viz:
	$(PYTHON) cli.py --analysis --steps embedding viz


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
	$(PYTHON) cli.py --analysis --steps $(STEPS)
	$(PYTHON) cli.py --sync
	$(PYTHON) cli.py --perception


paper:
	$(PYTHON) cli.py --analysis --steps embedding clustering viz
	$(PYTHON) cli.py --perception
	$(PYTHON) cli.py --regression


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

.PHONY: help setup generate fast clean preview \
        analysis analysis-light analysis-cluster analysis-viz \
        regression perception sync serve ui all paper experiment