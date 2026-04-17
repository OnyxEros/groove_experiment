# =========================================================
# CONFIG
# =========================================================

PYTHON=python
REPEATS=8
SEED=42


# =========================================================
# HELP
# =========================================================
help:
	@echo ""
	@echo "🎧 Groove Experiment System"
	@echo ""
	@echo "Core pipeline:"
	@echo "  make run         → generate dataset (MIDI + audio + metadata)"
	@echo "  make fast        → quick generation (no audio rendering)"
	@echo "  make analysis    → UMAP embedding (audio + groove space)"
	@echo "  make sync        → push dataset to Supabase"
	@echo "  make regression  → train perceptual model"
	@echo ""
	@echo "Full pipeline:"
	@echo "  make all         → full generation + analysis + sync"
	@echo "  make reset       → clean + rebuild everything"
	@echo ""
	@echo "Interface:"
	@echo "  make ui          → Streamlit explorer"
	@echo "  make serve       → FastAPI backend"
	@echo "  make experiment  → full stack (API + UI)"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean       → remove generated data"
	@echo "  make clean-cache → remove pycache files"
	@echo ""


# =========================================================
# SETUP
# =========================================================
setup:
	$(PYTHON) setup.py


# =========================================================
# PIPELINE (GENERATION)
# =========================================================
run:
	$(PYTHON) cli.py --repeats $(REPEATS) --seed $(SEED)

fast:
	$(PYTHON) cli.py --repeats 4 --seed $(SEED) --skip-audio

all:
	$(PYTHON) cli.py --clean
	$(PYTHON) cli.py --repeats $(REPEATS) --seed $(SEED)
	$(PYTHON) cli.py --analysis
	$(PYTHON) cli.py --sync


# =========================================================
# ANALYSIS (UMAP)
# =========================================================
analysis:
	$(PYTHON) cli.py --analysis-only


# =========================================================
# SUPABASE SYNC
# =========================================================
sync:
	$(PYTHON) cli.py --sync


# =========================================================
# REGRESSION
# =========================================================
regression:
	$(PYTHON) cli.py --regression


# =========================================================
# BACKEND
# =========================================================
serve:
	$(PYTHON) run_server.py


# =========================================================
# UI
# =========================================================
ui:
	streamlit run analysis/explorer/app.py


# =========================================================
# FULL STACK DEV MODE
# =========================================================
experiment:
	@echo "🚀 Starting full system..."
	@echo "1/ FastAPI backend..."
	@$(PYTHON) run_server.py &
	@sleep 2
	@echo "2/ Streamlit UI..."
	@streamlit run analysis/explorer/app.py


# =========================================================
# CLEAN
# =========================================================
clean:
	$(PYTHON) cli.py --clean

clean-cache:
	$(PYTHON) clean_pycache.py


# =========================================================
# RESET (SAFE FULL REBUILD)
# =========================================================
reset:
	@echo "🧹 Full reset pipeline..."
	$(PYTHON) cli.py --clean
	$(PYTHON) cli.py --repeats $(REPEATS) --seed $(SEED)
	$(PYTHON) cli.py --analysis
	$(PYTHON) cli.py --sync
	$(PYTHON) cli.py --regression


# =========================================================
# PHONY
# =========================================================
.PHONY: help setup run fast all analysis sync regression serve ui experiment clean reset