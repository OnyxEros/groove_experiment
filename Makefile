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
	@echo "  make run         → full generation (MIDI + audio + dataset)"
	@echo "  make fast        → generation sans audio"
	@echo "  make analysis    → analyse UMAP / pipeline audio"
	@echo "  make sync        → sync dataset vers Supabase"
	@echo "  make regression  → train modèle de régression"
	@echo "  make perception  → perceptual alignment (NEW)"
	@echo ""
	@echo "Full pipeline:"
	@echo "  make all         → clean + full pipeline + sync + perception"
	@echo "  make reset       → rebuild complet"
	@echo ""
	@echo "Interface:"
	@echo "  make serve       → backend FastAPI"
	@echo "  make ui          → Streamlit explorer"
	@echo "  make experiment  → backend + UI"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean       → delete generated data"
	@echo "  make clean-cache → remove __pycache__"
	@echo ""


# =========================================================
# SETUP
# =========================================================
setup:
	$(PYTHON) setup.py


# =========================================================
# PIPELINE
# =========================================================
run:
	$(PYTHON) cli.py --repeats $(REPEATS) --seed $(SEED)

fast:
	$(PYTHON) cli.py --repeats 4 --seed $(SEED) --skip-audio


# full clean + run + analysis + sync + perception
all:
	$(PYTHON) cli.py --clean
	$(PYTHON) cli.py --repeats $(REPEATS) --seed $(SEED)
	$(PYTHON) cli.py --analysis
	$(PYTHON) cli.py --sync
	$(PYTHON) cli.py --perception



paper:
	$(PYTHON) cli.py --analysis
	$(PYTHON) cli.py --perception
	$(PYTHON) cli.py --regression
	$(PYTHON) cli.py export-figures

# =========================================================
# ANALYSIS
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
# PERCEPTION (NEW)
# =========================================================
perception:
	$(PYTHON) cli.py --perception


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
	@echo "🚀 Starting system..."
	@$(PYTHON) run_server.py &
	@sleep 2
	@streamlit run analysis/explorer/app.py


# =========================================================
# CLEAN
# =========================================================
clean:
	$(PYTHON) cli.py --clean

clean-cache:
	$(PYTHON) clean_pycache.py


# =========================================================
# RESET
# =========================================================
reset:
	@echo "🧹 Full reset..."
	$(PYTHON) cli.py --clean
	$(PYTHON) cli.py --repeats $(REPEATS) --seed $(SEED)
	$(PYTHON) cli.py --analysis
	$(PYTHON) cli.py --sync
	$(PYTHON) cli.py --regression
	$(PYTHON) cli.py --perception


# =========================================================
# PHONY
# =========================================================
.PHONY: help setup run fast all analysis sync regression perception serve ui experiment clean clean-cache reset