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
	@echo "Groove Experiment - Commands"
	@echo ""
	@echo "make run         → full pipeline"
	@echo "make fast        → quick test"
	@echo "make analysis    → MFCC + UMAP"
	@echo "make ui          → Streamlit explorer"
	@echo "make serve       → API backend"
	@echo "make clean       → delete outputs"
	@echo "make reset       → clean + rebuild"
	@echo ""

# =========================================================
# SETUP
# =========================================================

setup:
	python setup.py



# =========================================================
# PIPELINE
# =========================================================

run:
	$(PYTHON) cli.py --repeats $(REPEATS) --seed $(SEED)

all:
	$(PYTHON) cli.py --clean --repeats $(REPEATS) --seed $(SEED)

fast:
	$(PYTHON) cli.py --repeats 4 --seed $(SEED) --skip-audio

debug:
	$(PYTHON) cli.py --repeats 2 --skip-audio --analysis


# =========================================================
# ANALYSIS
# =========================================================

analysis:
	$(PYTHON) cli.py --repeats $(REPEATS) --seed $(SEED) --analysis


# =========================================================
# UI
# =========================================================

ui:
	streamlit run analysis/explorer/app.py


# =========================================================
# SERVER
# =========================================================

serve:
	$(PYTHON) run_server.py


# =========================================================
# CLEAN
# =========================================================

clean:
	$(PYTHON) cli.py --clean


# =========================================================
# RESET (VERY IMPORTANT)
# =========================================================

reset:
	$(PYTHON) cli.py --clean
	$(PYTHON) cli.py --repeats $(REPEATS) --seed $(SEED) --analysis


# =========================================================
# MARK PHONY
# =========================================================

.PHONY: run all fast debug analysis ui serve clean reset help