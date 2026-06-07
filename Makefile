# ============================================================
# Makefile — Groove Experiment
# Mémoire TSMA2 — EMC — Viken Karaboghossian
# ============================================================
.DEFAULT_GOAL := help

# ============================================================
# VARIABLES CONFIGURABLES
# ============================================================

PYTHON         ?= python
SEED           ?= 42
REPEATS        ?=
EXCLUDE_SINGLE ?= 1
FEATURE_SET    ?= all
ANALYSIS_MODE  ?= groove
FIGURES_OUT    ?= figures_memoire
PORT           ?= 8000
FORCE          ?= 0
REFRESH        ?= 0

# Drapeaux conditionnels
_SINGLE_FLAG  = $(if $(filter 0,$(EXCLUDE_SINGLE)),--include-single,)
_REPEATS_FLAG = $(if $(REPEATS),--repeats $(REPEATS),)
_REFRESH_FLAG = $(if $(filter 1,$(REFRESH)),--refresh,)
_FORCE_FLAG   = $(if $(filter 1,$(FORCE)),--force,)

# Détection environnement virtuel
IN_VENV := $(if $(VIRTUAL_ENV),1,0)

# ============================================================
# .PHONY — liste exhaustive
# ============================================================

.PHONY: help install setup env-check \
        generate regenerate lock validate preview \
        serve sync sync-refresh \
        new-run analysis new-run-analysis \
        regression regression-all regression-interactions \
        perception perc-space \
        thesis figures \
        status doctor dry-generate dry-thesis \
        clean clean-outputs clean-analysis clean-responses clean-cache \
        setup-experiment model \
        _require-venv _require-metadata _require-responses \
        _require-run _require-soundfont _require-mp3

# ============================================================
# AIDE AUTO-GÉNÉRÉE
# ============================================================

help: ## Affiche ce menu d'aide
	@echo ""
	@echo "  ╔══════════════════════════════════════════════════════════╗"
	@echo "  ║  🎧  Groove Experiment — Makefile                       ║"
	@echo "  ║  Mémoire TSMA2  ·  Viken Karaboghossian                 ║"
	@echo "  ╚══════════════════════════════════════════════════════════╝"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "  \033[1;34mCommandes disponibles :\033[0m\n\n"} \
		/^##@/ { printf "\n  \033[1;33m%s\033[0m\n", substr($$0, 5) } \
		/^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-26s\033[0m %s\n", $$1, $$2 }' \
		$(MAKEFILE_LIST)
	@echo ""
	@echo "  Variables :  SEED=$(SEED)  FEATURE_SET=$(FEATURE_SET)  FORCE=$(FORCE)"
	@echo "               EXCLUDE_SINGLE=$(EXCLUDE_SINGLE)  PORT=$(PORT)"
	@echo "               ANALYSIS_MODE=$(ANALYSIS_MODE)  REFRESH=$(REFRESH)"
	@echo ""

# ============================================================
##@ §4.1  Génération des stimuli
# ============================================================

generate: ## Génère MIDI + MP3 + metadata.csv  (bloqué si verrou actif)
	@echo "🎛️   §4.1  Génération des stimuli  (seed=$(SEED))…"
	@if [ -f data/metadata.csv ] && [ "$(FORCE)" != "1" ]; then \
		echo "ℹ️   data/metadata.csv existe déjà. Utilisez 'make regenerate' pour forcer."; \
	fi
	$(PYTHON) cli.py --generate --seed $(SEED) $(_REPEATS_FLAG) \
		$(if $(filter 1,$(SKIP_AUDIO)),--skip-audio,) \
		$(_FORCE_FLAG)

regenerate: ## Force la régénération même si metadata.csv existe  (FORCE implicite)
	@echo "🎛️   §4.1  Régénération forcée  (seed=$(SEED))…"
	$(PYTHON) cli.py --generate --seed $(SEED) $(_REPEATS_FLAG) --force

lock: _require-mp3 ## Crée/vérifie le verrou campagne avec checksum MD5 des MP3
	@echo "🔒  Verrou campagne…"
	$(PYTHON) cli.py --lock

validate: _require-run ## Figures de validation structurelle du générateur (§4.1.4)
	@echo "🔬  §4.1.4  Validation structurelle…"
	$(PYTHON) cli.py --validate

preview: _require-soundfont ## 3 stimuli de référence pour écoute comparative
	@echo "🎧  §4.1  Preview (baseline · swing · syncopated)…"
	$(PYTHON) cli.py --preview --seed $(SEED)

# ============================================================
##@ §4.2  Collecte des données
# ============================================================

serve: ## Démarre le serveur FastAPI  (http://localhost:PORT)
	@echo "🌐  §4.2  Démarrage du serveur → http://localhost:$(PORT)"
	PORT=$(PORT) $(PYTHON) run_server.py

sync: ## Supabase → data/responses.csv  (+ validation stim_ids)
	@echo "☁️   §4.2  Synchronisation Supabase → cache local…"
	$(PYTHON) cli.py --sync

sync-refresh: ## Sync forcé — ignore le cache local
	@echo "☁️   §4.2  Sync forcé (--refresh)…"
	$(PYTHON) cli.py --sync --refresh

# ============================================================
##@ §5  Espace latent
# ============================================================

new-run: ## Crée un dossier de run horodaté dans data/analysis/
	@echo "📁  §5  Nouveau run d'analyse…"
	$(PYTHON) cli.py --new-run

analysis: _require-run _require-metadata ## Embeddings + UMAP + clustering + export
	@echo "🧠  §5  Analyse  (mode=$(ANALYSIS_MODE))…"
	$(PYTHON) cli.py --analysis --analysis-mode $(ANALYSIS_MODE)

new-run-analysis: _require-metadata ## Crée un run ET lance l'analyse immédiatement
	@echo "📁🧠  §5  Nouveau run + analyse…"
	$(PYTHON) cli.py --new-run
	$(PYTHON) cli.py --analysis --analysis-mode $(ANALYSIS_MODE)

# ============================================================
##@ §6  Modélisation statistique
# ============================================================

regression: _require-responses ## Régression groove  (FEATURE_SET=all par défaut)
	@echo "📈  §6.2  Régression  (features=$(FEATURE_SET))…"
	$(PYTHON) cli.py --regression \
		--feature-set $(FEATURE_SET) \
		$(_SINGLE_FLAG) \
		$(_REFRESH_FLAG) \
		--no-check-db

regression-all: _require-responses ## Régression sur design + acoustic + all
	@echo "📈  §6.2  Régression complète…"
	$(PYTHON) cli.py --regression-all \
		$(_SINGLE_FLAG) \
		$(_REFRESH_FLAG) \
		--no-check-db

regression-interactions: _require-responses ## Régression avec termes croisés  (D², D×P, S×E, D×S)
	@echo "📈  §6.2  Régression interactions…"
	$(PYTHON) cli.py --regression \
		--feature-set interactions \
		$(_SINGLE_FLAG) \
		--no-check-db

perception: _require-responses _require-run ## Alignement Ridge : espace latent → groove_mean
	@echo "🧠  §6.3  Alignement perceptif…"
	$(PYTHON) cli.py --perception $(_REFRESH_FLAG)

perc-space: _require-responses _require-run ## ICC · Mantel · géométrie locale k-NN
	@echo "🧠  §6.4  Espace perceptif…"
	$(PYTHON) cli.py --perception-space $(_REFRESH_FLAG)

# ============================================================
##@ Pipeline complet
# ============================================================

thesis: _require-metadata ## sync → new-run → analysis → regression → perception → figures
	@echo "📖  Pipeline complet du mémoire…"
	$(PYTHON) cli.py --thesis \
		$(_SINGLE_FLAG) \
		$(_REFRESH_FLAG) \
		--figures-out $(FIGURES_OUT)

figures: _require-run ## Collecte toutes les figures → FIGURES_OUT/
	@echo "🖼   Collecte → $(FIGURES_OUT)/…"
	$(PYTHON) cli.py --figures --figures-out $(FIGURES_OUT)

# ============================================================
##@ Utilitaires
# ============================================================

status: ## État du système + couverture stimuli + verrou
	$(PYTHON) cli.py --status

doctor: ## Diagnostic complet Supabase + environnement
	$(PYTHON) cli.py --doctor

install: _require-venv ## Installe les dépendances Python  (venv requis)
	pip install -r requirements.txt

setup: install env-check ## Install + vérification de l'environnement
	@echo "✔  Setup terminé"

env-check: ## Vérifie fluidsynth, ffmpeg, soundfont
	$(PYTHON) -c "from utils.env_check import run_env_check; run_env_check(strict=False)"

dry-generate: ## Simule --generate sans rien écrire
	$(PYTHON) cli.py --generate --dry-run --seed $(SEED) $(_REPEATS_FLAG)

dry-thesis: ## Simule --thesis sans rien écrire
	$(PYTHON) cli.py --thesis --dry-run $(_SINGLE_FLAG)

# ============================================================
##@ Nettoyage
# ============================================================

clean: ## Supprime TOUS les artefacts  (bloqué si verrou, FORCE=1 pour outrepasser)
	$(PYTHON) cli.py --clean all $(_FORCE_FLAG)

clean-outputs: ## Supprime MIDI + WAV + MP3 + preview  (bloqué si verrou actif)
	$(PYTHON) cli.py --clean outputs $(_FORCE_FLAG)

clean-analysis: ## Supprime les runs d'analyse et réinitialise .current_run
	$(PYTHON) cli.py --clean analysis

clean-responses: ## Supprime le cache local Supabase  (responses.csv)
	$(PYTHON) cli.py --clean responses

clean-cache: ## Supprime __pycache__ et les .pyc
	$(PYTHON) cli.py --clean cache

# ============================================================
##@ Raccourcis
# ============================================================

setup-experiment: generate lock validate ## §4 complet : génère + verrouille + valide
	@echo "✔  §4 prêt — lancez 'make serve' pour démarrer l'interface."

model: analysis regression-all perception perc-space figures ## §5–§6 complet depuis cache
	@echo "✔  §5–§6 terminés."

# ============================================================
# GARDES INTERNES (non listées dans help)
# ============================================================

_require-venv:
	@if [ "$(IN_VENV)" = "0" ]; then \
		echo "❌  Environnement virtuel Python non activé."; \
		echo "   Activez-le : source .venv/bin/activate"; \
		exit 1; \
	fi

_require-metadata:
	@test -f data/metadata.csv || \
		(echo "❌  data/metadata.csv introuvable. Lancez : make generate" && exit 1)

_require-responses:
	@test -f data/responses.csv || \
		(echo "❌  data/responses.csv introuvable. Lancez : make sync" && exit 1)

_require-run:
	@test -f .current_run || \
		(echo "❌  Aucun run courant. Lancez : make new-run" && exit 1)

_require-soundfont:
	@test -f data/soundfont/GeneralUser-GS.sf2 || \
		(echo "❌  SoundFont introuvable (data/soundfont/GeneralUser-GS.sf2)" && exit 1)

_require-mp3:
	@test -d data/mp3 && ls data/mp3/*.mp3 > /dev/null 2>&1 || \
		(echo "❌  Aucun MP3 dans data/mp3/. Lancez : make generate" && exit 1)