# ============================================================
# Makefile — Groove Experiment
# Mémoire TSMA2 — EMC — Viken Karaboghossian
# ============================================================
#
# Pipeline du mémoire (sections) :
#   §4.1  Génération       → generate, validate, preview
#   §4.2  Collecte         → serve, sync
#   §5    Espace latent    → new-run, analysis
#   §6    Modélisation     → regression, perception, perc-space
#   -     Pipeline complet → thesis, figures
#
# Usage rapide :
#   make help           liste toutes les cibles
#   make thesis         pipeline complet depuis les données en cache
#   make generate       génère les stimuli (MIDI + MP3 + metadata.csv)
#   make serve          démarre le serveur de l'expérience
# ============================================================

.DEFAULT_GOAL := help
.PHONY: help install setup env-check \
        generate validate preview \
        serve sync \
        new-run analysis \
        regression regression-all perception perc-space \
        thesis figures \
        status doctor \
        clean clean-outputs clean-analysis clean-responses clean-cache \
        dry-generate dry-thesis

# ============================================================
# VARIABLES CONFIGURABLES
# ============================================================

PYTHON        ?= python
SEED          ?= 42
REPEATS       ?=             # vide = utilise config.py
EXCLUDE_SINGLE ?= 1          # 1 = exclut n=1, 0 = les inclut (run v1)
FEATURE_SET   ?= all         # design | acoustic | all | interactions
ANALYSIS_MODE ?= groove      # groove | full | full_with_clustering
FIGURES_OUT   ?= figures_memoire
PORT          ?= 8000

# Drapeaux conditionnels
_SINGLE_FLAG  = $(if $(filter 0,$(EXCLUDE_SINGLE)),--include-single,)
_REPEATS_FLAG = $(if $(REPEATS),--repeats $(REPEATS),)
_REFRESH_FLAG =                # passer REFRESH=1 pour forcer

# ============================================================
# AIDE
# ============================================================

help:
	@echo ""
	@echo "  ╔══════════════════════════════════════════════════════════╗"
	@echo "  ║  🎧  Groove Experiment — Makefile                       ║"
	@echo "  ║  Mémoire TSMA2  ·  Viken Karaboghossian                 ║"
	@echo "  ╚══════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  ── §4.1  Génération des stimuli ──────────────────────────"
	@echo "  make generate        Génère MIDI + MP3 + metadata.csv"
	@echo "  make validate        Figures de validation du générateur"
	@echo "  make preview         3 stimuli de référence (écoute)"
	@echo ""
	@echo "  ── §4.2  Collecte des données ────────────────────────────"
	@echo "  make serve           Démarre le serveur (port $(PORT))"
	@echo "  make sync            Supabase → data/responses.csv"
	@echo ""
	@echo "  ── §5    Espace latent ───────────────────────────────────"
	@echo "  make new-run         Crée un dossier de run horodaté"
	@echo "  make analysis        Embeddings + UMAP + clustering"
	@echo ""
	@echo "  ── §6    Modélisation statistique ───────────────────────"
	@echo "  make regression      Régression (feature set = $(FEATURE_SET))"
	@echo "  make regression-all  Régression sur design + acoustic + all"
	@echo "  make perception      Alignement Ridge espace latent → groove"
	@echo "  make perc-space      ICC · Mantel · géométrie locale k-NN"
	@echo ""
	@echo "  ── Pipeline complet ─────────────────────────────────────"
	@echo "  make thesis          sync + analysis + modèles + figures"
	@echo "  make figures         Collecte figures → $(FIGURES_OUT)/"
	@echo ""
	@echo "  ── Utilitaires ──────────────────────────────────────────"
	@echo "  make status          État du système (fichiers, runs)"
	@echo "  make doctor          Diagnostic Supabase + environnement"
	@echo "  make install         Installe les dépendances Python"
	@echo "  make env-check       Vérifie fluidsynth, ffmpeg, soundfont"
	@echo "  make dry-generate    Simule --generate sans écrire"
	@echo "  make dry-thesis      Simule --thesis sans écrire"
	@echo ""
	@echo "  ── Nettoyage ────────────────────────────────────────────"
	@echo "  make clean           Supprime tous les artefacts"
	@echo "  make clean-outputs   Supprime MIDI + WAV + MP3 + preview"
	@echo "  make clean-analysis  Supprime les runs d'analyse"
	@echo "  make clean-responses Supprime le cache local Supabase"
	@echo "  make clean-cache     Supprime __pycache__ et .pyc"
	@echo ""
	@echo "  Variables :  SEED=$(SEED)  FEATURE_SET=$(FEATURE_SET)"
	@echo "               EXCLUDE_SINGLE=$(EXCLUDE_SINGLE)  PORT=$(PORT)"
	@echo "               ANALYSIS_MODE=$(ANALYSIS_MODE)"
	@echo ""

# ============================================================
# SETUP
# ============================================================

install:
	@echo "📦  Installation des dépendances Python…"
	pip install -r requirements.txt

setup: install env-check
	@echo "✔  Setup terminé"

env-check:
	@echo "🔍  Vérification de l'environnement…"
	$(PYTHON) -c "from utils.env_check import run_env_check; run_env_check(strict=False)"

# ============================================================
# §4.1 — GÉNÉRATION DES STIMULI
# ============================================================

## Génère l'ensemble des stimuli (MIDI → MP3 → metadata.csv).
## Variables : SEED, REPEATS (optionnel), SKIP_AUDIO (1 = pas de rendu)
generate: _require-metadata-absent
	@echo "🎛️   §4.1  Génération des stimuli  (seed=$(SEED))…"
	$(PYTHON) cli.py --generate --seed $(SEED) $(_REPEATS_FLAG) \
	    $(if $(filter 1,$(SKIP_AUDIO)),--skip-audio,)

## Force la régénération même si metadata.csv existe déjà.
regenerate:
	@echo "🎛️   §4.1  Régénération forcée  (seed=$(SEED))…"
	$(PYTHON) cli.py --generate --seed $(SEED) $(_REPEATS_FLAG)

## Figures de validation structurelle du générateur (§4.1.4).
## Produit : figures/generative_validation.pdf + dataset_structure.pdf
validate: _require-run
	@echo "🔬  §4.1.4  Validation structurelle du générateur…"
	$(PYTHON) cli.py --validate

## Génère 3 stimuli de référence pour écoute comparative.
preview: _require-soundfont
	@echo "🎧  §4.1  Preview (baseline · swing · syncopated)…"
	$(PYTHON) cli.py --preview --seed $(SEED)

# ============================================================
# §4.2 — COLLECTE DES DONNÉES
# ============================================================

## Démarre le serveur FastAPI de l'expérience perceptive.
## Accessible sur http://localhost:PORT
serve:
	@echo "🌐  §4.2  Démarrage du serveur → http://localhost:$(PORT)"
	PORT=$(PORT) $(PYTHON) run_server.py

## Rapatrie les réponses Supabase → data/responses.csv
sync:
	@echo "☁️   §4.2  Synchronisation Supabase → cache local…"
	$(PYTHON) cli.py --sync

## Sync forcé (ignore le cache local même récent)
sync-refresh:
	@echo "☁️   §4.2  Sync forcé (--refresh)…"
	$(PYTHON) cli.py --sync --refresh

# ============================================================
# §5 — ESPACE LATENT
# ============================================================

## Crée un dossier de run horodaté dans data/analysis/
new-run:
	@echo "📁  §5  Nouveau run d'analyse…"
	$(PYTHON) cli.py --new-run

## Pipeline d'analyse complet.
## Mode : ANALYSIS_MODE (défaut : groove)
## Produit : embeddings/ + clustering/ + figures/ + stim_id_map.json
analysis: _require-run _require-metadata
	@echo "🧠  §5  Analyse de l'espace latent  (mode=$(ANALYSIS_MODE))…"
	$(PYTHON) cli.py --analysis --analysis-mode $(ANALYSIS_MODE)

## Crée un run ET lance l'analyse immédiatement.
new-run-analysis: _require-metadata
	@echo "📁🧠  §5  Nouveau run + analyse…"
	$(PYTHON) cli.py --new-run
	$(PYTHON) cli.py --analysis --analysis-mode $(ANALYSIS_MODE)

# ============================================================
# §6 — MODÉLISATION STATISTIQUE
# ============================================================

## Régression groove sur un feature set.
## Feature set : FEATURE_SET (défaut : all)
## Exclut les stimuli n=1 sauf si EXCLUDE_SINGLE=0
regression: _require-responses
	@echo "📈  §6.2  Régression groove  (features=$(FEATURE_SET))…"
	$(PYTHON) cli.py --regression \
	    --feature-set $(FEATURE_SET) \
	    $(_SINGLE_FLAG) \
	    $(if $(filter 1,$(REFRESH)),--refresh,) \
	    --no-check-db

## Régression sur les 3 feature sets (design + acoustic + all).
regression-all: _require-responses
	@echo "📈  §6.2  Régression complète (design · acoustic · all)…"
	$(PYTHON) cli.py --regression-all \
	    $(_SINGLE_FLAG) \
	    $(if $(filter 1,$(REFRESH)),--refresh,) \
	    --no-check-db

## Régression avec les termes d'interaction (D², D×P, S×E, D×S).
regression-interactions: _require-responses
	@echo "📈  §6.2  Régression avec interactions…"
	$(PYTHON) cli.py --regression \
	    --feature-set interactions \
	    $(_SINGLE_FLAG) \
	    --no-check-db

## Alignement Ridge : espace latent acoustique → groove_mean.
perception: _require-responses _require-run
	@echo "🧠  §6.3  Alignement perceptif…"
	$(PYTHON) cli.py --perception \
	    $(if $(filter 1,$(REFRESH)),--refresh,)

## ICC · Test de Mantel · géométrie locale k-NN.
## Produit les figures de l'espace perceptif dans le run courant.
perc-space: _require-responses _require-run
	@echo "🧠  §6.4  Espace perceptif (ICC · Mantel · k-NN)…"
	$(PYTHON) cli.py --perception-space \
	    $(if $(filter 1,$(REFRESH)),--refresh,)

# ============================================================
# PIPELINE COMPLET DU MÉMOIRE
# ============================================================

## Pipeline thèse complet :
## sync → new-run → analysis → regression-all → perception → perc-space → figures
thesis: _require-metadata
	@echo "📖  Pipeline complet du mémoire…"
	$(PYTHON) cli.py --thesis \
	    $(_SINGLE_FLAG) \
	    $(if $(filter 1,$(REFRESH)),--refresh,) \
	    --figures-out $(FIGURES_OUT)

## Collecte toutes les figures générées → FIGURES_OUT/
figures: _require-run
	@echo "🖼   Collecte des figures → $(FIGURES_OUT)/…"
	$(PYTHON) cli.py --figures --figures-out $(FIGURES_OUT)

# ============================================================
# UTILITAIRES
# ============================================================

status:
	@echo "📊  État du système…"
	$(PYTHON) cli.py --status

doctor:
	@echo "🩺  Diagnostics complets…"
	$(PYTHON) cli.py --doctor

## Simule --generate sans rien écrire (vérifie la configuration)
dry-generate:
	$(PYTHON) cli.py --generate --dry-run --seed $(SEED) $(_REPEATS_FLAG)

## Simule --thesis sans rien écrire
dry-thesis:
	$(PYTHON) cli.py --thesis --dry-run $(_SINGLE_FLAG)

# ============================================================
# NETTOYAGE
# ============================================================

## Supprime TOUS les artefacts générés (MIDI, MP3, analyses, cache)
clean:
	@echo "🧹  Nettoyage complet…"
	$(PYTHON) cli.py --clean all

## Supprime les fichiers audio générés (MIDI, WAV, MP3, preview)
clean-outputs:
	@echo "🧹  Suppression des fichiers audio…"
	$(PYTHON) cli.py --clean outputs

## Supprime les runs d'analyse et réinitialise .current_run
clean-analysis:
	@echo "🧹  Suppression des runs d'analyse…"
	$(PYTHON) cli.py --clean analysis

## Supprime le cache local Supabase (responses.csv)
clean-responses:
	@echo "🧹  Suppression du cache Supabase local…"
	$(PYTHON) cli.py --clean responses

## Supprime __pycache__ et les .pyc
clean-cache:
	@echo "🧹  Suppression des caches Python…"
	$(PYTHON) cli.py --clean cache

# ============================================================
# GARDES (cibles internes de vérification)
# ============================================================

_require-metadata:
	@test -f data/metadata.csv || \
	    (echo "❌  data/metadata.csv introuvable. Lancez : make generate" && exit 1)

_require-metadata-absent:
	@if [ -f data/metadata.csv ]; then \
	    echo "ℹ️   data/metadata.csv existe déjà. Utilisez 'make regenerate' pour forcer."; \
	fi

_require-responses:
	@test -f data/responses.csv || \
	    (echo "❌  data/responses.csv introuvable. Lancez : make sync" && exit 1)

_require-run:
	@test -f .current_run || \
	    (echo "❌  Aucun run courant. Lancez : make new-run" && exit 1)

_require-soundfont:
	@test -f data/soundfont/GeneralUser-GS.sf2 || \
	    (echo "❌  SoundFont introuvable (data/soundfont/GeneralUser-GS.sf2)" && exit 1)

# ============================================================
# RACCOURCIS FRÉQUENTS
# ============================================================

## Workflow §4 complet : génère les stimuli ET lance le serveur
setup-experiment: generate validate
	@echo "✔  §4 prêt. Lancez 'make serve' pour démarrer l'interface."

## Workflow §5–§6 complet depuis un cache existant (sans sync réseau)
model: analysis regression-all perception perc-space figures
	@echo "✔  §5–§6 terminés."