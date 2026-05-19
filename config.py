"""
config.py
=========
Source unique de vérité pour le système Groove Experiment.

Notation :
    Paramètres génératifs (manipulés) : suffixe _mv  → S_mv, D_mv, E_mv, P_mv
    Descripteurs émergents (réalisés)  : sans indice  → D, I, V, S, E, P
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os

import numpy as np
from dotenv import load_dotenv

# =========================================================
# PROJECT ROOT
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()

ENV        = os.getenv("ENV", "dev")
DEBUG      = os.getenv("DEBUG", "1") == "1"
PORT       = int(os.getenv("PORT", "8000"))
API_HOST   = "0.0.0.0"
API_RELOAD = ENV == "dev"

# =========================================================
# DATA DIRECTORIES & FILES
# =========================================================

DATA_DIR     = BASE_DIR / "data"
MIDI_DIR     = DATA_DIR / "midi"
WAV_DIR      = DATA_DIR / "wav"
MP3_DIR      = DATA_DIR / "mp3"
PREVIEW_DIR  = DATA_DIR / "preview"
ANALYSIS_DIR = DATA_DIR / "analysis"

METADATA_PATH = DATA_DIR / "metadata.csv"
RESP_FILE     = DATA_DIR / "responses.csv"

BACKEND_DIR    = BASE_DIR / "backend"
INDEX_PATH     = BACKEND_DIR / "templates" / "index.html"
SOUNDFONT_PATH = DATA_DIR / "soundfont" / "GeneralUser-GS.sf2"

# =========================================================
# SUPABASE
# =========================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# =========================================================
# STRUCTURE TEMPORELLE
# =========================================================

BPM           = 90
STEPS_PER_BAR = 16
TOTAL_BARS    = 6
LOOP_BARS     = 2

assert TOTAL_BARS % LOOP_BARS == 0, (
    f"LOOP_BARS ({LOOP_BARS}) doit être un diviseur de TOTAL_BARS ({TOTAL_BARS})."
)

N_LOOPS = TOTAL_BARS // LOOP_BARS

# =========================================================
# PROFIL MÉTRIQUE
# =========================================================

METRIC_PROFILE = np.array([
    1.0, 0.2, 0.6, 0.2,   # temps 1
    0.8, 0.2, 0.5, 0.2,   # temps 2
    0.9, 0.2, 0.6, 0.2,   # temps 3
    0.7, 0.2, 0.5, 0.2,   # temps 4
], dtype=np.float64)

SYNCOPATION_STRONG_THRESHOLD = 0.5

assert len(METRIC_PROFILE) == STEPS_PER_BAR

# =========================================================
# HI-HAT — GÉNÉRATION STOCHASTIQUE
# =========================================================

HIHAT_DENSITY_PROBS: dict[int, float] = {
    0: 0.15,   # était 0.30 — franchement sparse, contraste clair avec niveau 1
    1: 0.55,   # était 0.50 — légèrement augmenté
    2: 0.85,   # était 0.70 — franchement dense, proche du 16ths constant
}

HIHAT_PROB_MIN = 0.01
HIHAT_PROB_MAX = 0.90

# =========================================================
# BASSE — LIGNE MÉLODIQUE
# =========================================================

BASS_PITCH = 36

BASS_PATTERN_BAR: list[float] = [
    1.0, 0.0, 0.0, 0.0,
    0.4, 0.0, 0.0, 0.0,
    1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.4, 0.0,
]

BASS_INTERVAL_BAR: list[int] = [
    0,  0,  0,  0,
    0,  0,  0,  0,
    7,  0,  0,  0,
    0,  0, 11,  0,
]

BASS_VELOCITY_BAR: list[int] = [
    75,  0,  0,  0,
    25,  0,  0,  0,
    55,  0,  0,  0,
     0,  0, 35,  0,
]

BASS_DURATION_BAR: list[float] = [
    2.2, 0.0, 0.0, 0.0,
    0.6, 0.0, 0.0, 0.0,
    1.8, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.8, 0.0,
]

BASS_VELOCITY     = 85
BASS_TIMING_SCALE = 0.20
BASS_VOICE_WEIGHT = 0.20

BASS_ANTICIPATION_RATIO:   float = -0.06
BASS_HUMANIZE_NOISE_RATIO: float = 0.03

# =========================================================
# MICRO-TIMING — JITTER EXPRESSIF
#
# Calibration v2 :
#   Les valeurs initiales (SWING_MAX_RATIO=0.12, DRIFT=0.10, NOISE=0.10)
#   produisaient un swing maximal de ~8ms à 90bpm, en-dessous du seuil
#   de détection documenté pour des non-musiciens (~10–15ms,
#   Honing & Ladinig 2009 ; Madison 2006).
#
#   Les valeurs ci-dessous sont recalibrées pour rester dans les bornes
#   réalistes de la batterie humaine tout en restant perceptibles :
#     E_mv=0   → swing baseline ~2.6ms  (tight, quasi-robotique)
#     E_mv=0.5 → swing total    ~9.2ms  (légèrement swingué)
#     E_mv=1.0 → swing total    ~15.5ms (swing jazz/funk audible)
#   Ces valeurs correspondent aux plages mesurées par Frane & Shams (2017)
#   sur des batteurs humains en contexte funk.
# =========================================================

SWING_BASELINE  = 0.04   # 4% du step — inchangé (E_mv=0 reste "serré")
SWING_MAX_RATIO = 0.20   # était 0.12 → swing max ~13ms à 90bpm
DRIFT_MAX_RATIO = 0.15   # était 0.10
NOISE_MAX_RATIO = 0.14   # était 0.10

# =========================================================
# HUMANISATION DE LA VÉLOCITÉ — liée à E_mv
#
# E_mv encode l'amplitude des déviations expressives.
# Cette définition est étendue à la dynamique (vélocité) en cohérence
# avec Gabrielsson (1999) : timing et vélocité sont les deux composantes
# principales du geste expressif en percussion.
#
#   E_mv=0   → vélocités fixes (pas de fluctuation)
#   E_mv=0.5 → sigma ~4pts MIDI  (~5%)
#   E_mv=1.0 → sigma ~8pts MIDI  (~10%) — en-dessous du seuil
#              de quantification perceptive mais perceptible globalement
# =========================================================

VELOCITY_HUMANIZE_SIGMA = 8.0   # sigma de base à E_mv=1.0 (pts MIDI)

# =========================================================
# HIÉRARCHIE PERCEPTIVE DES VOIX
# =========================================================

KICK_TIMING_SCALE  = 0.20
SNARE_TIMING_SCALE = 0.30
HIHAT_TIMING_SCALE = 1.00

KICK_VOICE_WEIGHT  = 0.30
SNARE_VOICE_WEIGHT = 0.50
HIHAT_VOICE_WEIGHT = 1.00

KICK_DENSITY_WEIGHT  = 0.20
SNARE_DENSITY_WEIGHT = 0.20
HIHAT_DENSITY_WEIGHT = 0.60

assert abs(KICK_DENSITY_WEIGHT + SNARE_DENSITY_WEIGHT + HIHAT_DENSITY_WEIGHT - 1.0) < 1e-9

# =========================================================
# PUSH/PULL INTER-VOIX (Keil 1995)
# =========================================================

PUSH_MAX_RATIO = 0.11

# =========================================================
# PAD HARMONIQUE INVARIANT
#
# Un accord Am7 tenu (vélocité faible, program=89 "Pad 2 warm")
# est ajouté comme fond harmonique constant sur tous les stimuli.
# Étant strictement invariant entre conditions, il ne confond
# aucune variable manipulée — même logique que la basse (cf. §2.2.2).
#
# Objectif : ancrer perceptivement les variations du hi-hat pour
# les participants sans culture musicale formelle.
# Référence : Witek et al. (2014) utilisent un fond harmonique stable
# dans leurs stimuli de groove.
# =========================================================

PAD_ENABLED    = True
PAD_PROGRAM    = 89          # General MIDI "Pad 2 (warm)"
PAD_VELOCITY   = 42          # discret, ne masque pas les percussions
PAD_PITCHES    = [45, 52, 55, 59]  # Am7 : A2, E3, G3, B3
PAD_CHANNEL    = 1           # canal séparé des percussions

# =========================================================
# DESIGN EXPÉRIMENTAL
# =========================================================

SEED = 42

REPEATS_P1 = 5
REPEATS_P2 = 4
REPEATS_P3 = 1

REPEATS = REPEATS_P3

S_mv_LEVELS = [0, 1, 2]
D_mv_LEVELS = [0, 1, 2]
E_mv_LEVELS = [0.0, 0.5, 1.0]
P_mv_LEVELS = [-1, 0, 1]

# Alias rétro-compatibilité (à supprimer progressivement)
S_LEVELS = S_mv_LEVELS
D_LEVELS = D_mv_LEVELS
E_LEVELS = E_mv_LEVELS
P_LEVELS = P_mv_LEVELS

# =========================================================
# UMAP
# =========================================================

UMAP_CONFIG = {
    "n_components":  3,
    "n_neighbors":   25,
    "min_dist":      0.08,
    "metric":        "cosine",
    "random_state":  SEED,
}


# =========================================================
# SEUILS RT — SOURCE DE VÉRITÉ UNIQUE
# Justification RT_MIN : stimulus ~6.7s à 90bpm.
# Un participant attentif peut répondre dès 1.5s d'écoute.
# 4.0s filtrait ~15% des réponses rapides légitimes (biais sélection).
# Référence : Madison (2006) seuil 1s pour stimuli rythmiques comparables.
# =========================================================
RT_MIN_S = 1.5
RT_MAX_S = 600.0


# =========================================================
# DERIVED VALUES
# =========================================================

def loop_steps() -> int:
    return STEPS_PER_BAR * LOOP_BARS

def total_steps() -> int:
    return STEPS_PER_BAR * TOTAL_BARS

def step_duration_seconds() -> float:
    return 60.0 / (BPM * (STEPS_PER_BAR / 4))

def stimulus_duration_seconds() -> float:
    return total_steps() * step_duration_seconds()

def alpha_from_sync_level(sync_level: int) -> float:
    max_level = max(S_mv_LEVELS)
    return sync_level / max_level if max_level > 0 else 0.0

def push_from_p_level(p_level: int) -> float:
    max_level = max(abs(p) for p in P_mv_LEVELS) if P_mv_LEVELS else 1
    return (p_level / max_level) * PUSH_MAX_RATIO if max_level > 0 else 0.0

# =========================================================
# HELPERS
# =========================================================

_CURRENT_RUN_FILE = BASE_DIR / ".current_run"

def new_run() -> Path:
    path = ANALYSIS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    path.mkdir(parents=True, exist_ok=True)
    _CURRENT_RUN_FILE.write_text(str(path))
    print(f"[run] new run → {path}")
    return path

def get_current_run() -> Path:
    if not _CURRENT_RUN_FILE.exists():
        raise RuntimeError(
            "Aucun run courant — lance d'abord : make new-run"
        )
    path = Path(_CURRENT_RUN_FILE.read_text().strip())
    if not path.exists():
        raise RuntimeError(
            f"Run introuvable : {path}\n"
            "Lance : make new-run"
        )
    return path

def ensure_data_dirs() -> None:
    for d in [DATA_DIR, MIDI_DIR, WAV_DIR, MP3_DIR, ANALYSIS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def get_run_dir() -> Path:
    path = ANALYSIS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_latest_run_dir() -> Path | None:
    if not ANALYSIS_DIR.exists():
        return None
    runs = sorted(ANALYSIS_DIR.glob("run_*"))
    return runs[-1] if runs else None

def print_config_summary() -> None:
    n_conditions = len(S_mv_LEVELS) * len(D_mv_LEVELS) * len(E_mv_LEVELS) * len(P_mv_LEVELS)
    sd_ms        = step_duration_seconds() * 1000
    swing_min_ms = SWING_BASELINE * sd_ms
    swing_max_ms = (SWING_BASELINE + SWING_MAX_RATIO) * sd_ms
    print("\n" + "=" * 62)
    print("  CONFIGURATION — Groove Experiment")
    print("=" * 62)
    print(f"  Tempo                  : {BPM} BPM")
    print(f"  Résolution             : {STEPS_PER_BAR} steps/bar  (16th notes)")
    print(f"  Durée d'un step        : {sd_ms:.1f} ms")
    print()
    print(f"  Stimulus               : {TOTAL_BARS} mesures  ({total_steps()} steps)")
    print(f"  Durée stimulus         : {stimulus_duration_seconds():.1f} s")
    print(f"  Boucle rythmique       : {LOOP_BARS} mesures  ({loop_steps()} steps)")
    print(f"  Répétitions de boucle  : {N_LOOPS}×")
    print()
    print(f"  S_mv_LEVELS            : {S_mv_LEVELS}  (syncopation)")
    print(f"  D_mv_LEVELS            : {D_mv_LEVELS}  (densité)")
    print(f"  E_mv_LEVELS            : {E_mv_LEVELS}  (micro-timing)")
    print(f"  P_mv_LEVELS            : {P_mv_LEVELS}  (push/pull inter-voix)")
    print(f"  Conditions factorielles: {n_conditions}")
    print(f"  Répétitions/condition  : {REPEATS}")
    print()
    print(f"  Swing baseline (E_mv=0): ~{swing_min_ms:.1f}ms")
    print(f"  Swing max  (E_mv=1.0)  : ~{swing_max_ms:.1f}ms  ← seuil de détection ~10–15ms")
    print(f"  Drift max              : {DRIFT_MAX_RATIO*100:.0f}% du step")
    print(f"  Noise max (σ)          : {NOISE_MAX_RATIO*100:.0f}% du step")
    print(f"  Vélocité humanisée σ   : {VELOCITY_HUMANIZE_SIGMA:.0f} pts MIDI (à E_mv=1)")
    print()
    print(f"  Pad harmonique         : {'activé' if PAD_ENABLED else 'désactivé'}")
    if PAD_ENABLED:
        print(f"    Program MIDI         : {PAD_PROGRAM} (Pad 2 warm)")
        print(f"    Vélocité             : {PAD_VELOCITY}")
        print(f"    Accord               : Am7 {PAD_PITCHES}")
    print("=" * 62 + "\n")