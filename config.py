"""
config.py
=========
Source unique de vérité pour le système Groove Experiment.

Sections :
    Environnement         — variables d'environnement et chemins
    Structure temporelle  — BPM, résolution, durée du stimulus
    Profil métrique       — poids de chaque position dans la mesure
    Hi-hat                — paramètres de génération stochastique
    Micro-timing          — paramètres du jitter expressif
    Hiérarchie des voix   — pondération perceptive kick/snare/hihat
    Push/pull inter-voix  — paramètre P (Keil 1995)
    Design expérimental   — niveaux des variables manipulées
    Analyse               — UMAP, clustering
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
    0: 0.30,
    1: 0.50,
    2: 0.70,
}

HIHAT_PROB_MIN = 0.01
HIHAT_PROB_MAX = 0.90


# =========================================================
# BASSE — LIGNE MÉLODIQUE (ancrage métrique, non expérimental)
# =========================================================

BASS_PITCH = 36   # C2 — root note

# Motif rythmique sur 16 steps (1 mesure) — répété sur les 6 mesures
# Positions : 0=temps1, 4=temps2, 8=temps3, 12=temps4 (en doubles croches)
# 1  = hit normal  |  0.4 = ghost note  |  0 = silence
BASS_PATTERN_BAR: list[float] = [
    1.0, 0.0, 0.0, 0.0,   # temps 1 — root, ancrage fort
    0.4, 0.0, 0.0, 0.0,   # temps 2 — ghost note, très légère
    1.0, 0.0, 0.0, 0.0,   # temps 3 — quinte, ancrage secondaire
    0.0, 0.0, 0.4, 0.0,   # temps 4 — note d'approche sur le "e" du 4
]

# Intervalles en demi-tons depuis BASS_PITCH, indexés par position dans la barre
# Même longueur que BASS_PATTERN_BAR — lu uniquement sur les hits
BASS_INTERVAL_BAR: list[int] = [
    0,  0,  0,  0,   # temps 1 → root (C2)
    0,  0,  0,  0,   # temps 2 → ghost sur root
    7,  0,  0,  0,   # temps 3 → quinte (G2)
    0,  0, 11,  0,   # temps 4 → note d'approche (B1, sensible, tension vers C)
]

# Vélocités de base par position (0–127), indexées comme BASS_PATTERN_BAR
# Les ghost notes ont une vélocité faible indépendamment de E
BASS_VELOCITY_BAR: list[int] = [
    75,  0,  0,  0,   # temps 1 — fort
    25,  0,  0,  0,   # temps 2 — ghost, très doux
    55,  0,  0,  0,   # temps 3 — moyen
     0,  0, 35,  0,   # temps 4 — note d'approche, doux
]

# Durée des notes en fraction de step_duration
# Les ghost notes sont courtes (staccato), les notes d'ancrage sont longues (sustain)
BASS_DURATION_BAR: list[float] = [
    2.2, 0.0, 0.0, 0.0,   # temps 1 — sustain jusqu'au ghost
    0.6, 0.0, 0.0, 0.0,   # temps 2 — ghost court
    1.8, 0.0, 0.0, 0.0,   # temps 3 — sustain jusqu'à l'approche
    0.0, 0.0, 0.8, 0.0,   # temps 4 — note d'approche courte
]

BASS_VELOCITY     = 85    # vélocité par défaut (fallback)
BASS_TIMING_SCALE = 0.20
BASS_VOICE_WEIGHT = 0.20

BASS_ANTICIPATION_RATIO:   float = -0.06   # légère avance (6% du step ≈ 5ms)
BASS_HUMANIZE_NOISE_RATIO: float = 0.03    # bruit résiduel (3% ≈ 2.5ms σ)

# =========================================================
# MICRO-TIMING — JITTER EXPRESSIF
# =========================================================

SWING_BASELINE = 0.04
"""
Swing incompressible appliqué à toutes les conditions, y compris E=0.

Rationale :
    Un batteur humain ne joue jamais à déviation zéro — même en jouant
    "straight", il produit un micro-swing naturel (Keil 1995).
    La grille MIDI parfaite n'est pas une référence perceptive humaine.

    E=0 représente donc "tight / peu expressif", pas "robot quantisé".
    Ce swing baseline garantit que toutes les conditions sonnent musicales,
    ce qui est une condition nécessaire pour que les participants puissent
    évaluer le groove plutôt que réagir à la mécanique du son.

    Valeur : 4% du step_duration ≈ 3ms à 90 BPM.
    En dessous du seuil de détection comme déviation isolée (~6ms),
    mais suffisant pour humaniser le feel global du pattern.

    Swing total = SWING_BASELINE + SWING_MAX_RATIO × E
    E=0   → 4%  du step ≈  3ms  (tight, humanisé)
    E=0.5 → 10% du step ≈  8ms  (groove modéré)
    E=1   → 16% du step ≈ 12ms  (groove expressif)
"""

SWING_MAX_RATIO = 0.12
"""
Amplitude additionnelle du swing contrôlée par E.
Swing total = SWING_BASELINE + SWING_MAX_RATIO × E × amount.
"""

DRIFT_MAX_RATIO = 0.10
"""Amplitude max du drift sinusoïdal en fraction du step_duration."""

NOISE_MAX_RATIO = 0.10
"""Écart-type max du bruit gaussien corrélé en fraction du step_duration."""

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

PUSH_MAX_RATIO = 0.18
"""
Décalage systématique maximal du hihat en fraction du step_duration.
≈ 14ms à 90 BPM.
P > 0 : hihat en avance (rushing)
P < 0 : hihat en retard (laid-back)
"""

# =========================================================
# DESIGN EXPÉRIMENTAL
# =========================================================

# Graine maître — reproductibilité globale
SEED = 42

# Répétitions par phase — différenciées pour équilibrer P1/P2 vs P3
REPEATS_P1 = 5
REPEATS_P2 = 4
REPEATS_P3 = 1
# Total : 128 stimuli

# Alias plat — utilisé par print_config_summary
REPEATS = REPEATS_P3

S_LEVELS = [0, 1, 2]
D_LEVELS = [0, 1, 2]
E_LEVELS = [0.0, 0.5, 1.0]
P_LEVELS = [-1, 0, 1]

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
    """S_mv → alpha continu [0, 1]. 0 = métrique, 1 = anti-métrique."""
    max_level = max(S_LEVELS)
    return sync_level / max_level if max_level > 0 else 0.0

def push_from_p_level(p_level: int) -> float:
    """P_level → décalage en fraction de step_duration."""
    max_level = max(abs(p) for p in P_LEVELS) if P_LEVELS else 1
    return (p_level / max_level) * PUSH_MAX_RATIO if max_level > 0 else 0.0

# =========================================================
# HELPERS
# =========================================================

_CURRENT_RUN_FILE = BASE_DIR / ".current_run"

def new_run() -> Path:
    """Crée un nouveau run et l'enregistre comme run courant."""
    path = ANALYSIS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    path.mkdir(parents=True, exist_ok=True)
    _CURRENT_RUN_FILE.write_text(str(path))
    print(f"[run] new run → {path}")
    return path

def get_current_run() -> Path:
    """Retourne le run courant (erreur claire si pas initialisé)."""
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
    """Retourne le run le plus récent, ou None si aucun."""
    if not ANALYSIS_DIR.exists():
        return None
    runs = sorted(ANALYSIS_DIR.glob("run_*"))
    return runs[-1] if runs else None

def print_config_summary() -> None:
    n_conditions = len(S_LEVELS) * len(D_LEVELS) * len(E_LEVELS) * len(P_LEVELS)
    sd_ms        = step_duration_seconds() * 1000
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
    print(f"  S_LEVELS               : {S_LEVELS}  (syncopation)")
    print(f"  D_LEVELS               : {D_LEVELS}  (densité)")
    print(f"  E_LEVELS               : {E_LEVELS}  (micro-timing)")
    print(f"  P_LEVELS               : {P_LEVELS}  (push/pull inter-voix)")
    print(f"  Conditions factorielles: {n_conditions}")
    print(f"  Répétitions/condition  : {REPEATS}")
    print()
    print(f"  Swing baseline         : {SWING_BASELINE*100:.0f}% du step"
          f" ≈ {SWING_BASELINE*sd_ms:.1f}ms  (E=0, toutes conditions)")
    print(f"  Swing max (E=1)        : {(SWING_BASELINE+SWING_MAX_RATIO)*100:.0f}%"
          f" ≈ {(SWING_BASELINE+SWING_MAX_RATIO)*sd_ms:.1f}ms")
    print(f"  Drift max              : {DRIFT_MAX_RATIO*100:.0f}% du step")
    print(f"  Noise max (σ)          : {NOISE_MAX_RATIO*100:.0f}% du step")
    print(f"  Push max               : {PUSH_MAX_RATIO*100:.0f}% du step")
    print("=" * 62 + "\n")