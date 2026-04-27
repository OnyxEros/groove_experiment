"""
config.py
=========
Single source of truth for the groove experiment system.
All paths, constants, and experiment parameters live here.
"""

from pathlib import Path
from datetime import datetime
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

ENV       = os.getenv("ENV", "dev")
DEBUG     = os.getenv("DEBUG", "1") == "1"
PORT      = int(os.getenv("PORT", "8000"))
API_HOST  = "0.0.0.0"
API_RELOAD = ENV == "dev"

# =========================================================
# DATA DIRECTORIES
# =========================================================

DATA_DIR    = BASE_DIR / "data"
MIDI_DIR    = DATA_DIR / "midi"
WAV_DIR     = DATA_DIR / "wav"
MP3_DIR     = DATA_DIR / "mp3"
PREVIEW_DIR = DATA_DIR / "preview"
ANALYSIS_DIR = DATA_DIR / "analysis"

# =========================================================
# FILES
# =========================================================

METADATA_PATH = DATA_DIR / "metadata.csv"
RESP_FILE     = DATA_DIR / "responses.csv"   # cache local Supabase

# =========================================================
# BACKEND
# =========================================================

BACKEND_DIR = BASE_DIR / "backend"
INDEX_PATH  = BACKEND_DIR / "templates" / "index.html"

# =========================================================
# AUDIO ENGINE
# =========================================================

BPM           = 90
STEPS_PER_BAR = 16          # résolution : doubles croches (16th notes)
BARS          = 4           # mesures par stimulus
SUBDIVISION   = 4           # 16th notes = 4 steps par temps
STEPS_PER_BEAT = SUBDIVISION  # alias explicite

# Profil métrique 4/4 en 16 steps
# Valeurs : force métrique de chaque position (0.2 = faible, 1.0 = temps fort)
# Justification : downbeat (1.0) > temps 3 (0.9) > contretemps (0.8/0.7) > offbeats (0.5/0.6) > doubles offbeats (0.2)
METRIC_PROFILE = np.array([
    1.0, 0.2, 0.6, 0.2,   # temps 1 et ses subdivisions
    0.8, 0.2, 0.5, 0.2,   # temps 2
    0.9, 0.2, 0.6, 0.2,   # temps 3
    0.7, 0.2, 0.5, 0.2,   # temps 4
], dtype=np.float64)

assert len(METRIC_PROFILE) == STEPS_PER_BAR, \
    f"METRIC_PROFILE doit avoir {STEPS_PER_BAR} entrées, en a {len(METRIC_PROFILE)}"

# =========================================================
# DERIVED VALUES
# =========================================================

def steps_total() -> int:
    return STEPS_PER_BAR * BARS

def step_duration_seconds() -> float:
    """Durée d'un step en secondes (cohérent avec BPM + résolution)."""
    return 60 / (BPM * (STEPS_PER_BAR / 4))

# =========================================================
# ASSETS
# =========================================================

SOUNDFONT_PATH = BASE_DIR / "soundfont.sf2"

# =========================================================
# SUPABASE
# =========================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# =========================================================
# EXPERIMENT DESIGN
# =========================================================

SEED    = 42
REPEATS = 8

S_LEVELS = [0, 1, 2]          # syncopation  : 0=métrique, 1=mixte, 2=anti-métrique
D_LEVELS = [0, 1, 2]          # density      : 0=sparse, 1=medium, 2=dense
E_LEVELS = [0.0, 0.5, 1.0]    # micro-timing : 0=quantisé, 0.5=léger, 1.0=fort

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
# HELPERS
# =========================================================

def ensure_data_dirs() -> None:
    """Crée les dossiers de données s'ils n'existent pas."""
    for d in [DATA_DIR, MIDI_DIR, WAV_DIR, MP3_DIR, ANALYSIS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def get_run_dir() -> Path:
    """Retourne un dossier d'analyse horodaté et le crée."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = ANALYSIS_DIR / f"run_{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path