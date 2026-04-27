from pathlib import Path
import os
from datetime import datetime

# =========================================================
# PROJECT ROOT
# =========================================================

BASE_DIR = Path(__file__).resolve().parent


# =========================================================
# ENVIRONMENT
# =========================================================

ENV = os.getenv("ENV", "dev")
DEBUG = os.getenv("DEBUG", "1") == "1"
PORT = int(os.getenv("PORT", "8000"))

API_HOST = "0.0.0.0"
API_RELOAD = ENV == "dev"


# =========================================================
# DATA ROOT
# =========================================================
DATA_DIR = BASE_DIR / "data"

MIDI_DIR = DATA_DIR / "midi"
WAV_DIR = DATA_DIR / "wav"
MP3_DIR = DATA_DIR / "mp3"
PREVIEW_DIR = DATA_DIR / "preview"

ANALYSIS_DIR = DATA_DIR / "analysis"


def get_run_dir():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = ANALYSIS_DIR / f"run_{timestamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


# =========================================================
# FILES
# =========================================================

METADATA_PATH = DATA_DIR / "metadata.csv"
RESP_FILE = DATA_DIR / "responses.csv"


# =========================================================
# BACKEND
# =========================================================

BACKEND_DIR = BASE_DIR / "backend"
INDEX_PATH = BACKEND_DIR / "templates" / "index.html"


# =========================================================
# AUDIO ENGINE (GROUND TRUTH)
# =========================================================

BPM = 90

# résolution musicale (grille principale)
STEPS_PER_BAR = 16

# nombre de mesures par stimulus
BARS = 4

# subdivision logique (notation musicale, PAS la grille brute)
SUBDIVISION = 4  # 16th notes = 4 steps per beat

# alias explicite (évite les confusions)
STEPS_PER_BEAT = SUBDIVISION


# =========================================================
# DERIVED VALUES (single source of truth)
# =========================================================

def steps_total():
    return STEPS_PER_BAR * BARS


def step_duration_seconds():
    """
    durée d’un step en secondes
    cohérent avec BPM + résolution de la grille
    """
    return 60 / (BPM * (STEPS_PER_BAR / 4))


# =========================================================
# AUDIO ENGINE ASSETS
# =========================================================

SOUNDFONT_PATH = BASE_DIR / "soundfont.sf2"


# =========================================================
# SUPABASE (CLOUD DB)
# =========================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


# =========================================================
# HELPERS
# =========================================================

def ensure_data_dirs():
    for d in [DATA_DIR, MIDI_DIR, WAV_DIR, MP3_DIR, ANALYSIS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# =========================================================
# EXPERIMENT CONFIG (IMPORTANT FIX)
# =========================================================

SEED = 42

# default, mais overrideable par CLI / run_experiment
REPEATS = 8

# factors expérimentaux
S_LEVELS = [0, 1, 2]      # syncopation / swing / structure
D_LEVELS = [0, 1, 2]      # density
E_LEVELS = [0.0, 0.5, 1.0]  # micro-timing energy


# =========================================================
# UMAP CONFIG
# =========================================================

UMAP_CONFIG = {
    "n_components": 3,
    "n_neighbors": 25,
    "min_dist": 0.08,
    "metric": "cosine",
    "random_state": 42,
}