from pathlib import Path
import os

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
ANALYSIS_DIR = DATA_DIR / "analysis"

# =========================================================
# FILES
# =========================================================

METADATA_PATH = DATA_DIR / "metadata.csv"
RESP_FILE = DATA_DIR / "responses.csv"


# =========================================================
# AUDIO ENGINE
# =========================================================

BPM = 120
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