import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from config import MP3_DIR, METADATA_PATH, INDEX_PATH
from backend.models import Response
from infra.supabase_client import insert_response

# 👉 NEW: design system import (safe fallback)
try:
    from backend.design.registry import StimulusRegistry
    DESIGN_MODE = True
except Exception:
    DESIGN_MODE = False


# =========================================================
# APP
# =========================================================

app = FastAPI()


# =========================================================
# STARTUP
# =========================================================

@app.on_event("startup")
def startup():

    df = pd.read_csv(METADATA_PATH)

    if df.empty:
        raise RuntimeError("metadata.csv is empty")

    if "mp3_path" not in df.columns:
        raise RuntimeError("Missing column: mp3_path")

    df["audio_file"] = df["mp3_path"].apply(lambda p: Path(p).name)

    app.state.df_global = df

    # =====================================================
    # NEW: experimental design pre-generation
    # =====================================================
    if DESIGN_MODE:
        registry = StimulusRegistry()
        stimuli = registry.build_stimuli(n_variants=3, seed=42)

        if len(stimuli) > 0:
            app.state.stimuli = stimuli
            print(f"🎧 Design system active → {len(stimuli)} stimuli")
        else:
            print("⚠️ Design system empty → fallback to dataframe sampling")
            app.state.stimuli = None
    else:
        app.state.stimuli = None


# =========================================================
# STATIC FILES
# =========================================================

app.mount("/audio", StaticFiles(directory=str(MP3_DIR)), name="audio")
app.mount("/static", StaticFiles(directory="backend/static"), name="static")


# =========================================================
# PARTICIPANT ID
# =========================================================

@app.get("/new_participant")
def new_participant():
    return {
        "participant_id": uuid.uuid4().hex[:8],
        "timestamp": datetime.utcnow().isoformat()
    }


# =========================================================
# STIMULI (UPDATED)
# =========================================================

@app.get("/stimuli")
def get_stimuli(n: int = 24):

    import random

    # =====================================================
    # MODE 1: DESIGN CONTROLLED (NEW)
    # =====================================================
    if hasattr(app.state, "stimuli") and app.state.stimuli is not None:
        stimuli = app.state.stimuli

        n = min(n, len(stimuli))
        sample = stimuli[:n]   # deterministic order (can shuffle if needed)
        random.shuffle(sample)

        for s in sample:
            if "mp3_path" in s:
                s["audio_url"] = f"/audio/{Path(s['mp3_path']).name}"

        return sample

    # =====================================================
    # MODE 2: FALLBACK (OLD BEHAVIOR)
    # =====================================================
    df = app.state.df_global
    n = min(n, len(df))

    sample = df.sample(n).copy()

    sample["audio_url"] = sample["audio_file"].apply(
        lambda f: f"/audio/{f}"
    )

    sample = sample.drop(columns=["mp3_path"], errors="ignore")

    return sample.to_dict(orient="records")


# =========================================================
# RESPONSE → SUPABASE
# =========================================================

@app.post("/response")
def save_response(resp: Response):

    row = resp.model_dump()

    clean_row = {
        "participant_id": row["participant_id"],
        "stim_id": row["stim_id"],
        "groove": row.get("groove"),
        "complexity": row.get("complexity"),
        "rt": row.get("rt"),
        "created_at": datetime.utcnow().isoformat()
    }

    try:
        insert_response(clean_row)

    except Exception as e:
        print("⚠️ Supabase error:", e)
        return {"status": "error", "detail": str(e)}

    return {"status": "ok"}


# =========================================================
# FRONTEND
# =========================================================

@app.get("/", response_class=HTMLResponse)
def home():
    return INDEX_PATH.read_text()