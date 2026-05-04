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
        sample = list(stimuli[:n])  # copie — ne pas muter app.state.stimuli
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

    if "stim_id" not in sample.columns and "id" in sample.columns:
        sample["stim_id"] = sample["id"].apply(lambda i: f"stim_{int(i):04d}")

    sample["audio_url"] = sample["audio_file"].apply(
        lambda f: f"/audio/{f}"
    )

    sample = sample.drop(columns=["mp3_path"], errors="ignore")

    return sample.to_dict(orient="records")




# =========================================================
# STIMULI EXAMPLE
# =========================================================
 
@app.get("/example")
def get_example():
    """
    Retourne l'URL audio du stimulus le plus groove du dataset.
    Critère : S_mv=2, D_mv=2, E=1.0 (anti-métrique, dense, micro-timing fort).
    Fallback : le stimulus avec le S_real le plus élevé.
    """
    df = app.state.df_global
 
    # Cherche le candidat idéal
    mask = (df["S_mv"] == 2) & (df["D_mv"] == 2) & (df["E"] == 1.0)
    candidates = df[mask]
 
    if candidates.empty:
        # Fallback : meilleur S_real global
        row = df.loc[df["S_real"].idxmax()]
    else:
        # Parmi les candidats, prend celui avec S_real le plus élevé
        row = candidates.loc[candidates["S_real"].idxmax()]
 
    audio_file = row["audio_file"] if "audio_file" in row else f"stim_{int(row['id']):04d}.mp3"
 
    return {
        "audio_url": f"/audio/{audio_file}",
        "stim_id":   str(row["id"]),
        "S_mv":      int(row["S_mv"]),
        "D_mv":      int(row["D_mv"]),
        "E":         float(row["E"]),
    }

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