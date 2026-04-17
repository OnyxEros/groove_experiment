import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from config import MP3_DIR, METADATA_PATH
from backend.models import Response
from backend.frontend import HTML_PAGE

from infra.supabase_client import insert_response


# =========================================================
# APP
# =========================================================

app = FastAPI()


# =========================================================
# STARTUP
# =========================================================

@app.on_event("startup")
def startup():
    """
    Load ONLY local stimuli metadata.
    Supabase is NOT loaded here.
    """

    df = pd.read_csv(METADATA_PATH)

    if df.empty:
        raise RuntimeError("metadata.csv is empty")

    if "mp3_path" not in df.columns:
        raise RuntimeError("Missing column: mp3_path")

    # prepare audio filenames for frontend
    df["audio_file"] = df["mp3_path"].apply(lambda p: Path(p).name)

    app.state.df_global = df


# =========================================================
# STATIC AUDIO SERVER
# =========================================================

app.mount(
    "/audio",
    StaticFiles(directory=str(MP3_DIR)),
    name="audio"
)


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
# STIMULI ENDPOINT
# =========================================================

@app.get("/stimuli")
def get_stimuli(n: int = 20):

    df = app.state.df_global
    n = min(n, len(df))

    sample = df.sample(n).copy()

    sample["audio_url"] = sample["audio_file"].apply(
        lambda f: f"/audio/{f}"
    )

    sample = sample.drop(columns=["mp3_path"], errors="ignore")

    return sample.to_dict(orient="records")


# =========================================================
# RESPONSE → SUPABASE (CLEAN RELAY)
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

        "phase": row.get("phase"),
        "bpm": row.get("bpm"),

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
    return HTML_PAGE