import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from backend.dataset import load_dataset
from backend.models import Response
from backend.startup import check_environment
from backend.frontend import HTML_PAGE
from backend.db import supabase

from config import MP3_DIR


# =========================================================
# APP
# =========================================================

app = FastAPI()


# =========================================================
# STARTUP
# =========================================================

@app.on_event("startup")
def startup():
    check_environment()

    df = load_dataset()

    if df.empty:
        raise RuntimeError("Dataset is empty")

    if "mp3_path" not in df.columns:
        raise RuntimeError("Missing column: mp3_path")

    df["audio_file"] = df["mp3_path"].apply(lambda p: Path(p).name)

    app.state.df_global = df


# =========================================================
# STATIC AUDIO
# =========================================================

app.mount(
    "/audio",
    StaticFiles(directory=str(MP3_DIR)),
    name="audio"
)


# =========================================================
# ENDPOINTS
# =========================================================

@app.get("/new_participant")
def new_participant():
    return {
        "participant_id": uuid.uuid4().hex[:8],
        "timestamp": datetime.utcnow().isoformat()
    }


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
# SUPABASE
# =========================================================

@app.post("/response")
def save_response(resp: Response):

    row = resp.model_dump()
    row["timestamp"] = datetime.utcnow().isoformat()

    try:
        supabase.table("responses").insert(row).execute()
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