import uuid
import random
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config import MP3_DIR, METADATA_PATH, INDEX_PATH
from backend.models import Response
from backend.startup import check_environment
from infra.supabase_client import insert_response

# ── Design system (optionnel) ─────────────────────────────
try:
    from backend.design.registry import StimulusRegistry
    DESIGN_MODE = True
except Exception:
    DESIGN_MODE = False


# =========================================================
# RATE LIMITER (in-memory, simple)
# =========================================================

_rate_store: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_REQUESTS = 60   # requêtes max
RATE_LIMIT_WINDOW   = 60   # par fenêtre de N secondes


def _check_rate_limit(client_ip: str) -> None:
    now = time.monotonic()
    window_start = now - RATE_LIMIT_WINDOW
    hits = _rate_store[client_ip]

    # Purge les entrées hors-fenêtre
    _rate_store[client_ip] = [t for t in hits if t > window_start]

    if len(_rate_store[client_ip]) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="Trop de requêtes — réessaie dans un instant.",
        )

    _rate_store[client_ip].append(now)


# =========================================================
# LIFESPAN
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Remplace @app.on_event('startup') / 'shutdown' (déprécié)."""

    # ── Startup ───────────────────────────────────────────
    check_environment()

    df = pd.read_csv(METADATA_PATH)
    df["audio_file"] = df["mp3_path"].apply(lambda p: Path(p).name)
    app.state.df_global = df

    if DESIGN_MODE:
        try:
            registry = StimulusRegistry()
            stimuli = registry.build_stimuli(n_variants=3, seed=42)
            app.state.stimuli = stimuli if stimuli else None
            if stimuli:
                print(f"🎧 Design system → {len(stimuli)} stimuli")
            else:
                print("⚠️  Design system vide → fallback dataframe")
        except Exception as e:
            print(f"⚠️  Design system error : {e} → fallback dataframe")
            app.state.stimuli = None
    else:
        app.state.stimuli = None

    yield

    # ── Shutdown ──────────────────────────────────────────
    _rate_store.clear()
    print("👋 Shutdown propre")


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="Groove Study API",
    version="2.0.0",
    lifespan=lifespan,
)


# =========================================================
# MIDDLEWARE — client IP helper
# =========================================================

def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# =========================================================
# STATIC FILES
# =========================================================

app.mount("/audio",  StaticFiles(directory=str(MP3_DIR)),        name="audio")
app.mount("/static", StaticFiles(directory="backend/static"),     name="static")


# =========================================================
# CACHE — stimuli globaux (recalculé si le df change)
# =========================================================

@lru_cache(maxsize=4)
def _cached_stimuli_from_df(df_hash: int, n: int) -> list[dict]:
    """
    Cache LRU sur le résultat du sampling dataframe.
    df_hash est un entier dérivé du df pour invalider le cache si les données changent.
    """
    df = app.state.df_global
    sample = df.sample(min(n, len(df))).copy()

    if "stim_id" not in sample.columns and "id" in sample.columns:
        sample["stim_id"] = sample["id"].apply(lambda i: f"stim_{int(i):04d}")

    sample["audio_url"] = sample["audio_file"].apply(lambda f: f"/audio/{f}")
    sample = sample.drop(columns=["mp3_path"], errors="ignore")

    return sample.to_dict(orient="records")


def _df_hash() -> int:
    """Hash léger basé sur shape + colonnes pour invalider le cache LRU."""
    df = app.state.df_global
    return hash((df.shape, tuple(df.columns)))


# =========================================================
# ENDPOINTS
# =========================================================

# ── Healthcheck ───────────────────────────────────────────

@app.get("/health", tags=["system"])
def health(request: Request):
    _check_rate_limit(_client_ip(request))
    df = app.state.df_global
    return {
        "status": "ok",
        "stimuli_count": len(df),
        "design_mode": DESIGN_MODE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Participant ───────────────────────────────────────────

@app.get("/new_participant", tags=["session"])
def new_participant(request: Request):
    _check_rate_limit(_client_ip(request))
    return {
        "participant_id": uuid.uuid4().hex[:8],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Stimuli ───────────────────────────────────────────────

@app.get("/stimuli", tags=["experiment"])
def get_stimuli(
    request: Request,
    n: int = Query(default=24, ge=1, le=200),
):
    _check_rate_limit(_client_ip(request))

    # MODE 1 : design system
    if getattr(app.state, "stimuli", None) is not None:
        stimuli: list[dict] = app.state.stimuli
        sample = list(stimuli[: min(n, len(stimuli))])
        random.shuffle(sample)

        for s in sample:
            if "mp3_path" in s:
                s["audio_url"] = f"/audio/{Path(s['mp3_path']).name}"

        return sample

    # MODE 2 : fallback dataframe avec cache
    return _cached_stimuli_from_df(_df_hash(), n)


# ── Example ───────────────────────────────────────────────

@app.get("/example", tags=["experiment"])
def get_example(request: Request):
    _check_rate_limit(_client_ip(request))

    df: pd.DataFrame = app.state.df_global

    required = {"S_mv", "D_mv", "E", "S_real"}
    if not required.issubset(df.columns):
        # Colonnes absentes → retourne le premier stimulus disponible
        row = df.iloc[0]
    else:
        mask = (df["S_mv"] == 2) & (df["D_mv"] == 2) & (df["E"] == 1.0)
        candidates = df[mask]
        row = (
            candidates.loc[candidates["S_real"].idxmax()]
            if not candidates.empty
            else df.loc[df["S_real"].idxmax()]
        )

    audio_file = (
        row["audio_file"]
        if "audio_file" in row
        else f"stim_{int(row['id']):04d}.mp3"
    )

    return {
        "audio_url": f"/audio/{audio_file}",
        "stim_id":   str(row.get("id", "unknown")),
        "S_mv":      int(row["S_mv"])   if "S_mv"   in row else None,
        "D_mv":      int(row["D_mv"])   if "D_mv"   in row else None,
        "E":         float(row["E"])    if "E"      in row else None,
    }


# ── Response → Supabase ───────────────────────────────────

@app.post("/response", tags=["experiment"])
async def save_response(resp: Response, request: Request):
    _check_rate_limit(_client_ip(request))

    row = resp.model_dump()

    clean_row: dict[str, Any] = {
        "participant_id": row["participant_id"],
        "stim_id":        row["stim_id"],
        "groove":         row["groove"],
        "complexity":     row["complexity"],
        "rt":             row["rt"],
        "rt_type":        row.get("rt_type"),
        "trial_index":    row.get("trial_index"),
        "session_id":     row.get("session_id"),
        "condition":      row.get("condition"),
        "created_at":     datetime.now(timezone.utc).isoformat(),
    }

    try:
        insert_response(clean_row)
    except Exception as e:
        # Log serveur mais ne révèle pas les détails à l'appelant
        print(f"⚠️  Supabase error [{resp.participant_id}]: {e}")
        raise HTTPException(
            status_code=503,
            detail="Erreur d'enregistrement — réessaie dans un instant.",
        )

    return {"status": "ok"}


# ── Frontend ──────────────────────────────────────────────

from fastapi.responses import FileResponse

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(INDEX_PATH)


# ── Global error handler ──────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail},
        )
    print(f"❌ Unhandled error on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Erreur interne — contacte l'administrateur."},
    )