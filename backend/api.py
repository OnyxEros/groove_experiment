import uuid
import random
import time
import os
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
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
# RATE LIMITER
# =========================================================
# Stratégie à deux niveaux :
#   1. Redis (si REDIS_URL présent) → fonctionne multi-worker
#   2. In-memory fallback           → single-worker, suffit pour Render Free
# =========================================================

RATE_LIMIT_REQUESTS = 60
RATE_LIMIT_WINDOW   = 60   # secondes

# ── Tentative Redis ───────────────────────────────────────
_redis = None
_REDIS_URL = os.getenv("REDIS_URL")

if _REDIS_URL:
    try:
        import redis.asyncio as aioredis
        _redis = aioredis.from_url(_REDIS_URL, decode_responses=True)
        print(f"✅ Redis rate limiter → {_REDIS_URL[:30]}…")
    except ImportError:
        print("⚠️  redis-py absent — fallback in-memory (pip install redis)")
    except Exception as e:
        print(f"⚠️  Redis indisponible ({e}) — fallback in-memory")

# ── In-memory fallback ────────────────────────────────────
_rate_store: dict[str, list[float]] = defaultdict(list)


async def _check_rate_limit(client_ip: str) -> None:
    """
    Rate limiter asynchrone.
    Redis (sliding window via ZADD/ZREMRANGEBYSCORE) si disponible,
    sinon liste in-memory (suffisant pour un seul worker Render).
    """
    if _redis:
        # Sliding window avec Redis sorted set
        now      = time.time()
        key      = f"rl:{client_ip}"
        pipe     = _redis.pipeline()
        pipe.zremrangebyscore(key, 0, now - RATE_LIMIT_WINDOW)
        pipe.zadd(key, {str(now): now})
        pipe.zcard(key)
        pipe.expire(key, RATE_LIMIT_WINDOW * 2)
        results  = await pipe.execute()
        count    = results[2]
    else:
        # In-memory (single-worker)
        now          = time.monotonic()
        window_start = now - RATE_LIMIT_WINDOW
        hits         = _rate_store[client_ip]
        _rate_store[client_ip] = [t for t in hits if t > window_start]
        _rate_store[client_ip].append(now)
        count = len(_rate_store[client_ip])

    if count > RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="Trop de requêtes — réessaie dans un instant.",
        )


# =========================================================
# LIFESPAN
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────
    check_environment()

    df = pd.read_csv(METADATA_PATH)
    df["audio_file"] = df["mp3_path"].apply(lambda p: Path(p).name)
    app.state.df_global = df

    # Index des stim_id valides pour validation côté serveur
    if "stim_id" in df.columns:
        app.state.valid_stim_ids = set(df["stim_id"].astype(str))
    elif "id" in df.columns:
        app.state.valid_stim_ids = {f"stim_{int(i):04d}" for i in df["id"]}
    else:
        app.state.valid_stim_ids = set()

    print(f"✅ {len(app.state.valid_stim_ids)} stim_id valides chargés")

    if DESIGN_MODE:
        try:
            registry = StimulusRegistry()
            stimuli  = registry.build_stimuli(n_variants=3, seed=42)
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
    if _redis:
        await _redis.aclose()
    print("👋 Shutdown propre")


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="Groove Study API",
    version="2.1.0",
    lifespan=lifespan,
)


# =========================================================
# HELPERS
# =========================================================

def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# =========================================================
# STATIC FILES
# =========================================================

app.mount("/audio",  StaticFiles(directory=str(MP3_DIR)),    name="audio")
app.mount("/static", StaticFiles(directory="backend/static"), name="static")


# =========================================================
# CACHE STIMULI
# =========================================================

@lru_cache(maxsize=4)
def _cached_stimuli_from_df(df_hash: int, n: int) -> list[dict]:
    df     = app.state.df_global
    sample = df.sample(min(n, len(df))).copy()

    if "stim_id" not in sample.columns and "id" in sample.columns:
        sample["stim_id"] = sample["id"].apply(lambda i: f"stim_{int(i):04d}")

    sample["audio_url"] = sample["audio_file"].apply(lambda f: f"/audio/{f}")
    sample = sample.drop(columns=["mp3_path"], errors="ignore")
    return sample.to_dict(orient="records")


def _df_hash() -> int:
    df = app.state.df_global
    return hash((df.shape, tuple(df.columns)))


# =========================================================
# ENDPOINTS
# =========================================================

# ── Healthcheck ───────────────────────────────────────────

@app.get("/health", tags=["system"])
async def health(request: Request):
    await _check_rate_limit(_client_ip(request))
    df = app.state.df_global
    return {
        "status":        "ok",
        "stimuli_count": len(df),
        "design_mode":   DESIGN_MODE,
        "rate_limiter":  "redis" if _redis else "in-memory",
        "timestamp":     datetime.now(timezone.utc).isoformat(),
    }


# ── Participant ───────────────────────────────────────────

@app.get("/new_participant", tags=["session"])
async def new_participant(request: Request):
    await _check_rate_limit(_client_ip(request))
    return {
        "participant_id": uuid.uuid4().hex[:8],
        "timestamp":      datetime.now(timezone.utc).isoformat(),
    }


# ── Stimuli ───────────────────────────────────────────────

@app.get("/stimuli", tags=["experiment"])
async def get_stimuli(
    request: Request,
    n: int = Query(default=24, ge=1, le=200),
):
    await _check_rate_limit(_client_ip(request))

    if getattr(app.state, "stimuli", None) is not None:
        stimuli: list[dict] = app.state.stimuli
        sample = list(stimuli[: min(n, len(stimuli))])
        random.shuffle(sample)
        for s in sample:
            if "mp3_path" in s:
                s["audio_url"] = f"/audio/{Path(s['mp3_path']).name}"
        return sample

    return _cached_stimuli_from_df(_df_hash(), n)


# ── Example ───────────────────────────────────────────────

@app.get("/example", tags=["experiment"])
async def get_example(request: Request):
    await _check_rate_limit(_client_ip(request))

    df: pd.DataFrame = app.state.df_global

    required = {"S_mv", "D_mv", "E", "S_real"}
    if not required.issubset(df.columns):
        row = df.iloc[0]
    else:
        mask       = (df["S_mv"] == 2) & (df["D_mv"] == 2) & (df["E"] == 1.0)
        candidates = df[mask]
        row        = (
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
        "S_mv":      int(row["S_mv"])   if "S_mv" in row else None,
        "D_mv":      int(row["D_mv"])   if "D_mv" in row else None,
        "E":         float(row["E"])    if "E"    in row else None,
    }


# ── Response → Supabase ───────────────────────────────────

@app.post("/response", tags=["experiment"])
async def save_response(resp: Response, request: Request):
    await _check_rate_limit(_client_ip(request))

    # ── Validation stim_id côté serveur ──────────────────
    valid_ids = getattr(app.state, "valid_stim_ids", set())
    if valid_ids and resp.stim_id not in valid_ids:
        raise HTTPException(
            status_code=422,
            detail=f"stim_id inconnu : {resp.stim_id}",
        )

    row = resp.model_dump()

    clean_row: dict[str, Any] = {
        "participant_id":  row["participant_id"],
        "stim_id":         row["stim_id"],
        "groove":          row["groove"],
        "complexity":      row["complexity"],
        "rt":              row["rt"],
        "rt_type":         row.get("rt_type"),
        "trial_index":     row.get("trial_index"),
        "session_id":      row.get("session_id"),
        "condition":       row.get("condition"),
        # ── Fix : listen_duration maintenant sauvegardé ──
        "listen_duration": row.get("listen_duration"),
        "created_at":      datetime.now(timezone.utc).isoformat(),
    }

    try:
        insert_response(clean_row)
    except Exception as e:
        print(f"⚠️  Supabase error [{resp.participant_id}]: {e}")
        raise HTTPException(
            status_code=503,
            detail="Erreur d'enregistrement — réessaie dans un instant.",
        )

    return {"status": "ok"}


# ── Frontend ──────────────────────────────────────────────

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