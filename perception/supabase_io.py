"""
perception/supabase_io.py
=========================
Fetch et cache local des réponses perceptives depuis Supabase.

Stratégie :
    - Si data/responses.csv existe et refresh=False → lit le cache (offline, rapide).
    - Sinon → fetch Supabase, valide, sauve dans RESP_FILE.

La table Supabase n'est jamais modifiée ni effacée.
"""

import pandas as pd
from pathlib import Path

from infra.supabase_client import fetch_responses
from config import RESP_FILE


# =========================================================
# FETCH + CACHE
# =========================================================

def fetch_ratings(refresh: bool = False) -> pd.DataFrame:
    """
    Charge les réponses perceptives (cache local ou Supabase).

    Args:
        refresh: forcer un re-fetch depuis Supabase même si le cache existe.

    Returns:
        DataFrame avec colonnes :
            participant_id, stim_id, groove, complexity, rt, created_at
    """
    cache_path = Path(RESP_FILE)

    if cache_path.exists() and not refresh:
        df = pd.read_csv(cache_path)
        _validate(df)
        return df

    # ── Fetch Supabase ────────────────────────────────────
    data = fetch_responses()

    if not data:
        raise ValueError(
            "Aucune réponse trouvée dans Supabase (table 'responses' vide).\n"
            "Lance d'abord la collecte via l'interface web, "
            "ou vérifie tes variables d'environnement SUPABASE_URL / SUPABASE_KEY."
        )

    df = pd.DataFrame(data)
    _validate(df)

    # ── Sauvegarde cache ──────────────────────────────────
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"[supabase_io] {len(df)} réponses sauvées → {cache_path}")

    return df


# =========================================================
# VALIDATION
# =========================================================

def _validate(df):
    """
    Vérifie les colonnes minimales, nettoie les types, filtre les RT aberrants.
    Modifie le DataFrame en place.
    """
    required = {"stim_id", "groove"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans les réponses : {missing}\n"
            f"Colonnes présentes : {list(df.columns)}"
        )
 
    # Nettoyage types
    df.dropna(subset=["stim_id", "groove"], inplace=True)
    df["stim_id"] = df["stim_id"].astype(str)
    df["groove"]  = pd.to_numeric(df["groove"],  errors="coerce")
 
    if "complexity" in df.columns:
        df["complexity"] = pd.to_numeric(df["complexity"], errors="coerce")
 
    if "rt" in df.columns:
        df["rt"] = pd.to_numeric(df["rt"], errors="coerce")
 
    # Drop les lignes où groove est NaN après coercion
    df.dropna(subset=["groove"], inplace=True)
 
    # ── Filtre RT ──────────────────────────────────────────
    # Importé ici pour éviter la dépendance circulaire
    # (supabase_io ← regression n'est pas un import normal)
    RT_MIN_S = 4.0
    RT_MAX_S = 600.0
 
    if "rt" in df.columns:
        before = len(df)
        df.drop(
            index=df[
                df["rt"].notna() &
                ~df["rt"].between(RT_MIN_S, RT_MAX_S, inclusive="both")
            ].index,
            inplace=True,
        )
        n_dropped = before - len(df)
        if n_dropped > 0:
            print(f"[supabase_io] {n_dropped} réponses filtrées (RT hors [{RT_MIN_S}s–{RT_MAX_S}s])")