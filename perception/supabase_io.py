import pandas as pd
from pathlib import Path

from infra.supabase_client import fetch_responses
from config import RESP_FILE


def fetch_ratings(refresh: bool = False) -> pd.DataFrame:
    """
    Charge les réponses perceptives.

    Stratégie :
      - Si RESP_FILE existe et refresh=False → lit le cache local (rapide, offline).
      - Sinon → fetch Supabase et sauve dans RESP_FILE.

    La table Supabase n'est jamais modifiée ni effacée.

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

    # --- fetch Supabase ---
    data = fetch_responses()
    if not data:
        raise ValueError(
            "Aucune réponse trouvée dans Supabase (table 'responses' vide). "
            "Lance d'abord la collecte de données via l'interface."
        )

    df = pd.DataFrame(data)
    _validate(df)

    # cache local
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)

    return df


def _validate(df: pd.DataFrame) -> None:
    """Vérifie les colonnes minimales et nettoie les types."""
    required = {"stim_id", "groove"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans les réponses : {missing}")

    df.dropna(subset=["stim_id", "groove"], inplace=True)
    df["groove"] = pd.to_numeric(df["groove"], errors="coerce")

    if "complexity" in df.columns:
        df["complexity"] = pd.to_numeric(df["complexity"], errors="coerce")
    if "rt" in df.columns:
        df["rt"] = pd.to_numeric(df["rt"], errors="coerce")