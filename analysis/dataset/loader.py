import pandas as pd
from config import METADATA_PATH


def normalize_columns(df):
    """
    Normalise les noms de colonnes pour la rétro-compatibilité.

    Notation cible :
        Paramètres génératifs : S_mv, D_mv, E_mv, P_mv
        Descripteurs émergents : D, I, V, S, E, P

    Ancienne notation → nouvelle :
        S_real → S
        E_real → E
        P_real → P
        bpm / Bpm → BPM
    """
    rename_map = {
        "bpm":    "BPM",
        "Bpm":    "BPM",
        "S_real": "S",
        "E_real": "E",
        "P_real": "P",
    }
    # Ne renommer que les colonnes présentes et dont la cible n'existe pas encore
    rename_map = {
        k: v for k, v in rename_map.items()
        if k in df.columns and v not in df.columns
    }
    return df.rename(columns=rename_map)


def load_dataset(limit: int | None = None):
    df = pd.read_csv(METADATA_PATH)
    df = normalize_columns(df)

    if limit:
        df = df.head(limit)

    # FIX #2 : required_cols aligné sur schema.py (REQUIRED_COLUMNS).
    # Colonnes manquantes précédemment : P_mv, phase, repeat, BPM.
    required_cols = [
        "id",
        "phase",
        "repeat",
        # Paramètres génératifs
        "S_mv", "D_mv", "E_mv", "P_mv",
        # Descripteurs émergents
        "D", "I", "V", "S", "E", "P",
        # Tempo
        "BPM",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[DATASET] colonnes manquantes : {missing}")

    print(f"[DATASET] {len(df)} stimuli chargés")
    return df