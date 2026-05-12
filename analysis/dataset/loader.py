import pandas as pd
from config import METADATA_PATH


def normalize_columns(df):
    """
    Normalise les noms de colonnes pour la rétro-compatibilité.

    Notation cible :
        Paramètres génératifs : S_mv, D_mv, E_mv, P_mv
        Descripteurs émergents : D, I, V, S, E, P

    Ancienne notation → nouvelle :
        S_real → S   (descripteur syncopation émergent)
        E_real → E   (descripteur micro-timing émergent)
        P_real → P   (descripteur push/pull émergent)  ← était manquant
        bpm/Bpm → BPM
    """
    rename_map = {
        "bpm":    "BPM",
        "Bpm":    "BPM",
        "S_real": "S",
        "E_real": "E",
        "P_real": "P",   # ← ajout : P_real n'était pas couvert
    }
    # On ne renomme que les colonnes présentes pour éviter les conflits
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns and v not in df.columns}
    return df.rename(columns=rename_map)


def load_dataset(limit: int | None = None):

    df = pd.read_csv(METADATA_PATH)
    df = normalize_columns(df)

    if limit:
        df = df.head(limit)

    required_cols = [
        "id", "S_mv", "D_mv", "E_mv",
        "D", "I", "V", "S", "E",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[DATASET] missing columns: {missing}")

    print(f"[DATASET] loaded {len(df)} samples")

    return df