import pandas as pd
from config import METADATA_PATH


def normalize_columns(df):
    return df.rename(columns={
        "bpm": "BPM",
        "Bpm": "BPM"
    })


def load_dataset(limit: int | None = None):

    df = pd.read_csv(METADATA_PATH)
    df = normalize_columns(df)

    if limit:
        df = df.head(limit)

    # ✅ VALIDATION MINIMALE
    required_cols = [
        "id", "S_mv", "D_mv", "E",
        "D", "I", "V", "S_real", "E_real"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[DATASET] missing columns: {missing}")

    print(f"[DATASET] loaded {len(df)} samples")

    return df