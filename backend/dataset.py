import pandas as pd
from config import METADATA_PATH


def load_dataset():

    if not METADATA_PATH.exists():
        raise RuntimeError(f"metadata.csv introuvable: {METADATA_PATH}")

    df = pd.read_csv(METADATA_PATH)

    if "mp3_path" not in df.columns:
        raise RuntimeError("mp3_path missing in dataset")

    return df