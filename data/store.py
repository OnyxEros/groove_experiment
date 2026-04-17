import pandas as pd
from config import METADATA_PATH


def load_dataset():
    if not METADATA_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(METADATA_PATH)
