import pandas as pd
from config import METADATA_PATH


def load_stimuli():
    df = pd.read_csv(METADATA_PATH)

    if "mp3_path" in df.columns:
        df["stim_id"] = df["mp3_path"].apply(lambda p: 
p.split("/")[-1].replace(".mp3", ""))

    return df
