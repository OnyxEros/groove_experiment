import pandas as pd
from config import METADATA_PATH


def load_stimuli():
    df = pd.read_csv(METADATA_PATH)

    # Construit stim_id depuis mp3_path seulement si stim_id est absent
    if "stim_id" not in df.columns:
        if "mp3_path" in df.columns:
            df["stim_id"] = df["mp3_path"].apply(
                lambda p: str(p).split("/")[-1].replace(".mp3", "")
            )
        elif "id" in df.columns:
            df["stim_id"] = df["id"].apply(lambda i: f"stim_{int(i):04d}")
        else:
            raise ValueError(
                "metadata.csv ne contient ni 'stim_id', ni 'mp3_path', ni 'id'."
            )

    df["stim_id"] = df["stim_id"].astype(str)
    return df