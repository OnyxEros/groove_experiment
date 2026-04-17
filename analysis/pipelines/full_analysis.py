import pandas as pd
from config import METADATA_PATH

from analysis.pipelines.audio_space import build_audio_embeddings
from analysis.pipelines.groove_space import run_groove_pipeline


def run_full_analysis(mp3_dir, save=True):

    # 1. AUDIO
    emb, stim_ids, paths = build_audio_embeddings(mp3_dir)

    audio_df = pd.DataFrame({
        "stim_id": stim_ids,
        "mp3_path": paths
    })

    # 2. METADATA
    meta = pd.read_csv(METADATA_PATH)

    # 3. JOIN (CRITICAL STEP)
    df = meta.merge(audio_df, on="stim_id", how="inner")

    # 4. GROOVE EMBEDDING
    df, groove_reducer = run_groove_pipeline(df)

    return {
        "df": df,
        "audio_embedding": emb,
        "groove_reducer": groove_reducer,
    }