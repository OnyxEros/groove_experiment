import pandas as pd

REQUIRED_COLS = {"stimulus_id", "groove", "complexity"}

def validate_perception_df(df: pd.DataFrame):
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Seul groove est strictement requis sans NaN
    if df["groove"].isna().any():
        raise ValueError("NaN values detected in 'groove' column")
    
    if "stimulus_id" in df.columns and df["stimulus_id"].isna().any():
        raise ValueError("NaN values detected in 'stimulus_id' column")

    return True