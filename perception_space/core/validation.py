import pandas as pd

REQUIRED_COLS = {"stimulus_id", "groove", "complexity"}

def validate_perception_df(df: pd.DataFrame):
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df.isna().any().any():
        raise ValueError("NaN values detected in perception data")

    return True
