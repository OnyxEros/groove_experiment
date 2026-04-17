import pandas as pd

from regression.supabase_loader import load_responses
from regression.stimulus_loader import load_stimuli
from regression.features import add_features


def build_regression_dataset():
    """
    Merge:
    - human responses (Supabase)
    - stimulus metadata (local)
    """

    responses = load_responses()
    stimuli = load_stimuli()

    if responses.empty:
        raise ValueError("No Supabase data")

    # merge on stim_id
    df = responses.merge(
        stimuli,
        on="stim_id",
        how="left"
    )

    return df


def load_dataset():
    responses = load_responses()
    stimuli = load_stimuli()

    df = responses.merge(stimuli, on="stim_id", how="left")

    df = add_features(df)

    # target
    df["y"] = df["groove"]

    return df
