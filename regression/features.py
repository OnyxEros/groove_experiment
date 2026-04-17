import numpy as np


FEATURES = ["D", "V", "S_real", "u1", "u2", "u3"]


def build_feature_matrix(df):
    """
    Build regression feature matrix.

    Returns:
        X (np.ndarray)
        y (np.ndarray)
    """

    required = FEATURES + ["response"]

    df = df.dropna(subset=required)

    X = df[FEATURES].values.astype(float)
    y = df["response"].values.astype(float)

    return X, y
