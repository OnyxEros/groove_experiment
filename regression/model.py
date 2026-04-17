import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# =========================================================
# OLS MODEL (paper baseline)
# =========================================================

def train_ols(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model


# =========================================================
# RIDGE MODEL (stable scientific model)
# =========================================================

def train_ridge(X, y, alpha=1.0):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha))
    ])

    model.fit(X, y)
    return model


# =========================================================
# FEATURE MATRIX BUILDER
# =========================================================

def build_X(df):
    """
    Select regression features
    """

    features = [
        "S_mv",
        "S_mv_sq",
        "E",
        "D",
        "I",
        "V",
        "bpm"
    ]

    # interaction term (important for ton paper)
    if "S_mv" in df.columns and "E" in df.columns:
        df["S_mv_E"] = df["S_mv"] * df["E"]
        features.append("S_mv_E")

    X = df[features].fillna(0)
    return X, features


def get_target(df):
    return df["groove"].astype(float)