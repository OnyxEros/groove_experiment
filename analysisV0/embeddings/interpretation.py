import numpy as np
import pandas as pd


# =========================================================
# 1. CLUSTER PROFILING
# =========================================================

def profile_clusters(df, labels, feature_cols):
    """
    Build statistical description of each cluster.
    """

    df = df.copy()
    df["cluster"] = labels

    profiles = []

    for c in sorted(df["cluster"].unique()):
        if c == -1:
            continue  # noise

        sub = df[df["cluster"] == c]

        profile = {
            "cluster": c,
            "size": len(sub),
        }

        for f in feature_cols:
            profile[f"{f}_mean"] = sub[f].mean()
            profile[f"{f}_std"] = sub[f].std()

        profiles.append(profile)

    return pd.DataFrame(profiles)