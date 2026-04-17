from analysis.embeddings.interpretation import profile_clusters
from analysis.embeddings.semantic import semantic_label_groove


def run_interpretation(df, labels):
    feature_cols = ["D", "V", "S_real"]

    profiles = profile_clusters(df, labels, feature_cols)

    profiles["label"] = profiles.apply(semantic_label_groove, axis=1)

    return profiles
