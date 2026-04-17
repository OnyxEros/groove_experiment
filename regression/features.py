FEATURES = ["S_mv", "E", "D", "V", "I", "BPM"]
TARGET = "groove_rating"

def prepare_features(df):
    df = df.copy()
    df["S_mv2"] = df["S_mv"] ** 2
    return df