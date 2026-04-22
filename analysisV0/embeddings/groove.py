import numpy as np

FEATURES = ["D", "V", "S_real"]


def build_groove_embedding(manager, df):
    X = df[FEATURES].values.astype(np.float32)
    return manager.fit("groove", X)


def project_groove(manager, df):
    X = df[FEATURES].values.astype(np.float32)
    return manager.transform("groove", X)