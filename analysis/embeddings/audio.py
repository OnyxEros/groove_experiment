import numpy as np


def build_audio_embedding(manager, X: np.ndarray):
    return manager.fit("audio", X)


def project_audio(manager, X: np.ndarray):
    return manager.transform("audio", X)