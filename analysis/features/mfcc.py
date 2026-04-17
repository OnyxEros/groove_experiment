import librosa
import numpy as np


# =========================================================
# CORE
# =========================================================

def extract_mfcc(path: str, n_mfcc: int = 13) -> np.ndarray:
    try:
        y, sr = librosa.load(path, sr=22050)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)

        feat = np.concatenate([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(delta, axis=1),
            np.mean(delta2, axis=1)
        ])

        return feat.astype(np.float32)

    except Exception as e:
        print(f"⚠️ MFCC failed for {path}: {e}")
        return np.zeros(n_mfcc * 4, dtype=np.float32)


# =========================================================
# DATASET
# =========================================================

def extract_mfcc_dataset(paths: list[str], n_mfcc: int = 13) -> np.ndarray:
    features = []

    for p in paths:
        feat = extract_mfcc(p, n_mfcc=n_mfcc)
        features.append(feat)

    return np.vstack(features)