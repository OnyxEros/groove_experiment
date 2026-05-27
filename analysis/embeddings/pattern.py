import numpy as np
from sklearn.preprocessing import RobustScaler
from analysis.embeddings.base import BaseEmbedding


class PatternEmbedding(BaseEmbedding):
    """
    ESPACE MUSICAL STRUCTUREL (FORME)
    RobustScaler utilisé pour éviter le collapse sur les outliers.
    """

    name = "pattern"
    VOICES = ["kick", "bass", "snare", "hihat"]

    def compute(self, df, cache=None):
        stim_cache = cache.get("stim_cache") if cache else None
        if stim_cache is None:
            raise ValueError("PatternEmbedding requires stim_cache in cache")

        X = []
        for _, row in df.iterrows():
            stim = stim_cache.get(row.get("id", None), None)
            if stim is None:
                X.append(np.zeros(len(self.VOICES) * 6))
                continue

            feats = []
            for voice in self.VOICES:
                seq = stim.get(voice, None)
                if seq is None or len(seq) == 0:
                    feats.extend([0.0] * 6)
                    continue
                feats.extend(self._features(np.asarray(seq, dtype=np.float32)))

            X.append(feats)

        X = np.asarray(X, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        scaler = RobustScaler()
        return scaler.fit_transform(X)

    def _features(self, seq):
        n = len(seq)
        if n == 0:
            return np.zeros(6, dtype=np.float32)

        onsets = np.where(seq > 0)[0]
        if len(onsets) == 0:
            return np.zeros(6, dtype=np.float32)

        density   = len(onsets) / max(n, 1)
        centroid  = np.mean(onsets) / max(n, 1)

        if len(onsets) > 1:
            ioi        = np.diff(onsets)
            regularity = 1.0 / (1.0 + np.std(ioi))
            dispersion = np.std(ioi)
        else:
            # FIX #5 : un seul onset = pattern parfaitement régulier par définition.
            # regularity=1.0 (maximum), dispersion=0.0 (aucun écart inter-onset possible).
            # L'ancienne valeur regularity=0.0 était sémantiquement incorrecte.
            regularity = 1.0
            dispersion = 0.0

        syncopation = np.mean(onsets % 4 != 0)

        hist, _ = np.histogram(onsets, bins=8, range=(0, max(n, 1)))
        p       = hist / (np.sum(hist) + 1e-9)
        entropy = -np.sum(p * np.log(p + 1e-9))

        return np.array(
            [density, centroid, regularity, syncopation, dispersion, entropy],
            dtype=np.float32,
        )