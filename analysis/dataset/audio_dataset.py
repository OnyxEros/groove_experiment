import os
import numpy as np
from analysis.features.mfcc import extract_mfcc


def load_audio_paths(mp3_dir):
    paths = [
        os.path.join(root, f)
        for root, _, files in os.walk(mp3_dir)
        for f in files
        if f.endswith(".mp3")
    ]
    return sorted(paths)


def build_audio_embeddings(mp3_dir):

    data = []

    for p in load_audio_paths(mp3_dir):
        try:
            feat = extract_mfcc(p)

            if feat is None:
                continue

            data.append((p, feat))

        except Exception as e:
            print(f"[WARN] failed MFCC for {p}: {e}")

    if not data:
        raise ValueError("No audio features extracted.")

    paths, vectors = zip(*data)

    return np.vstack(vectors), list(paths)