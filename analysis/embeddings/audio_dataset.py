import os
import numpy as np
from .mfcc import extract_mfcc

def build_audio_embeddings(mp3_dir):
    paths = []
    vectors = []

    for root, _, files in os.walk(mp3_dir):
        for f in files:
            if f.endswith(".mp3"):
                p = os.path.join(root, f)
                try:
                    vectors.append(extract_mfcc(p))
                    paths.append(p)
                except Exception:
                    pass

    return np.array(vectors), paths
