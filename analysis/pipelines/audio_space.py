import numpy as np
from analysis.features.mfcc import extract_mfcc
from analysis.dataset.audio_dataset import load_audio_paths


def build_audio_embeddings(mp3_dir):

    data = []

    for p in load_audio_paths(mp3_dir):
        try:
            feat = extract_mfcc(p)

            if feat is None:
                continue

            stim_id = p.split("/")[-1].replace(".mp3", "")

            data.append((stim_id, p, feat))

        except Exception as e:
            print(f"[WARN] MFCC failed: {p} -> {e}")

    if not data:
        raise ValueError("No audio features extracted.")

    stim_ids, paths, vectors = zip(*data)

    return np.vstack(vectors), list(stim_ids), list(paths)