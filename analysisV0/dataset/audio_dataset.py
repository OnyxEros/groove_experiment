import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from data.loader import load_audio_paths
from analysis.features.mfcc import extract_mfcc


def _process(path):
    try:
        feat = extract_mfcc(path)
        if feat is None:
            return None

        return path, feat

    except Exception:
        return None


def load_dataset(n_jobs=None):
    paths = load_audio_paths()

    results = []

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = [ex.submit(_process, p) for p in paths]

        for f in as_completed(futures):
            r = f.result()
            if r:
                results.append(r)

    if not results:
        raise ValueError("No features extracted")

    results.sort(key=lambda x: x[0])

    paths = [r[0] for r in results]
    X = np.vstack([r[1] for r in results])

    df = pd.DataFrame({
        "path": paths,
        "stimulus_id": [os.path.basename(p).split(".")[0] for p in paths]
    })

    return df, X