from pathlib import Path
import numpy as np
import json


def load_analysis_run(run_dir: Path):
    embeddings_dir = run_dir / "embeddings"

    structural = np.load(embeddings_dir / "structural.npy")
    realized = np.load(embeddings_dir / "realized.npy")

    clusters = np.load(run_dir / "clustering" / "labels.npy")

    summary = json.loads((run_dir / "summary.json").read_text())

    # 🔒 sanity checks
    n = realized.shape[0]

    assert clusters.shape[0] == n, "Cluster mismatch with embeddings"

    return {
        "structural": structural,
        "realized": realized,
        "clusters": clusters,
        "summary": summary,
    }
