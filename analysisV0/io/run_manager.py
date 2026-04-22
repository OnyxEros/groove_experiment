from pathlib import Path
import json
import numpy as np
import pandas as pd


class RunManager:

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # CONFIG
    # =====================================================
    def save_config(self, config: dict):
        with open(self.run_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    # =====================================================
    # EMBEDDINGS
    # =====================================================
    def save_embedding(self, name: str, Z: np.ndarray):
        np.save(self.run_dir / f"{name}_embedding.npy", Z)

    def load_embedding(self, name: str):
        return np.load(self.run_dir / f"{name}_embedding.npy")

    # =====================================================
    # CLUSTERS
    # =====================================================
    def save_clusters(self, name: str, labels):
        df = pd.DataFrame({"cluster": labels})
        df.to_csv(self.run_dir / f"{name}_clusters.csv", index=False)

    def load_clusters(self, name: str):
        return pd.read_csv(self.run_dir / f"{name}_clusters.csv")

    # =====================================================
    # GENERIC
    # =====================================================
    def path(self, filename: str):
        return self.run_dir / filename
