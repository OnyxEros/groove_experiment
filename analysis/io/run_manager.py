from pathlib import Path
import json
import numpy as np
import pandas as pd


class RunManager:

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir

        self.paths = {
            "embeddings": run_dir / "embeddings",
            "clustering": run_dir / "clustering",
            "figures": run_dir / "figures",
        }

        for p in self.paths.values():
            p.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # SAVE GENERIC
    # ----------------------------

    def save_json(self, name, obj):
        path = self.run_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(obj, f, indent=2)

    def save_npy(self, folder, name, array):
        path = self.paths[folder] / f"{name}.npy"
        np.save(path, array)

    def save_pickle(self, folder, name, obj):
        import pickle
        path = self.paths[folder] / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    def save_dataframe(self, df, name):
        path = self.run_dir / f"{name}.parquet"
        df.to_parquet(path)
