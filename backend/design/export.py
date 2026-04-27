import pandas as pd
from pathlib import Path

from backend.design.registry import StimulusRegistry


def export_design_csv(path="data/cache/design.csv", seed=42):

    registry = StimulusRegistry()
    stimuli = registry.build_stimuli(n_variants=3, seed=seed)

    df = pd.DataFrame(stimuli)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    print(f"📦 design exported → {path}")
