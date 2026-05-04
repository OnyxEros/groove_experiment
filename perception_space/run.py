from pathlib import Path
import pandas as pd

from perception_space.core.loader import load_analysis_run
from perception_space.core.align import align_embeddings_with_perception
from perception_space.core.manifold import compute_local_geometry

from perception_space.core.normalize import normalize
from perception_space.core.validation import validate_perception_df

def run_perception_space(run_dir: str, perception_data):

    run_dir = Path(run_dir)

    analysis = load_analysis_run(run_dir)

    X = analysis["realized"]

    X, y_groove, y_complexity = align_embeddings_with_perception(
        X,
        perception_data
    )

    validate_perception_df(perception_data)

    # central normalization
    X = normalize(X)

    groove_geometry = compute_local_geometry(X, y_groove)
    complexity_geometry = compute_local_geometry(X, y_complexity)

    return {
        "groove": groove_geometry,
        "complexity": complexity_geometry,
        "clusters": analysis["clusters"],
        "summary": analysis["summary"]
    }