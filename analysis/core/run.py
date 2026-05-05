"""
analysis/core/run.py
====================
Point d'entrée du module d'analyse du dataset.
"""

from analysis.core.engine import AnalysisEngine
from analysis.core.context import AnalysisContext
from analysis.core.pipeline import build_pipeline
from analysis.core.registry import load_steps
from groove.generator import run_experiment
from config import get_current_run


def run_analysis(mode: str, steps=None, save=True, seed=42):

    # =====================================================
    # registry init
    # =====================================================
    load_steps()

    run_dir = get_current_run()   # lit .current_run — erreur claire si absent

    # =====================================================
    # data source unique
    # =====================================================
    df, stim_cache = run_experiment(seed=seed)

    context = AnalysisContext(
        run_dir=run_dir,
        dataset=df,
        seed=seed,
        config={
            "seed":       seed,
            "n_clusters": 6,
        }
    )

    context.cache["stim_cache"] = stim_cache

    # =====================================================
    # pipeline
    # =====================================================
    if steps is not None:
        pipeline_steps = steps

    else:
        if mode == "full":
            pipeline_steps = [
                "embeddings",
                "projection",
                "clustering",
                "metrics_view",
                "interpretation",
                "viz",
                "export",
            ]
        elif mode == "audio":
            pipeline_steps = [
                "embeddings",
                "projection",
                "clustering",
                "viz",
                "export",
            ]
        elif mode == "groove":
            pipeline_steps = [
                "embeddings",
                "clustering",
                "interpretation",
                "viz",
                "export",
            ]
        else:
            raise ValueError(f"Unknown mode: {mode}")

    pipeline = build_pipeline(pipeline_steps)
    engine   = AnalysisEngine(pipeline)

    return engine.run(context)