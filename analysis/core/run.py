"""
analysis/core/run.py
====================
Point d'entrée du module d'analyse.

Modes disponibles :
    full                — embeddings → projection → viz → export
    full_with_clustering — + clustering → interpretation
    groove              — embeddings → clustering → interpretation → viz → export
    audio               — embeddings → projection → viz → export  (alias de full)

Note : metrics_view retiré du mode "full" — il produisait metrics_matrix
sans qu'aucun step suivant ne le consomme dans ce mode.
"""

from analysis.core.engine import AnalysisEngine
from analysis.core.context import AnalysisContext
from analysis.core.pipeline import build_pipeline
from analysis.core.registry import load_steps
from analysis.dataset.loader import load_dataset
from groove.generator import run_experiment
from config import get_current_run



_PIPELINES = {
    "full": [
        "embeddings",
        "projection",
        "viz",
        "export",
    ],
    "full_with_clustering": [
        "embeddings",
        "projection",
        "clustering",
        "metrics_view",
        "interpretation",
        "viz",
        "export",
    ],
    "groove": [
        "embeddings",
        "clustering",
        "interpretation",
        "viz",
        "export",
    ],
    "audio": [
        "embeddings",
        "projection",
        "viz",
        "export",
    ],
}


def run_analysis(mode: str, steps=None, save=True, seed=42):
    load_steps()
    run_dir = get_current_run()

    # 1. On charge les deux mondes
    df_gen, stim_cache = run_experiment(seed=seed)
    df_real = load_dataset()

    # 2. On alimente le contexte
    context = AnalysisContext(
        run_dir=run_dir,
        dataset=df_gen,  # Le dataset par défaut reste le génératif
        seed=seed,
        config={"seed": seed, "n_clusters": 6},
    )
    context.cache["stim_cache"] = stim_cache
    context.cache["df_real"] = df_real  # <-- Injection du corpus réel dans le cache
    if steps is not None:
        pipeline_steps = steps
    elif mode in _PIPELINES:
        pipeline_steps = _PIPELINES[mode]
    else:
        raise ValueError(
            f"Mode inconnu : '{mode}'. "
            f"Modes disponibles : {list(_PIPELINES.keys())}"
        )

    pipeline = build_pipeline(pipeline_steps)
    engine   = AnalysisEngine(pipeline)

    return engine.run(context)