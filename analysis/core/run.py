"""
analysis/core/run.py — VERSION CORRIGÉE POLARITÉ PUSH/PULL
==========================================================
Point d'entrée du module d'analyse globale du corpus.

Modifications de polarité :
    Rétablit le sens physique standard pour le désalignement (P / P_mv) :
    P > 0  →  Rushing (Avance temporelle du hi-hat)
    P < 0  →  Laid-back (Retard temporel du hi-hat)
    
    Le correctif applique l'inversion sur les colonnes du DataFrame génératif
    `df_gen` immédiatement après l'exécution de l'expérience afin de diffuser
    la bonne polarité dans les étapes suivantes (projection, exports, matrices).
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
    "groove": [
        "embeddings",
        "projection",     # Ajouté à la place du clustering pour projeter tes stimuli
        "metrics_view",   # Optionnel : ajoute-le si tu veux calculer tes matrices de métriques
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

    # =========================================================================
    # CORRECTIF DE POLARITÉ (PIPELINE D'ANALYSE CORE)
    # =========================================================================
    from config import apply_polarity_fix_df
    if df_gen is not None:
        df_gen = apply_polarity_fix_df(df_gen)
    # =========================================================================

    # 2. On alimente le contexte
    context = AnalysisContext(
        run_dir=run_dir,
        dataset=df_gen,  # Le dataset par défaut contient maintenant les signes redressés
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