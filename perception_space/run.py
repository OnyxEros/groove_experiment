"""
perception_space/run.py
=======================
Analyse géométrique et statistique du groove dans l'espace latent.

Corrections v2 :
    #2 — permutation_test (Mantel) branché dans le pipeline + figure générée
    #4 — compute_local_geometry appelé sur y_groove_agg (agrégé par stimulus)
         au lieu des ratings bruts (instables, multi-lignes par stimulus)
    #7 — _safe_fig accepte un flag debug (via GROOVE_DEBUG_FIGS=1) qui
         re-raise les exceptions au lieu de les avaler silencieusement
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

# Core
from perception_space.core.loader    import load_analysis_run
from perception_space.core.align     import align_embeddings_with_perception
from perception_space.core.manifold  import compute_local_geometry
from perception_space.core.normalize import normalize
from perception_space.core.validation import validate_perception_df
from perception_space.core.icc       import (
    compute_icc, ratings_to_wide, compute_per_stimulus_variance,
)
from perception_space.core.stats     import (
    kruskal_by_condition, post_hoc_bonferroni,
    permutation_test, compute_condition_stats, coverage_report,
)

# Viz
from perception_space.viz.umap_groove    import plot_umap_groove
from perception_space.viz.cluster_groove import plot_cluster_groove
from perception_space.viz.geometry_plots import (
    plot_local_geometry, plot_permutation_test,
)
from perception_space.viz.icc_plot import (
    plot_icc_summary, plot_per_stimulus_variance,
)
from perception_space.viz.plot_condition_effects import plot_condition_effects

from config import get_current_run

# ── Seuils ───────────────────────────────────────────────
N_MIN_PARTICIPANT_RESPONSES = 10
N_MIN_STIMULUS_RESPONSES    = 2

# Fix #7 — mode debug via variable d'environnement
# Activer avec : GROOVE_DEBUG_FIGS=1 python cli.py --perception
DEBUG_FIGS = os.getenv("GROOVE_DEBUG_FIGS", "0") == "1"


def run_perception_space(perception_data: pd.DataFrame) -> dict:
    run_dir = get_current_run()
    out_dir = run_dir / "perception_space"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[perception_space] run_dir → {run_dir}")
    if DEBUG_FIGS:
        print("[perception_space] ⚠️  DEBUG_FIGS=1 — les erreurs de figures seront levées")

    pdata = perception_data.copy()
    pdata["stimulus_id"] = pdata["stimulus_id"].astype(str)

    # 1. Nettoyage
    if "participant_id" in pdata.columns:
        pdata = pdata.sort_values("stimulus_id")
        pdata = pdata.drop_duplicates(subset=["participant_id", "stimulus_id"], keep="first")

        resp_per_p = pdata.groupby("participant_id").size()
        ghosts = resp_per_p[resp_per_p < N_MIN_PARTICIPANT_RESPONSES].index.tolist()
        pdata = pdata[~pdata["participant_id"].isin(ghosts)]

    # 1.5 Figures exploratoires (Conditions) — avant alignement
    for col in ["musical_background", "experience", "style"]:
        if col in pdata.columns:
            _safe_fig(f"effect_{col}.png", plot_condition_effects, fig_dir,
                      df=pdata, condition=col, target="groove")

    # 2. Embeddings
    analysis = load_analysis_run(run_dir)
    X_full, clusters = analysis["realized"], analysis["clusters"]
    stim_id_to_row = analysis["stim_id_to_row"]
    umap_2d_full = analysis["umap_2d"]

    # 3. Alignement
    validate_perception_df(pdata)
    X, y_groove, y_complexity = align_embeddings_with_perception(
        X_full, pdata, stim_id_to_row=stim_id_to_row
    )

    valid_sids   = pdata.loc[pdata["stimulus_id"].isin(stim_id_to_row), "stimulus_id"]
    aligned_sids = np.array(pd.unique(valid_sids))
    aligned_rows = np.array([stim_id_to_row[sid] for sid in aligned_sids])

    umap_2d_aligned  = umap_2d_full[aligned_rows] if umap_2d_full is not None else None
    clusters_aligned = clusters[aligned_rows]

    # 4. Agrégation par stimulus (fix #4 — avant compute_local_geometry)
    agg_ratings      = pdata.groupby("stimulus_id")[["groove", "complexity"]].mean()
    y_groove_agg     = np.array([
        agg_ratings.loc[sid, "groove"]
        if sid in agg_ratings.index else np.nan
        for sid in aligned_sids
    ])
    y_complexity_agg = np.array([
        agg_ratings.loc[sid, "complexity"]
        if sid in agg_ratings.index else np.nan
        for sid in aligned_sids
    ])

    # 5. Normalisation + Géométrie
    X_norm = normalize(X)

    # Fix #4 — on passe les ratings agrégés, pas les bruts
    groove_geometry = compute_local_geometry(X_norm, y_groove_agg)

    # Fix #2 — test de Mantel (était importé mais jamais appelé)
    print("[perception_space] Test de permutation (Mantel)…")
    perm_result = permutation_test(X_norm, y_groove_agg, n_permutations=1000)
    print(
        f"[perception_space] Mantel r={perm_result['observed_r']:.3f}  "
        f"p={perm_result['p_value']:.4f}  "
        f"({'✓ sig.' if perm_result['significant'] else '✗ n.s.'})"
    )

    # 6. Figures
    print("[perception_space] Génération des figures alignées…")
    emb_2d = umap_2d_aligned if umap_2d_aligned is not None else _project_2d(X_norm)

    _safe_fig("umap_groove.png", plot_umap_groove, fig_dir,
              embedding=X_norm, groove=y_groove_agg, complexity=y_complexity_agg,
              clusters=clusters_aligned, umap_2d=umap_2d_aligned)

    _safe_fig("cluster_groove.png", plot_cluster_groove, fig_dir,
              embedding=X_norm, clusters=clusters_aligned, groove=y_groove_agg)

    _safe_fig("local_geometry_groove.png", plot_local_geometry, fig_dir,
              geometry=groove_geometry, embedding_2d=emb_2d, title_prefix="Groove")

    # Fix #2 — figure permutation_test maintenant générée
    _safe_fig("permutation_test.png", plot_permutation_test, fig_dir,
              perm_result=perm_result)

    # 7. Statistiques ICC et Variabilité
    print("[perception_space] Calcul des statistiques ICC…")
    wide_groove = ratings_to_wide(
        pdata,
        stim_col="stimulus_id",
        participant_col="participant_id",
        rating_col="groove",
    )
    icc_res = compute_icc(wide_groove)

    _safe_fig("icc_summary.png", plot_icc_summary, fig_dir, icc_groove=icc_res)

    stim_var_df = compute_per_stimulus_variance(
        pdata, stim_col="stimulus_id", rating_col="groove"
    )
    _safe_fig("stimulus_variance.png", plot_per_stimulus_variance, fig_dir,
              stim_variance=stim_var_df, stim_col="stimulus_id")

    return {
        "status":       "success",
        "mantel_r":     perm_result["observed_r"],
        "mantel_p":     perm_result["p_value"],
        "icc":          icc_res["icc"],
        "icc_interp":   icc_res["interpretation"],
    }


# ── HELPERS ──────────────────────────────────────────────

def _safe_fig(filename, fn, fig_dir, **kwargs):
    """
    Appelle fn(..., out_path=fig_dir/filename) en capturant les erreurs.

    Fix #7 : si GROOVE_DEBUG_FIGS=1, re-raise l'exception pour faciliter
    le débogage en développement. En production, l'erreur est loggée et
    le pipeline continue.
    """
    try:
        fn(**kwargs, out_path=fig_dir / filename)
    except Exception as e:
        if DEBUG_FIGS:
            raise
        print(f"[perception_space] figure '{filename}' failed : {e}")


def _project_2d(X):
    from sklearn.decomposition import PCA
    return PCA(n_components=2).fit_transform(X)