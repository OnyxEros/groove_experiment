"""
perception_space/run.py
=======================
Analyse géométrique et statistique du groove dans l'espace latent.

v3 :
    - umap_2d chargé depuis run_dir/embeddings/umap_2d.npy (ExportStep)
      et transmis à plot_umap_groove → cohérence visuelle avec spaces_figure
    - clusters_aligned transmis à plot_umap_groove → contours de clusters
      superposés aux ratings
    - _project_2d utilise umap_2d si disponible, fallback PCA sinon
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

from perception_space.core.loader     import load_analysis_run
from perception_space.core.align      import align_embeddings_with_perception
from perception_space.core.manifold   import compute_local_geometry
from perception_space.core.normalize  import normalize
from perception_space.core.validation import validate_perception_df
from perception_space.core.icc        import (
    compute_icc, ratings_to_wide, compute_per_stimulus_variance, icc_summary
)
from perception_space.core.stats      import (
    kruskal_by_condition, permutation_test, compute_condition_stats
)

from perception_space.viz.umap_groove    import plot_umap_groove
from perception_space.viz.cluster_groove import plot_cluster_groove
from perception_space.viz.geometry_plots import (
    plot_local_geometry, plot_permutation_test, plot_condition_stats
)
from perception_space.viz.icc_plot import (
    plot_icc_summary, plot_per_stimulus_variance
)

from config import get_current_run


def run_perception_space(perception_data: pd.DataFrame) -> dict:
    """
    Args:
        perception_data : DataFrame avec colonnes
            stimulus_id (str), groove, complexity (optionnel),
            participant_id (optionnel, requis pour ICC)
    """
    run_dir = get_current_run()
    out_dir = run_dir / "perception_space"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[perception_space] run_dir → {run_dir}")
    print(f"[perception_space] {len(perception_data)} entrées dans perception_data")

    # ── 1. Chargement embeddings + mapping ───────────────
    analysis       = load_analysis_run(run_dir)
    X_full         = analysis["realized"]
    clusters       = analysis["clusters"]
    stim_id_to_row = analysis["stim_id_to_row"]
    umap_2d_full   = analysis["umap_2d"]   # None si umap_2d.npy absent

    if umap_2d_full is not None:
        print("[perception_space] umap_2d chargé depuis run → cohérence visuelle")
    else:
        print("[perception_space] umap_2d absent → fallback PCA dans les figures")

    # ── 2. Normalisation stim_id ─────────────────────────
    pdata = perception_data.copy()
    pdata["stimulus_id"] = pdata["stimulus_id"].astype(str)

    if not pdata["stimulus_id"].isin(stim_id_to_row).any():
        def _normalize_stim_id(sid: str) -> str:
            try:
                return f"stim_{int(sid):04d}"
            except ValueError:
                return sid
        pdata["stimulus_id"] = pdata["stimulus_id"].apply(_normalize_stim_id)
        print("[perception_space] stim_id normalisés vers format stim_XXXX")

    validate_perception_df(pdata)

    # ── 3. Alignement embeddings × ratings ───────────────
    X, y_groove, y_complexity = align_embeddings_with_perception(
        X_full, pdata, stim_id_to_row=stim_id_to_row
    )

    n_aligned = len(X)
    print(f"[perception_space] {n_aligned} stimuli alignés")

    if n_aligned < 5:
        raise ValueError(
            f"Seulement {n_aligned} stimuli alignés — "
            "vérifie que les stim_id dans Supabase correspondent au run courant."
        )

    # ── 4. Alignement umap_2d et clusters sur les stim alignés ──
    # On récupère les row indices des stimuli alignés pour extraire
    # les bonnes lignes de umap_2d et clusters.
    aligned_sids = pdata["stimulus_id"].values[
        pdata["stimulus_id"].isin(stim_id_to_row).values
    ]
    aligned_rows = np.array([stim_id_to_row[sid] for sid in aligned_sids])

    umap_2d_aligned = None
    if umap_2d_full is not None and len(aligned_rows) == n_aligned:
        umap_2d_aligned = umap_2d_full[aligned_rows]

    clusters_aligned = (
        clusters[aligned_rows]
        if len(aligned_rows) == n_aligned
        else np.zeros(n_aligned, dtype=int)
    )

    # ── 5. Normalisation ──────────────────────────────────
    X_norm = normalize(X)

    # ── 6. ICC inter-participants ─────────────────────────
    icc_groove_result     = None
    icc_complexity_result = None
    stim_variance_groove  = None

    has_participants = "participant_id" in pdata.columns

    if has_participants:
        print("[perception_space] Calcul ICC inter-participants…")
        try:
            wide_groove = ratings_to_wide(
                pdata,
                stim_col="stimulus_id",
                participant_col="participant_id",
                rating_col="groove",
            )
            if wide_groove.shape[0] >= 3 and wide_groove.shape[1] >= 2:
                icc_groove_result = compute_icc(wide_groove, model="ICC2")
                icc_summary(icc_groove_result)

            if "complexity" in pdata.columns:
                wide_complexity = ratings_to_wide(
                    pdata,
                    stim_col="stimulus_id",
                    participant_col="participant_id",
                    rating_col="complexity",
                )
                if wide_complexity.shape[0] >= 3 and wide_complexity.shape[1] >= 2:
                    icc_complexity_result = compute_icc(wide_complexity, model="ICC2")

        except Exception as e:
            print(f"[perception_space] ⚠️  ICC failed : {e}")

        try:
            stim_variance_groove = compute_per_stimulus_variance(
                pdata,
                stim_col="stimulus_id",
                rating_col="groove",
                participant_col="participant_id",
            )
        except Exception as e:
            print(f"[perception_space] ⚠️  per-stimulus variance failed : {e}")

    else:
        print("[perception_space] ⚠️  participant_id absent — ICC ignoré")

    # ── 7. Géométrie locale ───────────────────────────────
    print("[perception_space] Calcul géométrie locale…")
    groove_geometry     = compute_local_geometry(X_norm, y_groove)
    complexity_geometry = compute_local_geometry(X_norm, y_complexity)

    print(
        f"[perception_space] k_effective groove={groove_geometry['k_effective']}  "
        f"complexity={complexity_geometry['k_effective']}"
    )

    # ── 8. Test de permutation ────────────────────────────
    print("[perception_space] Test de permutation…")
    perm_result = None
    try:
        perm_result = permutation_test(X_norm, y_groove, n_permutations=1000)
        sig = "★" if perm_result["significant"] else "n.s."
        print(
            f"[perception_space] permutation test : "
            f"r={perm_result['observed_r']:.3f}  p={perm_result['p_value']:.3f}  {sig}"
        )
    except Exception as e:
        print(f"[perception_space] ⚠️  permutation test failed : {e}")

    # ── 9. Stats par condition de design ──────────────────
    kruskal_results    = None
    condition_stats_df = None
    groove_col_joint   = "groove_mean"

    try:
        from config import METADATA_PATH
        meta = pd.read_csv(METADATA_PATH)

        if "stim_id" not in meta.columns and "id" in meta.columns:
            meta["stim_id"] = meta["id"].apply(lambda i: f"stim_{int(i):04d}")
        meta["stim_id"] = meta["stim_id"].astype(str)

        agg = (
            pdata
            .groupby("stimulus_id")["groove"]
            .mean()
            .reset_index()
            .rename(columns={"stimulus_id": "stim_id", "groove": groove_col_joint})
        )
        agg["stim_id"] = agg["stim_id"].astype(str)

        df_joint = meta.merge(agg, on="stim_id", how="inner")

        if not df_joint.empty:
            cond_cols = [c for c in ["S_mv", "D_mv", "E", "P"] if c in df_joint.columns]
            kruskal_results = kruskal_by_condition(
                df_joint,
                groove_col=groove_col_joint,
                condition_cols=cond_cols,
            )
            condition_stats_df = df_joint
            print("\n[perception_space] Résultats tests par condition :")
            print(kruskal_results.to_string(index=False))

    except Exception as e:
        print(f"[perception_space] ⚠️  stats par condition failed : {e}")

    # ── 10. Sauvegarde .npy ───────────────────────────────
    np.save(out_dir / "groove_local_mean.npy",      groove_geometry["local_mean"])
    np.save(out_dir / "groove_local_std.npy",       groove_geometry["local_std"])
    np.save(out_dir / "groove_local_slope.npy",     groove_geometry["local_slope"])
    np.save(out_dir / "groove_local_coherence.npy", groove_geometry["local_coherence"])

    np.save(out_dir / "complexity_local_mean.npy",      complexity_geometry["local_mean"])
    np.save(out_dir / "complexity_local_std.npy",       complexity_geometry["local_std"])
    np.save(out_dir / "complexity_local_slope.npy",     complexity_geometry["local_slope"])
    np.save(out_dir / "complexity_local_coherence.npy", complexity_geometry["local_coherence"])

    # ── 11. summary.json ──────────────────────────────────
    summary = {
        "n_stimuli_aligned":   n_aligned,
        "k_effective_groove":  int(groove_geometry["k_effective"]),
        "k_effective_complex": int(complexity_geometry["k_effective"]),
        "has_icc":             icc_groove_result is not None,
        "has_permutation":     perm_result is not None,
        "has_condition_stats": kruskal_results is not None,
        "used_umap_from_run":  umap_2d_aligned is not None,
    }

    if icc_groove_result:
        summary["icc_groove"] = {
            k: v for k, v in icc_groove_result.items()
            if k != "permutation_dist"
        }
    if icc_complexity_result:
        summary["icc_complexity"] = {
            k: v for k, v in icc_complexity_result.items()
            if k != "permutation_dist"
        }
    if perm_result:
        summary["permutation_test"] = {
            k: v for k, v in perm_result.items()
            if k != "permutation_dist"
        }
    if kruskal_results is not None and not kruskal_results.empty:
        summary["condition_tests"] = kruskal_results.to_dict(orient="records")

    summary.update(analysis["summary"])

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=_json_serializable)

    print(f"\n[perception_space] résultats → {out_dir}")

    # ── 12. Figures ───────────────────────────────────────
    print("[perception_space] Génération des figures…")

    # Projection 2D pour les figures de géométrie locale
    # (utilise umap_2d_aligned si dispo, sinon PCA sur X_norm)
    emb_2d = umap_2d_aligned if umap_2d_aligned is not None \
        else _project_2d(X_norm, run_dir=run_dir)

    # Figure centrale : groove superposé sur l'UMAP du run
    _safe_fig("umap_groove.png", plot_umap_groove, fig_dir,
              embedding=X_norm,
              groove=y_groove,
              complexity=y_complexity,
              clusters=clusters_aligned,
              umap_2d=umap_2d_aligned)

    _safe_fig("cluster_groove.png", plot_cluster_groove, fig_dir,
              embedding=X_norm,
              clusters=clusters_aligned,
              groove=y_groove)

    _safe_fig("local_geometry_groove.png", plot_local_geometry, fig_dir,
              geometry=groove_geometry,
              embedding_2d=emb_2d,
              title_prefix="Groove")

    _safe_fig("local_geometry_complexity.png", plot_local_geometry, fig_dir,
              geometry=complexity_geometry,
              embedding_2d=emb_2d,
              title_prefix="Complexity")

    if perm_result and perm_result.get("permutation_dist"):
        _safe_fig("permutation_test.png", plot_permutation_test, fig_dir,
                  perm_result=perm_result)

    if icc_groove_result:
        _safe_fig("icc_summary.png", plot_icc_summary, fig_dir,
                  icc_groove=icc_groove_result,
                  icc_complexity=icc_complexity_result)

    if stim_variance_groove is not None and not stim_variance_groove.empty:
        _safe_fig("per_stimulus_variance.png", plot_per_stimulus_variance, fig_dir,
                  stim_variance=stim_variance_groove)

    if condition_stats_df is not None and kruskal_results is not None:
        _safe_fig("condition_stats.png", plot_condition_stats, fig_dir,
                  condition_stats=condition_stats_df,
                  anova_results=kruskal_results,
                  groove_col=groove_col_joint)

    print(f"[perception_space] figures → {fig_dir}")

    return {
        "groove":           groove_geometry,
        "complexity":       complexity_geometry,
        "clusters":         clusters_aligned,
        "summary":          summary,
        "icc_groove":       icc_groove_result,
        "permutation_test": perm_result,
        "umap_2d_aligned":  umap_2d_aligned,
    }


# ── Helpers ───────────────────────────────────────────────

def _project_2d(X: np.ndarray, run_dir: Path | None = None) -> np.ndarray:
    """
    Fallback PCA si umap_2d_aligned n'est pas disponible.
    Normalement umap_2d est récupéré depuis load_analysis_run.
    """
    if X.shape[1] <= 2:
        return X

    # Tente UMAP à la volée (cohérence relative)
    try:
        import umap as umap_lib
        reducer = umap_lib.UMAP(
            n_components=2, metric="cosine",
            random_state=42, n_neighbors=min(15, len(X) - 1), min_dist=0.1,
        )
        print("[perception_space] projection 2D : UMAP à la volée (fallback)")
        return reducer.fit_transform(X)
    except ImportError:
        pass

    from sklearn.decomposition import PCA
    print("[perception_space] projection 2D : PCA fallback")
    return PCA(n_components=2, random_state=42).fit_transform(X)


def _safe_fig(filename, fn, fig_dir, **kwargs):
    try:
        fn(**kwargs, out_path=fig_dir / filename)
    except Exception as e:
        print(f"[perception_space] ⚠️  figure '{filename}' failed : {e}")


def _json_serializable(obj):
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray):     return obj.tolist()
    if isinstance(obj, bool):           return bool(obj)
    raise TypeError(f"Not serializable: {type(obj)}")