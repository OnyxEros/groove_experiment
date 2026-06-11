"""
perception_space/run.py  — VERSION CORRIGÉE POLARITÉ PUSH/PULL
===============================================================
Rétablit le sens physique standard pour le désalignement (P) :
    P > 0  →  Rushing (Avance temporelle du hi-hat)
    P < 0  →  Laid-back (Retard temporel du hi-hat)

v4.1 : logs console complets pour chaque figure (valeurs représentées
       + récapitulatif fichiers produits avec tailles).
"""

from __future__ import annotations

import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

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

N_MIN_PARTICIPANT_RESPONSES = 10
N_MIN_STIMULUS_RESPONSES    = 2
DEBUG_FIGS = os.getenv("GROOVE_DEBUG_FIGS", "0") == "1"

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES VISUELLES
# ─────────────────────────────────────────────────────────────────────────────
W = 72

def _sep(c="─"): return c * W
def _title(t, c="═"): return f"{c*2}  {t}  {c * max(0, W - len(t) - 4)}"
def _sub(t): return f"\n  ── {t} {'─' * max(0, W - len(t) - 6)}"
def _ok(t):   return f"  ✔  {t}"
def _warn(t): return f"  ⚠  {t}"
def _info(t): return f"  ℹ  {t}"
def _arrow(t):return f"  →  {t}"

def _bar(v, scale=1.0, w=20, signed=False):
    if math.isnan(float(v)): return "?" * w
    half = w // 2
    n = min(int(abs(v) / (scale + 1e-9) * (half if signed else w)), half if signed else w)
    if signed:
        return (" " * (half - n) + "█" * n) if v < 0 else (" " * half + "█" * n)
    return "█" * n + "░" * (w - n)

def _pstars(p):
    if p < 0.001: return "★★★"
    if p < 0.01:  return "★★ "
    if p < 0.05:  return "★  "
    if p < 0.10:  return "†  "
    return "   "

def _fmt(v, d=3):
    if isinstance(v, float) and math.isnan(v): return "nan"
    return f"{v:.{d}f}"

def _hbar(v, vmin, vmax, w=24):
    """Barre horizontale normalisée entre vmin et vmax."""
    if vmax <= vmin: return "░" * w
    ratio = max(0.0, min(1.0, (float(v) - vmin) / (vmax - vmin)))
    n = int(ratio * w)
    return "█" * n + "░" * (w - n)

def _matrix_row(label, values, fmt="+.3f", col_w=9):
    """Formate une ligne de matrice : label + valeurs alignées."""
    row = label.ljust(6)
    for v in values:
        row += f"{v:{fmt}}".ljust(col_w)
    return row


# ─────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────────────────────────────────────

def run_perception_space(perception_data: pd.DataFrame) -> dict:
    run_dir = get_current_run()
    out_dir = run_dir / "perception_space"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print()
    print(_sep("═"))
    print(_title("📊  ANALYSE DE L'ESPACE PERCEPTIF DU GROOVE"))
    print(_sep("═"))
    print(f"  run_dir → {run_dir}")
    if DEBUG_FIGS:
        print(_warn("DEBUG_FIGS=1 — les erreurs de figures seront levées"))

    fig_errors: list[str] = []

    # ── 1. Nettoyage ─────────────────────────────────────────────────────────
    print(_sub("1. Nettoyage des données perceptives"))
    pdata = perception_data.copy()
    pdata["stimulus_id"] = pdata["stimulus_id"].astype(str)
    n_raw = len(pdata)

    if "participant_id" in pdata.columns:
        pdata = pdata.sort_values("stimulus_id")
        n_before_dedup = len(pdata)
        pdata = pdata.drop_duplicates(subset=["participant_id", "stimulus_id"], keep="first")
        n_dedup = n_before_dedup - len(pdata)

        resp_per_p = pdata.groupby("participant_id").size()
        ghosts = resp_per_p[resp_per_p < N_MIN_PARTICIPANT_RESPONSES].index.tolist()
        pdata = pdata[~pdata["participant_id"].isin(ghosts)]
        n_after = len(pdata)

        n_participants = pdata["participant_id"].nunique()
        n_stimuli_raw  = pdata["stimulus_id"].nunique()

        print()
        print(f"  Réponses brutes        : {n_raw}")
        print(f"  Doublons supprimés     : {n_dedup}")
        print(f"  Participants fantômes  : {len(ghosts)}  (< {N_MIN_PARTICIPANT_RESPONSES} réponses)")
        if ghosts:
            print(f"  IDs exclus             : {ghosts}")
        print(f"  Réponses conservées    : {n_after}")
        print(f"  Participants actifs    : {n_participants}")
        print(f"  Stimuli couverts       : {n_stimuli_raw}")

        print()
        print(f"  Réponses par participant :")
        resp_sorted = resp_per_p[~resp_per_p.index.isin(ghosts)].sort_values(ascending=False)
        for pid, n_r in resp_sorted.items():
            bar = _hbar(n_r, resp_sorted.min(), resp_sorted.max(), w=18)
            print(f"    {str(pid):<20} {n_r:>4}  [{bar}]")
        print(f"  μ = {resp_sorted.mean():.1f}  σ = {resp_sorted.std():.1f}  "
              f"min = {resp_sorted.min()}  max = {resp_sorted.max()}")

        if "groove" in pdata.columns:
            g_vals = pdata["groove"].dropna()
            print()
            print(f"  Distribution groove brut (toutes réponses, N={len(g_vals)}) :")
            bins = [1, 2, 3, 4, 5, 6, 7, 8]
            counts, _ = np.histogram(g_vals, bins=bins)
            scale = max(counts) + 1
            for lo, hi, c in zip(bins[:-1], bins[1:], counts):
                bar = "█" * int(c / scale * 24)
                print(f"    [{lo}–{hi})  {bar:<24}  n={c:>4}  ({c/len(g_vals)*100:.1f}%)")
            print(f"  μ={g_vals.mean():.3f}  σ={g_vals.std():.3f}  "
                  f"médiane={g_vals.median():.3f}  "
                  f"min={g_vals.min():.1f}  max={g_vals.max():.1f}")

    cov = coverage_report(
        pdata, stim_col="stimulus_id",
        participant_col="participant_id" if "participant_id" in pdata.columns else "stimulus_id",
        rating_col="groove",
        n_min=N_MIN_STIMULUS_RESPONSES,
    )
    print()
    print(f"  Couverture stimuli     : {cov['n_covered']}/{cov['n_total']}  ({cov['coverage_pct']:.1f}%)")
    print(f"  Réponses/stimulus      : μ={_fmt(cov['mean_responses'], 1)}  médiane={_fmt(cov['median_responses'], 1)}")
    fill_pct = cov["n_covered"] / max(cov["n_total"], 1) * 100
    fill_bar = _hbar(fill_pct, 0, 100, w=30)
    print(f"  Taux remplissage       : [{fill_bar}]  {fill_pct:.1f}%")
    if cov["n_excluded"] > 0:
        print(_warn(f"{cov['n_excluded']} stimuli exclus (< {N_MIN_STIMULUS_RESPONSES} réponses) : "
                    f"{cov['excluded_stims'][:5]}{'…' if len(cov['excluded_stims']) > 5 else ''}"))

    # ── 1.5 Figures exploratoires ─────────────────────────────────────────────
    for col in ["musical_background", "experience", "style"]:
        if col in pdata.columns:
            _safe_fig(f"effect_{col}.png", plot_condition_effects, fig_dir,
                      fig_errors=fig_errors,
                      df=pdata, condition=col, target="groove")

    # ── 2. Chargement embeddings ──────────────────────────────────────────────
    print(_sub("2. Chargement du run d'analyse"))
    analysis = load_analysis_run(run_dir)
    X_full, clusters = analysis["realized"], analysis["clusters"]
    stim_id_to_row = analysis["stim_id_to_row"]
    umap_2d_full   = analysis["umap_2d"]

    from config import apply_polarity_fix_array
    from analysis.embeddings.realized import RealizedEmbedding
    X_full = apply_polarity_fix_array(X_full, RealizedEmbedding.COLS)

    n_embed    = X_full.shape[0]
    n_clusters = len(np.unique(clusters))

    print()
    print(f"  Embeddings réalisés    : {n_embed} × {X_full.shape[1]}")
    print(f"  Clusters               : {n_clusters}  (labels : {sorted(np.unique(clusters).tolist())})")
    print(f"  UMAP 2D disponible     : {'oui' if umap_2d_full is not None else 'non (fallback PCA)'}")

    cols_emb = RealizedEmbedding.COLS if hasattr(RealizedEmbedding, "COLS") else ["D", "S", "E", "P"]
    print()
    print(f"  Statistiques de X_full (embeddings réalisés standardisés) :")
    col_w = 9
    header = "  " + "Dim".ljust(6) + "μ".rjust(col_w) + "σ".rjust(col_w) + \
             "min".rjust(col_w) + "med".rjust(col_w) + "max".rjust(col_w)
    print(header)
    print("  " + "─" * (6 + col_w * 5))
    for i, col in enumerate(cols_emb if len(cols_emb) == X_full.shape[1] else range(X_full.shape[1])):
        label = col if isinstance(col, str) else f"dim{col}"
        v = X_full[:, i]
        print(f"  {label:<6} {np.mean(v):>{col_w}.3f} {np.std(v):>{col_w}.3f} "
              f"{np.min(v):>{col_w}.3f} {np.median(v):>{col_w}.3f} {np.max(v):>{col_w}.3f}")

    _print_cluster_sizes(clusters)

    # ── 3. Alignement ─────────────────────────────────────────────────────────
    print(_sub("3. Alignement embeddings × ratings"))
    validate_perception_df(pdata)

    X, y_groove, y_complexity = align_embeddings_with_perception(
        X_full, pdata, stim_id_to_row=stim_id_to_row
    )

    valid_sids   = pdata.loc[pdata["stimulus_id"].isin(stim_id_to_row), "stimulus_id"]
    aligned_sids = np.array(pd.unique(valid_sids))
    aligned_rows = np.array([stim_id_to_row[sid] for sid in aligned_sids])

    umap_2d_aligned  = umap_2d_full[aligned_rows] if umap_2d_full is not None else None
    clusters_aligned = clusters[aligned_rows]
    X_agg = X_full[aligned_rows]

    print()
    print(f"  Stimuli alignés        : {len(aligned_sids)}/{n_embed}")
    n_missing_align = n_embed - len(aligned_sids)
    if n_missing_align > 0:
        print(_warn(f"{n_missing_align} stimuli sans rating — exclus de l'analyse"))

    print()
    print(f"  Stimuli alignés par cluster :")
    unique_cl, counts_cl = np.unique(clusters_aligned, return_counts=True)
    total_al = len(clusters_aligned)
    for c, n in zip(unique_cl, counts_cl):
        bar = _hbar(n, 0, total_al, w=16)
        print(f"    C{c}  {n:>4}  ({n/total_al*100:.1f}%)  [{bar}]")

    # ── 4. Agrégation ─────────────────────────────────────────────────────────
    print(_sub("4. Agrégation par stimulus"))
    agg_ratings      = pdata.groupby("stimulus_id")[["groove", "complexity"]].mean()
    y_groove_agg     = np.array([
        agg_ratings.loc[sid, "groove"]     if sid in agg_ratings.index else np.nan
        for sid in aligned_sids
    ])
    y_complexity_agg = np.array([
        agg_ratings.loc[sid, "complexity"] if sid in agg_ratings.index else np.nan
        for sid in aligned_sids
    ])

    n_valid_groove = int(np.isfinite(y_groove_agg).sum())
    g_valid = y_groove_agg[np.isfinite(y_groove_agg)]

    print()
    print(f"  Groove agrégé          : {n_valid_groove} stimuli valides")
    print(f"    μ        = {_fmt(np.nanmean(y_groove_agg))}")
    print(f"    σ        = {_fmt(np.nanstd(y_groove_agg))}")
    print(f"    médiane  = {_fmt(np.nanmedian(y_groove_agg))}")
    print(f"    min      = {_fmt(np.nanmin(y_groove_agg))}")
    print(f"    max      = {_fmt(np.nanmax(y_groove_agg))}")
    print(f"    p25      = {_fmt(float(np.nanpercentile(y_groove_agg, 25)))}")
    print(f"    p75      = {_fmt(float(np.nanpercentile(y_groove_agg, 75)))}")
    print(f"    IQR      = {_fmt(float(np.nanpercentile(y_groove_agg, 75) - np.nanpercentile(y_groove_agg, 25)))}")

    _print_groove_distribution(y_groove_agg)

    if len(g_valid) > 0:
        sort_idx = np.argsort(y_groove_agg)
        sort_idx_valid = [i for i in sort_idx if np.isfinite(y_groove_agg[i])]
        print()
        print(f"  Top 5 stimuli — groove le plus élevé :")
        for i in sort_idx_valid[-5:][::-1]:
            sid = aligned_sids[i]
            bar = _hbar(y_groove_agg[i], 1, 7, w=16)
            print(f"    {str(sid):<20}  groove={y_groove_agg[i]:.3f}  [{bar}]")
        print(f"  Top 5 stimuli — groove le plus faible :")
        for i in sort_idx_valid[:5]:
            sid = aligned_sids[i]
            bar = _hbar(y_groove_agg[i], 1, 7, w=16)
            print(f"    {str(sid):<20}  groove={y_groove_agg[i]:.3f}  [{bar}]")

    if np.isfinite(y_complexity_agg).sum() > 0:
        print()
        print(f"  Complexité agrégée     : μ={_fmt(np.nanmean(y_complexity_agg))}  "
              f"σ={_fmt(np.nanstd(y_complexity_agg))}  "
              f"min={_fmt(np.nanmin(y_complexity_agg))}  max={_fmt(np.nanmax(y_complexity_agg))}")
        r_gc = float(np.corrcoef(
            y_groove_agg[np.isfinite(y_groove_agg) & np.isfinite(y_complexity_agg)],
            y_complexity_agg[np.isfinite(y_groove_agg) & np.isfinite(y_complexity_agg)]
        )[0, 1]) if np.isfinite(y_complexity_agg).sum() > 2 else float("nan")
        print(f"  Corrélation groove×complexité : r = {r_gc:+.3f}")

    # ── 5. Normalisation + Géométrie locale ───────────────────────────────────
    print(_sub("5. Géométrie locale de l'espace perceptif"))
    X_norm = normalize(X_agg)

    print()
    print(f"  Effet RobustScaler (avant → après) :")
    print(f"  {'Dim':<6}  {'μ_avant':>9}  {'μ_après':>9}  {'σ_avant':>9}  {'σ_après':>9}")
    print("  " + "─" * 46)
    for i, col in enumerate(cols_emb if len(cols_emb) == X_agg.shape[1] else range(X_agg.shape[1])):
        label = col if isinstance(col, str) else f"dim{col}"
        print(f"  {label:<6}  {np.mean(X_agg[:, i]):>9.3f}  {np.mean(X_norm[:, i]):>9.3f}  "
              f"{np.std(X_agg[:, i]):>9.3f}  {np.std(X_norm[:, i]):>9.3f}")

    groove_geometry = compute_local_geometry(X_norm, y_groove_agg)
    _print_geometry_report(groove_geometry)

    print()
    print(f"  Corrélations X_norm[:,i] × groove_agg :")
    for i, col in enumerate(cols_emb if len(cols_emb) == X_norm.shape[1] else range(X_norm.shape[1])):
        label = col if isinstance(col, str) else f"dim{col}"
        mask = np.isfinite(y_groove_agg)
        if mask.sum() > 2:
            r = float(np.corrcoef(X_norm[mask, i], y_groove_agg[mask])[0, 1])
            bar = _bar(abs(r), 1.0, w=16)
            flag = "★" if abs(r) > 0.3 else ("·" if abs(r) > 0.15 else " ")
            print(f"    {label:<6}  r = {r:+.3f}  [{bar}]  {flag}")

    # ── 6. Test de Mantel ─────────────────────────────────────────────────────
    print(_sub("6. Test de permutation (Mantel)"))
    print(f"\n  N permutations         : 1 000")
    perm_result = permutation_test(X_norm, y_groove_agg, n_permutations=1000)
    _print_mantel_report(perm_result)

    if perm_result.get("permutation_dist"):
        null = np.array(perm_result["permutation_dist"])
        obs  = perm_result["observed_r"]
        p95, p99 = np.percentile(null, 95), np.percentile(null, 99)
        z_score = (obs - null.mean()) / (null.std() + 1e-10)
        print()
        print(f"  Distribution nulle (1000 permutations) :")
        print(f"    μ_null   = {null.mean():+.4f}   σ_null  = {null.std():.4f}")
        print(f"    p95      = {p95:+.4f}   p99     = {p99:+.4f}")
        print(f"    r_observé= {obs:+.4f}   z-score = {z_score:+.2f}")
        print()
        hist_counts, bin_edges = np.histogram(null, bins=12)
        scale_h = max(hist_counts) + 1
        print(f"  Histogramme distribution nulle :")
        for lo, hi, c in zip(bin_edges[:-1], bin_edges[1:], hist_counts):
            bar = "█" * int(c / scale_h * 22)
            marker = " ← r_obs" if lo <= obs < hi else ""
            print(f"    [{lo:+.3f},{hi:+.3f})  {bar:<22}  n={c}{marker}")

    # ── 7. Tests statistiques par condition ───────────────────────────────────
    print(_sub("7. Tests statistiques par paramètre génératif"))

    try:
        from regression.data.loader import load_aggregated
        df_agg, _, _, _ = load_aggregated(feature_set="design", normalize=False)
        df_test = df_agg.copy()

        from config import apply_polarity_fix_df
        df_test = apply_polarity_fix_df(df_test)

        df_test["groove_mean"] = df_test.get("groove_mean", pd.Series(dtype=float))
        kruskal_results = kruskal_by_condition(df_test, groove_col="groove_mean")
        _print_kruskal_report(kruskal_results)

        cond_cols = [c for c in ["S_mv", "D_mv", "E_mv", "P_mv"] if c in df_test.columns]
        for cond in cond_cols:
            lvls = sorted(df_test[cond].dropna().unique())
            print()
            print(f"  {cond} — groove moyen par niveau :")
            print(f"  {'Niveau':>8}  {'n':>5}  {'μ':>7}  {'σ':>7}  {'médiane':>8}  Barre")
            print("  " + "─" * 54)
            all_means = [df_test.loc[df_test[cond] == lvl, "groove_mean"].dropna().mean()
                         for lvl in lvls if not df_test.loc[df_test[cond] == lvl, "groove_mean"].dropna().empty]
            g_min = min(all_means) if all_means else 1.0
            g_max = max(all_means) if all_means else 7.0
            for lvl in lvls:
                sub = df_test.loc[df_test[cond] == lvl, "groove_mean"].dropna()
                if sub.empty: continue
                bar = _hbar(sub.mean(), g_min - 0.1, g_max + 0.1, w=14)
                print(f"  {str(lvl):>8}  {len(sub):>5}  {sub.mean():>7.3f}  "
                      f"{sub.std():>7.3f}  {sub.median():>8.3f}  [{bar}]")

        sig_conditions = kruskal_results[kruskal_results["significant"]]["condition"].tolist()
        for cond in sig_conditions:
            print()
            print(f"  Post-hoc Bonferroni — {cond}")
            try:
                ph = post_hoc_bonferroni(df_test, groove_col="groove_mean", condition_col=cond)
                _print_posthoc_report(ph, cond)
            except Exception as e:
                print(_warn(f"  Post-hoc échoué : {e}"))
    except Exception as e:
        print(_warn(f"Tests par condition ignorés : {e}"))

    # ── 8. ICC ────────────────────────────────────────────────────────────────
    print(_sub("8. ICC inter-participants — fiabilité"))
    wide_groove = ratings_to_wide(
        pdata,
        stim_col="stimulus_id",
        participant_col="participant_id",
        rating_col="groove",
    )
    icc_res = compute_icc(wide_groove)
    _print_icc_report(icc_res, label="Groove")

    print()
    print(f"  Décomposition de la variance :")
    ms_r, ms_e = icc_res["MS_r"], icc_res["MS_e"]
    icc_val = icc_res["icc"]
    var_stimuli_pct = icc_val * 100
    var_residu_pct  = (1 - icc_val) * 100
    bar_s = _hbar(var_stimuli_pct, 0, 100, w=20)
    bar_r = _hbar(var_residu_pct, 0, 100, w=20)
    print(f"    Variance stimuli   : [{bar_s}]  {var_stimuli_pct:.1f}%  (ICC)")
    print(f"    Variance résiduelle: [{bar_r}]  {var_residu_pct:.1f}%  (1 - ICC)")
    print(f"    MS_r               : {ms_r:.4f}")
    print(f"    MS_e               : {ms_e:.4f}")
    print(f"    Rapport MS_r/MS_e  : {ms_r / (ms_e + 1e-10):.3f}")
    print(f"    Complétude matrice : {icc_res['completeness_pct']:.1f}%  "
          f"({icc_res['n_stimuli']} stimuli × {icc_res['n_raters']} participants, "
          f"k̄={icc_res['k_bar']})")
    print(f"    F({icc_res['df1']}, {icc_res['df2']}) = {icc_res['F']:.3f}   "
          f"p = {icc_res['p_value']:.6f}  {_pstars(icc_res['p_value'])}")
    print(f"    IC95%  [{icc_res['ci95_low']:.3f} – {icc_res['ci95_high']:.3f}]  "
          f"(largeur = {icc_res['ci95_high'] - icc_res['ci95_low']:.3f})")

    # ── 9. Variabilité par stimulus ────────────────────────────────────────────
    print(_sub("9. Variabilité inter-participants par stimulus"))
    stim_var_df = compute_per_stimulus_variance(
        pdata, stim_col="stimulus_id", rating_col="groove"
    )
    _print_stimulus_variance_report(stim_var_df)

    if not stim_var_df.empty:
        std_vals = stim_var_df["std"].values
        print()
        print(f"  Distribution σ inter-participants par stimulus :")
        print(f"    μ_σ    = {std_vals.mean():.3f}   σ_σ  = {std_vals.std():.3f}")
        print(f"    min_σ  = {std_vals.min():.3f}   max_σ = {std_vals.max():.3f}")
        print(f"    p25_σ  = {np.percentile(std_vals, 25):.3f}   "
              f"p75_σ = {np.percentile(std_vals, 75):.3f}")
        n_high = int(np.sum(std_vals > 1.5))
        n_low  = int(np.sum(std_vals < 0.5))
        print(f"    σ > 1.5 (très ambigu)  : {n_high} stimuli  "
              f"({n_high/len(std_vals)*100:.1f}%)")
        print(f"    σ < 0.5 (très consensuel): {n_low} stimuli  "
              f"({n_low/len(std_vals)*100:.1f}%)")
        hist_s, bins_s = np.histogram(std_vals, bins=8)
        scale_s = max(hist_s) + 1
        print()
        print(f"  Histogramme σ :")
        for lo, hi, c in zip(bins_s[:-1], bins_s[1:], hist_s):
            bar = "█" * int(c / scale_s * 20)
            print(f"    [{lo:.2f},{hi:.2f})  {bar:<20}  n={c}")

    # ── 10. Figures ─────────────────────────────────────────────────────────
    print(_sub("10. Génération des figures"))
    print()
    emb_2d = umap_2d_aligned if umap_2d_aligned is not None else _project_2d(X_norm)

    # ── LOG fig : umap_groove ────────────────────────────────────────────────
    print(f"  {_sep()}")
    print(f"  [fig] umap_groove — données représentées")
    print(f"  {_sep('─')}")
    g_finite = y_groove_agg[np.isfinite(y_groove_agg)]
    print(f"  n stimuli          : {len(g_finite)}")
    print(f"  Groove range       : [{_fmt(float(g_finite.min()))} – {_fmt(float(g_finite.max()))}]")
    print(f"  Groove μ ± σ       : {_fmt(float(g_finite.mean()))} ± {_fmt(float(g_finite.std()))}")
    print(f"  Projection 2D      : {'UMAP (run)' if umap_2d_aligned is not None else 'PCA fallback'}")
    top5_idx = np.argsort(y_groove_agg)[-5:][::-1]
    print(f"  Top 5 groove (annotés) :")
    for i in top5_idx:
        if np.isfinite(y_groove_agg[i]):
            print(f"    {str(aligned_sids[i]):<20}  groove={y_groove_agg[i]:.3f}  "
                  f"umap=({emb_2d[i, 0]:.2f}, {emb_2d[i, 1]:.2f})")
    print(f"  Clusters représentés : {sorted(np.unique(clusters_aligned).tolist())}")
    print(f"  {_sep()}")

    _safe_fig("umap_groove.png", plot_umap_groove, fig_dir,
              fig_errors=fig_errors,
              embedding=X_norm, groove=y_groove_agg, complexity=y_complexity_agg,
              clusters=clusters_aligned, umap_2d=umap_2d_aligned)

    # ── LOG fig : cluster_groove ─────────────────────────────────────────────
    print(f"  {_sep()}")
    print(f"  [fig] cluster_groove — groove moyen par cluster")
    print(f"  {_sep('─')}")
    for c in np.unique(clusters_aligned):
        vals_c = y_groove_agg[clusters_aligned == c]
        vals_c = vals_c[np.isfinite(vals_c)]
        if len(vals_c) == 0:
            continue
        ci95_c = 1.96 * vals_c.std(ddof=1) / np.sqrt(len(vals_c)) if len(vals_c) > 1 else float("nan")
        print(f"  C{c}  n={len(vals_c):>4}  μ={_fmt(float(vals_c.mean()))}  "
              f"σ={_fmt(float(vals_c.std(ddof=1))) if len(vals_c) > 1 else 'n/a'}  "
              f"CI95=±{_fmt(ci95_c) if math.isfinite(ci95_c) else 'n/a'}")
    print(f"  {_sep()}")

    _safe_fig("cluster_groove.png", plot_cluster_groove, fig_dir,
              fig_errors=fig_errors,
              embedding=X_norm, clusters=clusters_aligned, groove=y_groove_agg)

    # ── LOG fig : local_geometry_groove ─────────────────────────────────────
    print(f"  {_sep()}")
    print(f"  [fig] local_geometry_groove — métriques représentées (4 panneaux)")
    print(f"  {_sep('─')}")
    for metric_key, metric_label in [
        ("local_mean",      "Moyenne locale"),
        ("local_std",       "Std locale"),
        ("local_agreement", "Accord local"),
        ("local_slope",     "Gradient slope"),
    ]:
        if metric_key not in groove_geometry:
            continue
        v = groove_geometry[metric_key]
        v_fin = v[np.isfinite(v)]
        if len(v_fin) == 0:
            continue
        print(f"  {metric_label:<22}  μ={_fmt(float(v_fin.mean())):<8}  "
              f"σ={_fmt(float(v_fin.std())):<8}  "
              f"[{_fmt(float(v_fin.min()))} – {_fmt(float(v_fin.max()))}]  "
              f"p25={_fmt(float(np.percentile(v_fin,25)))}  p75={_fmt(float(np.percentile(v_fin,75)))}")
    print(f"  k effectif : {groove_geometry.get('k_effective', '?')}")
    print(f"  {_sep()}")

    _safe_fig("local_geometry_groove.png", plot_local_geometry, fig_dir,
              fig_errors=fig_errors,
              geometry=groove_geometry, embedding_2d=emb_2d, title_prefix="Groove")

    # ── LOG fig : permutation_test ───────────────────────────────────────────
    print(f"  {_sep()}")
    print(f"  [fig] permutation_test — distribution nulle")
    print(f"  {_sep('─')}")
    null_arr = np.array(perm_result.get("permutation_dist", []))
    if len(null_arr) > 0:
        p95_fig = float(np.percentile(null_arr, 95))
        p99_fig = float(np.percentile(null_arr, 99))
        z_fig   = (perm_result["observed_r"] - null_arr.mean()) / (null_arr.std() + 1e-10)
        print(f"  r observé   : {perm_result['observed_r']:+.4f}  "
              f"{'> p99 ★★' if perm_result['observed_r'] > p99_fig else '> p95 ★' if perm_result['observed_r'] > p95_fig else ''}")
        print(f"  p-value     : {perm_result['p_value']:.4f}  {_pstars(perm_result['p_value'])}")
        print(f"  z-score     : {z_fig:+.2f}")
        print(f"  Null μ ± σ  : {null_arr.mean():+.4f} ± {null_arr.std():.4f}")
        print(f"  p95 / p99   : {p95_fig:.4f} / {p99_fig:.4f}")
        print(f"  N perm      : {perm_result['n_permutations']}  |  N paires : {perm_result['n_pairs']}")
        hist_f, bins_f = np.histogram(null_arr, bins=10)
        scale_f = max(hist_f) + 1
        obs_f = perm_result["observed_r"]
        print(f"  Histogramme distribution nulle :")
        for lo, hi, c in zip(bins_f[:-1], bins_f[1:], hist_f):
            bar = "█" * int(c / scale_f * 18)
            marker = " ← r_obs" if lo <= obs_f < hi else ""
            print(f"    [{lo:+.3f},{hi:+.3f})  {bar:<18}  n={c}{marker}")
    print(f"  {_sep()}")

    _safe_fig("permutation_test.png", plot_permutation_test, fig_dir,
              fig_errors=fig_errors,
              perm_result=perm_result)

    # ── LOG fig : icc_summary ────────────────────────────────────────────────
    print(f"  {_sep()}")
    print(f"  [fig] icc_summary — gauge ICC représentée")
    print(f"  {_sep('─')}")
    print(f"  ICC(2,1)        : {icc_res['icc']:.3f}  "
          f"[{icc_res['ci95_low']:.3f} – {icc_res['ci95_high']:.3f}]  "
          f"largeur IC={icc_res['ci95_high']-icc_res['ci95_low']:.3f}")
    print(f"  F({icc_res['df1']}, {icc_res['df2']})         : {icc_res['F']:.3f}   "
          f"p={icc_res['p_value']:.6f}  {_pstars(icc_res['p_value'])}")
    print(f"  Interprétation  : {icc_res['interpretation'].upper()}")
    print(f"  Complétude      : {icc_res['completeness_pct']:.1f}%  "
          f"({icc_res['n_stimuli']} stimuli × {icc_res['n_raters']} participants, k̄={icc_res['k_bar']})")
    print(f"  Variance stimuli: {icc_res['icc']*100:.1f}%  |  Résiduelle: {(1-icc_res['icc'])*100:.1f}%")
    print(f"  MS_r / MS_e     : {icc_res['MS_r']:.4f} / {icc_res['MS_e']:.4f}  "
          f"(rapport: {icc_res['MS_r']/(icc_res['MS_e']+1e-10):.3f})")
    for lo, hi, lbl in [(0.90, 1.00, "Excellente"), (0.75, 0.90, "Bonne"),
                         (0.50, 0.75, "Modérée"),    (0.00, 0.50, "Faible")]:
        marker = "◄" if lo <= icc_res["icc"] < hi else " "
        print(f"    {marker}  [{lo:.2f}–{hi:.2f}]  {lbl}")
    print(f"  {_sep()}")

    _safe_fig("icc_summary.png", plot_icc_summary, fig_dir,
              fig_errors=fig_errors,
              icc_groove=icc_res)

    # ── LOG fig : stimulus_variance ──────────────────────────────────────────
    print(f"  {_sep()}")
    print(f"  [fig] stimulus_variance — barres groove ± σ par stimulus")
    print(f"  {_sep('─')}")
    if not stim_var_df.empty:
        sv = stim_var_df["std"].values
        gm = stim_var_df["mean"].values
        print(f"  N stimuli       : {len(stim_var_df)}")
        print(f"  Groove μ range  : [{_fmt(float(gm.min()))} – {_fmt(float(gm.max()))}]")
        print(f"  σ : μ={_fmt(float(sv.mean()))}  σ={_fmt(float(sv.std()))}  "
              f"[{_fmt(float(sv.min()))} – {_fmt(float(sv.max()))}]")
        print(f"  p25/p50/p75 σ   : {_fmt(float(np.percentile(sv,25)))} / "
              f"{_fmt(float(np.percentile(sv,50)))} / "
              f"{_fmt(float(np.percentile(sv,75)))}")
        n_high_sv = int(np.sum(sv > 1.5))
        n_low_sv  = int(np.sum(sv < 0.5))
        print(f"  σ > 1.5 (ambigus)    : {n_high_sv}  ({n_high_sv/len(sv)*100:.1f}%)")
        print(f"  σ < 0.5 (consensuels): {n_low_sv}  ({n_low_sv/len(sv)*100:.1f}%)")
        print(f"  Top 3 ambigus : " + "  |  ".join(
            f"{row.get('stimulus_id', '?')} σ={row['std']:.2f} μ={row['mean']:.2f}"
            for _, row in stim_var_df.head(3).iterrows()
        ))
        print(f"  Top 3 consensuels : " + "  |  ".join(
            f"{row.get('stimulus_id', '?')} σ={row['std']:.2f} μ={row['mean']:.2f}"
            for _, row in stim_var_df.tail(3).iterrows()
        ))
    print(f"  {_sep()}")

    _safe_fig("stimulus_variance.png", plot_per_stimulus_variance, fig_dir,
              fig_errors=fig_errors,
              stim_variance=stim_var_df, stim_col="stimulus_id")

    # ── Récapitulatif fichiers produits ──────────────────────────────────────
    expected_figs = [
        "umap_groove.png",
        "cluster_groove.png",
        "local_geometry_groove.png",
        "permutation_test.png",
        "icc_summary.png",
        "stimulus_variance.png",
    ]
    print()
    print(f"  {'─'*52}")
    print(f"  Récapitulatif figures → {fig_dir}")
    print(f"  {'─'*52}")
    n_fig_ok = 0
    for fname in expected_figs:
        p_fig = fig_dir / fname
        if p_fig.exists():
            size_kb = p_fig.stat().st_size / 1024
            print(f"  ✔  {fname:<42}  ({size_kb:.0f} KB)")
            n_fig_ok += 1
        else:
            print(f"  ✗  {fname:<42}  MANQUANT")
    print(f"  {'─'*52}")
    print(f"  {n_fig_ok}/{len(expected_figs)} figures produites")

    # ── Rapport final ──────────────────────────────────────────────────────────
    print()
    print(_sep("═"))
    print(_title("✔  SYNTHÈSE PERCEPTION SPACE"))
    print(_sep("═"))
    _print_final_summary(perm_result, icc_res, groove_geometry, y_groove_agg,
                         y_complexity_agg, stim_var_df, fig_errors,
                         n_participants=pdata["participant_id"].nunique()
                         if "participant_id" in pdata.columns else None,
                         n_stimuli=cov["n_covered"],
                         n_obs=len(pdata))
    print(_sep("═"))
    print()

    return {
        "status":       "success" if not fig_errors else "partial",
        "fig_errors":   fig_errors,
        "mantel_r":     perm_result["observed_r"],
        "mantel_p":     perm_result["p_value"],
        "icc":          icc_res["icc"],
        "icc_interp":   icc_res["interpretation"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# BLOCS DE RAPPORT DÉTAILLÉS
# (identiques à la version précédente — aucun changement)
# ─────────────────────────────────────────────────────────────────────────────

def _print_cluster_sizes(clusters: np.ndarray) -> None:
    unique, counts = np.unique(clusters, return_counts=True)
    total = len(clusters)
    print()
    print(f"  {'Cluster':<10} {'n':>5}  {'%':>6}  {'Barre'}")
    print("  " + "─" * 40)
    for c, n in zip(unique, counts):
        pct = n / total * 100
        bar = _bar(n, total, w=16)
        print(f"  C{c:<9} {n:>5}  {pct:>5.1f}%  [{bar}]")


def _print_groove_distribution(y: np.ndarray) -> None:
    valid = y[np.isfinite(y)]
    if len(valid) == 0:
        return
    bins = np.linspace(1, 7, 7)
    counts, _ = np.histogram(valid, bins=bins)
    scale = max(counts) + 1
    print()
    print(f"  Distribution groove agrégé (1→7) :")
    for i, (lo, hi, c) in enumerate(zip(bins[:-1], bins[1:], counts)):
        bar = "█" * int(c / scale * 20)
        print(f"    [{lo:.0f}–{hi:.0f})  {bar:<20}  n={c}")


def _print_geometry_report(g: dict) -> None:
    k = g.get("k_effective", "?")
    lm  = g["local_mean"]
    ls  = g["local_std"]
    la  = g["local_agreement"]
    lsl = g["local_slope"]

    print()
    print(f"  k voisins effectif     : {k}")
    print()
    print(f"  {'Métrique':<22} {'μ':>8}  {'σ':>8}  {'min':>8}  {'p25':>8}  "
          f"{'med':>8}  {'p75':>8}  {'max':>8}")
    print("  " + "─" * 72)

    for label, arr in [
        ("local_mean",      lm),
        ("local_std",       ls),
        ("local_agreement", la),
        ("local_slope",     lsl),
    ]:
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            print(f"  {label:<22}  (vide)")
            continue
        print(f"  {label:<22} {np.mean(valid):>8.3f}  {np.std(valid):>8.3f}  "
              f"{np.min(valid):>8.3f}  {np.percentile(valid,25):>8.3f}  "
              f"{np.median(valid):>8.3f}  {np.percentile(valid,75):>8.3f}  "
              f"{np.max(valid):>8.3f}")

    print()
    agree_mean = float(np.nanmean(la))
    agree_std  = float(np.nanstd(la))
    bar_a = _hbar(agree_mean, 0, 1, w=20)
    if agree_mean > 0.7:
        print(_ok(f"Accord local élevé : μ={agree_mean:.3f} ± {agree_std:.3f}  [{bar_a}]"))
        print(_arrow("L'espace est perceptivement cohérent — voisinage acoustique ≈ voisinage perceptif"))
    elif agree_mean > 0.5:
        print(_info(f"Accord local modéré : μ={agree_mean:.3f} ± {agree_std:.3f}  [{bar_a}]"))
    else:
        print(_warn(f"Accord local faible : μ={agree_mean:.3f} ± {agree_std:.3f}  [{bar_a}]"))

    slope_mean = float(np.nanmean(np.abs(lsl)))
    slope_std  = float(np.nanstd(np.abs(lsl)))
    bar_sl = _hbar(slope_mean, 0, 0.5, w=20)
    print(_info(f"Gradient local |slope| : μ={slope_mean:.3f} ± {slope_std:.3f}  [{bar_sl}]  "
                + ("→ gradient significatif" if slope_mean > 0.15 else "→ espace relativement plat")))


def _print_mantel_report(r: dict) -> None:
    obs    = r["observed_r"]
    p      = r["p_value"]
    n_perm = r["n_permutations"]
    n_pairs= r["n_pairs"]
    sig    = r["significant"]

    null = np.array(r.get("permutation_dist", []))
    p95  = float(np.percentile(null, 95)) if len(null) > 0 else float("nan")
    p99  = float(np.percentile(null, 99)) if len(null) > 0 else float("nan")
    z    = (obs - null.mean()) / (null.std() + 1e-10) if len(null) > 0 else float("nan")

    print()
    print(f"  r observé              : {obs:+.4f}")
    print(f"  p-value                : {p:.4f}  {_pstars(p)}")
    print(f"  z-score                : {z:+.2f}")
    print(f"  N permutations         : {n_perm}")
    print(f"  N paires analysées     : {n_pairs}  (stimuli × stimuli)")
    print(f"  Distribution nulle p95 : {p95:.4f}")
    print(f"  Distribution nulle p99 : {p99:.4f}")
    print()

    bar_r = _hbar(abs(obs), 0, 0.5, w=20)
    if sig:
        print(_ok(f"Mantel SIGNIFICATIF (p={p:.4f})  r={obs:+.4f}  [{bar_r}]"))
        print(_arrow("La structure de l'espace latent est corrélée avec les différences de groove perçu."))
        print(_info("Les stimuli proches dans l'espace réalisé ont des grooves similaires."))
    else:
        print(_warn(f"Mantel NON significatif (p={p:.4f}) — la distance dans l'espace"))
        print(_arrow("latent ne prédit pas systématiquement les différences de groove."))

    if "warning" in r:
        print(_warn(r["warning"]))


def _print_kruskal_report(df: pd.DataFrame) -> None:
    if df.empty:
        print(_warn("Aucun résultat de test disponible."))
        return

    print()
    print(f"  {'Condition':<12} {'Test':<18} {'Stat':>8}  {'p':>8}  {'Sig':>4}  "
          f"{'η²':>8}  {'Taille effet':<12}  Normalité")
    print("  " + "─" * 78)

    for _, row in df.iterrows():
        sig = _pstars(row["p_value"])
        norm = "✔ ok" if row.get("normality_ok", True) else "✗ non-normale"
        p_s = f"{row['p_value']:.4f}"
        eta_s = f"{row['eta2']:.3f}"
        stat_s = f"{row['statistic']:.2f}"
        interp = row.get("interpretation", "")
        bar = _bar(row["eta2"], 0.15, w=8)

        print(f"  {row['condition']:<12} {row['test_used']:<18} {stat_s:>8}  "
              f"{p_s:>8}  {sig}  {eta_s:>8}  {interp:<12}  {norm}")
        if row["significant"]:
            print(f"  {'':12}   [{bar}] η²={row['eta2']:.3f} → effet {interp}")

    sig_df = df[df["significant"]]
    print()
    if len(sig_df) > 0:
        print(_ok(f"{len(sig_df)}/{len(df)} conditions atteignent p < 0.05 :"))
        for _, row in sig_df.iterrows():
            print(f"    ★  {row['condition']}  (η²={row['eta2']:.3f}, {row['interpretation']})")
    else:
        print(_warn("Aucune condition n'atteint p < 0.05 avec correction de groupe."))


def _print_posthoc_report(df: pd.DataFrame, condition: str) -> None:
    if df.empty:
        print(_warn("Aucune comparaison disponible."))
        return

    print()
    print(f"  {'Niveaux':<16} {'μ_A':>7}  {'μ_B':>7}  {'Δμ':>8}  {'t':>7}  "
          f"{'df':>6}  {'p_raw':>8}  {'p_bonf':>8}  {'Sig':>4}  {'|d|':>6}")
    print("  " + "─" * 85)

    for _, row in df.iterrows():
        pair = f"{row['level_a']} vs {row['level_b']}"
        sig = _pstars(row["p_bonferroni"])
        d_abs = abs(row["cohen_d"])
        d_interp = ("grand" if d_abs > 0.8 else
                    ("moyen" if d_abs > 0.5 else
                     ("petit" if d_abs > 0.2 else "nég.")))

        print(f"  {pair:<16} {row['mean_a']:>7.3f}  {row['mean_b']:>7.3f}  "
              f"{row['mean_diff']:>+8.3f}  {row['t']:>7.2f}  {row['df_welch']:>6.1f}  "
              f"{row['p_raw']:>8.4f}  {row['p_bonferroni']:>8.4f}  {sig}  "
              f"{d_abs:>5.2f} ({d_interp})")

    sig_pairs = df[df["significant"]]
    print()
    if len(sig_pairs) > 0:
        print(_ok(f"{len(sig_pairs)}/{len(df)} paires significatives après Bonferroni :"))
        for _, row in sig_pairs.iterrows():
            direction = "↑" if row["mean_a"] > row["mean_b"] else "↓"
            print(f"    ★  {row['level_a']} {direction} {row['level_b']}  "
                  f"Δ={row['mean_diff']:+.3f}  d={abs(row['cohen_d']):.2f}")
    else:
        print(_info("Aucune paire significative après correction Bonferroni."))


def _print_icc_report(r: dict, label: str = "Groove") -> None:
    icc = r["icc"]
    low = r["ci95_low"]
    high = r["ci95_high"]
    p = r["p_value"]
    comp = r["completeness_pct"]
    bar = _bar(max(icc, 0), 1.0, w=20)

    print()
    print(f"  ICC(2,1) — {label}")
    print(f"  {'─'*60}")
    print(f"  Stimuli          : {r['n_stimuli']}")
    print(f"  Participants     : {r['n_raters']}  (k̄ = {r['k_bar']} réponses/stimulus)")
    print(f"  Complétude       : {comp:.1f}%  "
          f"{'⚠ incomplète' if comp < 60 else '✔ correcte'}")
    print()
    print(f"  ICC              : {icc:.3f}  [{low:.3f} – {high:.3f}]  IC95%")
    print(f"  Jauge ICC        : [{bar}]  {icc*100:.1f}%")
    print(f"  F({r['df1']}, {r['df2']})          : {r['F']:.3f}   p = {p:.4f}  {_pstars(p)}")
    print()
    print(f"  Fiabilité        : {r['interpretation'].upper()}")

    zones = [
        (0.90, 1.00, "Excellente  (ICC > 0.90)"),
        (0.75, 0.90, "Bonne       (0.75 – 0.90)"),
        (0.50, 0.75, "Modérée     (0.50 – 0.75)"),
        (0.00, 0.50, "Faible      (< 0.50)"),
    ]
    for lo, hi, lbl in zones:
        marker = "◄" if lo <= icc < hi else " "
        print(f"    {marker}  {lbl}")

    print()
    if icc < 0.50:
        print(_warn("Fiabilité faible — les participants ne s'accordent pas bien."))
    elif icc < 0.75:
        print(_info("Fiabilité modérée — accord partiel entre participants."))
    else:
        print(_ok(f"Fiabilité {r['interpretation']} — les ratings sont reproductibles."))
    print(f"\n  Référence : Koo & Mae (2016), J. Chiropr. Med. 15(2):155–163")


def _print_stimulus_variance_report(df: pd.DataFrame) -> None:
    if df.empty:
        print(_warn("Aucune donnée de variabilité disponible."))
        return
    n = len(df)
    std_mean   = float(df["std"].mean())
    std_median = float(df["std"].median())
    high_var   = df[df["std"] > 1.5]
    low_var    = df[df["std"] < 0.5]

    print()
    print(f"  Stimuli analysés       : {n}")
    print(f"  σ moyen/stimulus       : {std_mean:.3f}  (médiane : {std_median:.3f})")
    print(f"  σ > 1.5 (très ambigu)  : {len(high_var)}")
    print(f"  σ < 0.5 (consensuel)   : {len(low_var)}")
    print(f"  Les 5 les plus ambigus (σ élevé) :")
    for _, row in df.head(5).iterrows():
        bar = _bar(row["std"], 2.0, w=12)
        print(f"    {str(row.get('stimulus_id', row.index)):.<20} "
              f"groove={row['mean']:>5.3f}  σ={row['std']:>5.3f}  "
              f"n={row['n_raters']:>3}  [{bar}]")
    print(f"  Les 5 les plus consensuels (σ faible) :")
    for _, row in df.tail(5).iterrows():
        bar = _bar(row["std"], 2.0, w=12)
        print(f"    {str(row.get('stimulus_id', row.index)):.<20} "
              f"groove={row['mean']:>5.3f}  σ={row['std']:>5.3f}  "
              f"n={row['n_raters']:>3}  [{bar}]")


def _print_final_summary(perm, icc, geom, y_groove, y_complexity,
                         stim_var_df, fig_errors,
                         n_participants=None, n_stimuli=None, n_obs=None):
    print()

    mantel_sig  = perm["significant"]
    icc_val     = icc["icc"]
    agree_mean  = float(np.nanmean(geom["local_agreement"]))
    slope_mean  = float(np.nanmean(np.abs(geom["local_slope"])))
    groove_mu   = float(np.nanmean(y_groove))
    groove_sd   = float(np.nanstd(y_groove))

    print(f"  {'━'*68}")
    print(f"  CORPUS")
    print(f"  {'─'*68}")
    if n_obs is not None:
        print(f"  Observations             : {n_obs}")
    if n_participants is not None:
        print(f"  Participants actifs      : {n_participants}")
    if n_stimuli is not None:
        print(f"  Stimuli couverts         : {n_stimuli}")
    print(f"  Groove μ ± σ             : {groove_mu:.3f} ± {groove_sd:.3f}  "
          f"(médiane = {float(np.nanmedian(y_groove)):.3f})")
    if np.isfinite(y_complexity).sum() > 0:
        print(f"  Complexité μ             : {float(np.nanmean(y_complexity)):.3f}")

    print(f"\n  {'━'*68}")
    print(f"  FIABILITÉ INTER-PARTICIPANTS")
    print(f"  {'─'*68}")
    bar_icc = _hbar(max(icc_val, 0), 0, 1, w=20)
    print(f"  ICC(2,1)                 : {icc_val:.3f}  [{icc['ci95_low']:.3f}–{icc['ci95_high']:.3f}]  "
          f"[{bar_icc}]")
    print(f"  F({icc['df1']}, {icc['df2']})                : {icc['F']:.3f}   "
          f"p = {icc['p_value']:.6f}  {_pstars(icc['p_value'])}")
    print(f"  Interprétation           : {icc['interpretation'].upper()}")
    print(f"  Complétude matrice       : {icc['completeness_pct']:.1f}%")
    var_stimuli_pct = max(icc_val, 0) * 100
    print(f"  Variance expliquée stim  : {var_stimuli_pct:.1f}%  (variance résiduelle : {100-var_stimuli_pct:.1f}%)")

    print(f"\n  {'━'*68}")
    print(f"  GÉOMÉTRIE DE L'ESPACE PERCEPTIF")
    print(f"  {'─'*68}")
    bar_ag = _hbar(agree_mean, 0, 1, w=20)
    bar_sl = _hbar(slope_mean, 0, 0.5, w=20)
    print(f"  Accord local (k={geom['k_effective']})       : "
          f"μ={agree_mean:.3f}  σ={float(np.nanstd(geom['local_agreement'])):.3f}  [{bar_ag}]")
    print(f"  Gradient local |slope|   : "
          f"μ={slope_mean:.3f}  σ={float(np.nanstd(np.abs(geom['local_slope']))):.3f}  [{bar_sl}]")
    verdict_geom = ("✔ espace cohérent et gradué" if agree_mean > 0.7 and slope_mean > 0.15
                    else "~ cohérence partielle" if agree_mean > 0.5
                    else "✗ espace incohérent perceptivement")
    print(f"  Verdict                  : {verdict_geom}")

    print(f"\n  {'━'*68}")
    print(f"  TEST DE MANTEL (structure acoustique × groove perçu)")
    print(f"  {'─'*68}")
    null = np.array(perm.get("permutation_dist", []))
    p95 = float(np.percentile(null, 95)) if len(null) > 0 else float("nan")
    p99 = float(np.percentile(null, 99)) if len(null) > 0 else float("nan")
    z = ((perm['observed_r'] - null.mean()) / (null.std() + 1e-10)
         if len(null) > 0 else float("nan"))
    bar_r = _hbar(abs(perm['observed_r']), 0, 0.4, w=20)
    print(f"  r observé                : {perm['observed_r']:+.4f}  [{bar_r}]")
    print(f"  p-value                  : {perm['p_value']:.4f}  {_pstars(perm['p_value'])}")
    print(f"  z-score                  : {z:+.2f}")
    print(f"  Seuils p95 / p99         : {p95:.4f} / {p99:.4f}")
    print(f"  N paires                 : {perm['n_pairs']}")
    verdict_mantel = ("✔ structure géométrique significative" if mantel_sig
                      else "✗ pas de structure géométrique détectée")
    print(f"  Verdict                  : {verdict_mantel}")

    if stim_var_df is not None and not stim_var_df.empty:
        std_vals = stim_var_df["std"].values
        print(f"\n  {'━'*68}")
        print(f"  VARIABILITÉ INTER-PARTICIPANTS PAR STIMULUS")
        print(f"  {'─'*68}")
        print(f"  σ moyen                  : {std_vals.mean():.3f}  ± {std_vals.std():.3f}")
        print(f"  σ médian                 : {np.median(std_vals):.3f}")
        n_high = int(np.sum(std_vals > 1.5))
        n_low  = int(np.sum(std_vals < 0.5))
        print(f"  Stimuli très ambigus (σ>1.5)  : {n_high}  ({n_high/len(std_vals)*100:.1f}%)")
        print(f"  Stimuli consensuels  (σ<0.5)  : {n_low}  ({n_low/len(std_vals)*100:.1f}%)")

    print(f"\n  {'━'*68}")
    print(f"  FIGURES")
    print(f"  {'─'*68}")
    n_total_figs = 6
    n_ok = n_total_figs - len(fig_errors)
    print(f"  Figures générées         : {n_ok}/{n_total_figs}")
    if fig_errors:
        for f in fig_errors:
            print(f"  ✗ {f}")
    else:
        print(f"  ✔ Toutes les figures générées avec succès")

    print(f"\n  {'━'*68}")
    print(f"  VERDICT GLOBAL")
    print(f"  {'─'*68}")
    checks = [
        ("ICC significatif (p<0.001)",  icc["p_value"] < 0.001),
        ("Accord inter-auditeurs ≥ 0.2", icc_val >= 0.2),
        ("Espace géométriquement cohérent", agree_mean >= 0.7),
        ("Gradient perceptif présent",   slope_mean >= 0.15),
        ("Mantel significatif",          mantel_sig),
        ("Figures OK",                   not fig_errors),
    ]
    n_pass = sum(v for _, v in checks)
    for lbl, ok in checks:
        print(f"  {'✔' if ok else '✗'}  {lbl}")
    print()
    print(f"  Score global : {n_pass}/{len(checks)}")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_fig(filename, fn, fig_dir, fig_errors: list[str], **kwargs):
    """Appel sécurisé d'une fonction figure avec vérification post-sauvegarde."""
    try:
        fn(**kwargs, out_path=fig_dir / filename)
        p = fig_dir / filename
        if p.exists():
            size_kb = p.stat().st_size / 1024
            print(f"  [fig] {filename:<42}  ✔  ({size_kb:.0f} KB)")
        else:
            print(f"  [fig] {filename:<40}  ✔")
    except Exception as e:
        if DEBUG_FIGS:
            raise
        fig_errors.append(filename)
        print(f"  [fig] {filename:<40}  ✗  {type(e).__name__}: {e}")


def _project_2d(X):
    from sklearn.decomposition import PCA
    return PCA(n_components=2).fit_transform(X)