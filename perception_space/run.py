"""
perception_space/run.py  — VERSION CORRIGÉE POLARITÉ PUSH/PULL
===============================================================
Rétablit le sens physique standard pour le désalignement (P) :
    P > 0  →  Rushing (Avance temporelle du hi-hat)
    P < 0  →  Laid-back (Retard temporel du hi-hat)

Modifié pour redresser la polarité de P et P_mv dès l'extraction 
des embeddings afin de correspondre aux réponses récoltées sur l'ancien 
générateur sans altérer les stimuli d'origine.
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

        print()
        print(f"  Réponses brutes        : {n_raw}")
        print(f"  Doublons supprimés     : {n_dedup}")
        print(f"  Participants fantômes  : {len(ghosts)}  (< {N_MIN_PARTICIPANT_RESPONSES} réponses)")
        if ghosts:
            print(f"  IDs exclus             : {ghosts}")
        print(f"  Réponses conservées    : {n_after}")
        print(f"  Participants actifs    : {pdata['participant_id'].nunique()}")
        print(f"  Stimuli couverts       : {pdata['stimulus_id'].nunique()}")

    # Couverture
    cov = coverage_report(
        pdata, stim_col="stimulus_id",
        participant_col="participant_id" if "participant_id" in pdata.columns else "stimulus_id",
        rating_col="groove",
        n_min=N_MIN_STIMULUS_RESPONSES,
    )
    print()
    print(f"  Couverture stimuli     : {cov['n_covered']}/{cov['n_total']}  ({cov['coverage_pct']:.1f}%)")
    print(f"  Réponses/stimulus      : μ={_fmt(cov['mean_responses'], 1)}  médiane={_fmt(cov['median_responses'], 1)}")
    if cov["n_excluded"] > 0:
        print(_warn(f"{cov['n_excluded']} stimuli exclus (< {N_MIN_STIMULUS_RESPONSES} réponses) : {cov['excluded_stims'][:5]}{'…' if len(cov['excluded_stims']) > 5 else ''}"))

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

    # =========================================================================
    # CORRECTIF DE POLARITÉ DES SIGNES (P / P_mv)
    # =========================================================================
    from config import apply_polarity_fix_array
    from analysis.embeddings.realized import RealizedEmbedding
    X_full = apply_polarity_fix_array(X_full, RealizedEmbedding.COLS)
    # =========================================================================

    n_embed    = X_full.shape[0]
    n_clusters = len(np.unique(clusters))
    print()
    print(f"  Embeddings réalisés    : {n_embed} × {X_full.shape[1]}")
    print(f"  Clusters               : {n_clusters}  (labels : {sorted(np.unique(clusters).tolist())})")
    print(f"  UMAP 2D disponible     : {'oui' if umap_2d_full is not None else 'non (fallback PCA)'}")
    _print_cluster_sizes(clusters)

# ── 3. Alignement ─────────────────────────────────────────────────────────
    print(_sub("3. Alignement embeddings × ratings"))
    validate_perception_df(pdata)
    
    # X et y_groove contiennent 455 lignes (une par réponse brute individuelle)
    X, y_groove, y_complexity = align_embeddings_with_perception(
        X_full, pdata, stim_id_to_row=stim_id_to_row
    )

    valid_sids   = pdata.loc[pdata["stimulus_id"].isin(stim_id_to_row), "stimulus_id"]
    aligned_sids = np.array(pd.unique(valid_sids))
    aligned_rows = np.array([stim_id_to_row[sid] for sid in aligned_sids])

    umap_2d_aligned  = umap_2d_full[aligned_rows] if umap_2d_full is not None else None
    clusters_aligned = clusters[aligned_rows]

    # =========================================================================
    # SÉCURISATION ET CONSTRUTION DE LA MATRICE D'EMBEDDINGS AGRÉGÉE (117, 4)
    # =========================================================================
    # On extrait les 117 lignes uniques de X_full correspondant aux stimuli notés
    X_agg = X_full[aligned_rows]
    # =========================================================================

    print()
    print(f"  Stimuli alignés        : {len(aligned_sids)}/{n_embed}")
    n_missing_align = n_embed - len(aligned_sids)
    if n_missing_align > 0:
        print(_warn(f"{n_missing_align} stimuli sans rating — exclus de l'analyse"))

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
    print()
    print(f"  Groove agrégé          : {n_valid_groove} stimuli valides")
    print(f"    μ = {_fmt(np.nanmean(y_groove_agg))}  ·  σ = {_fmt(np.nanstd(y_groove_agg))}")
    print(f"    min = {_fmt(np.nanmin(y_groove_agg))}  ·  max = {_fmt(np.nanmax(y_groove_agg))}")
    print(f"    médiane = {_fmt(np.nanmedian(y_groove_agg))}")
    _print_groove_distribution(y_groove_agg)

    if np.isfinite(y_complexity_agg).sum() > 0:
        print(f"  Complexité agrégée     : μ={_fmt(np.nanmean(y_complexity_agg))}  σ={_fmt(np.nanstd(y_complexity_agg))}")

    # ── 5. Normalisation + Géométrie locale ───────────────────────────────────
    print(_sub("5. Géométrie locale de l'espace perceptif"))
    # CORRECTIF INTERNE : On normalise et passe la matrice X_agg (117 points) au lieu de X (455 points)
    X_norm = normalize(X_agg)
    groove_geometry = compute_local_geometry(X_norm, y_groove_agg)
    _print_geometry_report(groove_geometry)

    # ── 6. Test de Mantel ─────────────────────────────────────────────────────
    print(_sub("6. Test de permutation (Mantel)"))
    print(f"\n  N permutations         : 1 000")
    # CORRECTIF INTERNE : Le test de Mantel doit tourner sur la matrice des distances de stimuli réels (117)
    perm_result = permutation_test(X_norm, y_groove_agg, n_permutations=1000)
    _print_mantel_report(perm_result)

    # ── 7. Tests statistiques par condition ───────────────────────────────────
    print(_sub("7. Tests statistiques par paramètre génératif"))

    # Reconstruire df agrégé avec paramètres génératifs pour les tests
    try:
        from regression.data.loader import load_aggregated
        df_agg, _, _, _ = load_aggregated(feature_set="design", normalize=False)
        df_test = df_agg.copy()
        
        # =====================================================================
        # CORRECTIF DE POLARITÉ DES SIGNES SUR LE PARADIGME DU DESIGN (P_mv / P)
        # =====================================================================
        # Si la base agrégée de régression contient les variables brutes du design,
        # on applique l'inversion sur P et P_mv pour conserver l'isomorphisme.
        from config import apply_polarity_fix_df
        df_test = apply_polarity_fix_df(df_test)
        # =====================================================================
        
        df_test["groove_mean"] = df_test.get("groove_mean", pd.Series(dtype=float))
        kruskal_results = kruskal_by_condition(df_test, groove_col="groove_mean")
        _print_kruskal_report(kruskal_results)

        # Post-hoc si significatif
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

    # ── 9. Variabilité par stimulus ────────────────────────────────────────────
    print(_sub("9. Variabilité inter-participants par stimulus"))
    stim_var_df = compute_per_stimulus_variance(
        pdata, stim_col="stimulus_id", rating_col="groove"
    )
    _print_stimulus_variance_report(stim_var_df)

    # ── 10. Figures ────────────────────────────────────────────────────────────
    print(_sub("10. Génération des figures"))
    print()
    emb_2d = umap_2d_aligned if umap_2d_aligned is not None else _project_2d(X_norm)

    _safe_fig("umap_groove.png", plot_umap_groove, fig_dir,
              fig_errors=fig_errors,
              embedding=X_norm, groove=y_groove_agg, complexity=y_complexity_agg,
              clusters=clusters_aligned, umap_2d=umap_2d_aligned)

    _safe_fig("cluster_groove.png", plot_cluster_groove, fig_dir,
              fig_errors=fig_errors,
              embedding=X_norm, clusters=clusters_aligned, groove=y_groove_agg)

    _safe_fig("local_geometry_groove.png", plot_local_geometry, fig_dir,
              fig_errors=fig_errors,
              geometry=groove_geometry, embedding_2d=emb_2d, title_prefix="Groove")

    _safe_fig("permutation_test.png", plot_permutation_test, fig_dir,
              fig_errors=fig_errors,
              perm_result=perm_result)

    _safe_fig("icc_summary.png", plot_icc_summary, fig_dir,
              fig_errors=fig_errors,
              icc_groove=icc_res)

    _safe_fig("stimulus_variance.png", plot_per_stimulus_variance, fig_dir,
              fig_errors=fig_errors,
              stim_variance=stim_var_df, stim_col="stimulus_id")

    # ── Rapport final ──────────────────────────────────────────────────────────
    print()
    print(_sep("═"))
    print(_title("✔  SYNTHÈSE PERCEPTION SPACE"))
    print(_sep("═"))
    _print_final_summary(perm_result, icc_res, groove_geometry, y_groove_agg, fig_errors)
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
# BLOCS DE RAPPORT DÉTAILLÉS (INCHANGÉS)
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
    print(f"  Distribution groove (1→7) :")
    for i, (lo, hi, c) in enumerate(zip(bins[:-1], bins[1:], counts)):
        bar = "█" * int(c / scale * 20)
        print(f"  [{lo:.0f}–{hi:.0f})  {bar:<20}  n={c}")

def _print_geometry_report(g: dict) -> None:
    k = g.get("k_effective", "?")
    lm = g["local_mean"]
    ls = g["local_std"]
    la = g["local_agreement"]
    lsl = g["local_slope"]

    print()
    print(f"  k voisins effectif     : {k}")
    print()
    print(f"  {'Métrique':<22} {'μ':>8}  {'σ':>8}  {'min':>8}  {'max':>8}")
    print("  " + "─" * 58)

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
        print(f"  {label:<22} {np.mean(valid):>8.3f}  {np.std(valid):>8.3f}  {np.min(valid):>8.3f}  {np.max(valid):>8.3f}")

    print()
    agree_mean = float(np.nanmean(la))
    if agree_mean > 0.7:
        print(_ok(f"Accord local élevé ({agree_mean:.3f}) — l'espace est perceptivement cohérent"))
    elif agree_mean > 0.5:
        print(_info(f"Accord local modéré ({agree_mean:.3f}) — cohérence partielle"))
    else:
        print(_warn(f"Accord local faible ({agree_mean:.3f}) — forte variabilité perceptive locale"))

    slope_mean = float(np.nanmean(np.abs(lsl)))
    print(_info(f"Gradient local moyen |slope| = {slope_mean:.3f} — " +
                ("gradient significatif dans l'espace" if slope_mean > 0.15 else "espace relativement plat perceptivement")))

def _print_mantel_report(r: dict) -> None:
    obs    = r["observed_r"]
    p      = r["p_value"]
    n_perm = r["n_permutations"]
    n_pairs= r["n_pairs"]
    sig    = r["significant"]

    null = np.array(r["permutation_dist"])
    p95  = float(np.percentile(null, 95)) if len(null) > 0 else float("nan")
    p99  = float(np.percentile(null, 99)) if len(null) > 0 else float("nan")

    print()
    print(f"  r observé              : {obs:+.4f}")
    print(f"  p-value                : {p:.4f}  {_pstars(p)}")
    print(f"  N permutations         : {n_perm}")
    print(f"  N paires analysées     : {n_pairs}  (stimuli × stimuli)")
    print(f"  Distribution nulle p95 : {p95:.4f}")
    print(f"  Distribution nulle p99 : {p99:.4f}")
    print()

    if sig:
        print(_ok(f"Mantel SIGNIFICATIF (p={p:.4f}) — la structure de l'espace latent"))
        print(_arrow("est corrélée avec les différences de groove perçu."))
        print(_info("Les stimuli proches dans l'espace réalisé ont des grooves similaires."))
        if obs > 0.3:
            print(_info(f"r={obs:.3f} : corrélation modérée à forte — signal géométrique réel."))
        elif obs > 0.1:
            print(_info(f"r={obs:.3f} : corrélation faible mais robuste aux permutations."))
    else:
        print(_warn(f"Mantel NON significatif (p={p:.4f}) — la distance dans l'espace"))
        print(_arrow("latent ne prédit pas systématiquement les différences de groove."))
        print(_info("L'espace réalisé (D, S, E, P) ne structure pas la perception de façon"))
        print(_info("géométriquement cohérente — ou le signal est dilué par la variabilité inter-participants."))

    if "warning" in r:
        print(_warn(r["warning"]))

def _print_kruskal_report(df: pd.DataFrame) -> None:
    if df.empty:
        print(_warn("Aucun résultat de test disponible."))
        return

    print()
    print(f"  {'Condition':<12} {'Test':<18} {'Stat':>8}  {'p':>8}  {'Sig':>4}  {'η²':>8}  {'Taille effet':<12}  Normalité")
    print("  " + "─" * 78)

    for _, row in df.iterrows():
        sig = _pstars(row["p_value"])
        norm = "✔ ok" if row.get("normality_ok", True) else "✗ non-normale"
        p_s = f"{row['p_value']:.4f}"
        eta_s = f"{row['eta2']:.3f}"
        stat_s = f"{row['statistic']:.2f}"
        interp = row.get("interpretation", "")
        bar = _bar(row["eta2"], 0.15, w=8)

        print(f"  {row['condition']:<12} {row['test_used']:<18} {stat_s:>8}  {p_s:>8}  {sig}  {eta_s:>8}  {interp:<12}  {norm}")
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
    print(f"  {'Niveaux':<16} {'μ_A':>7}  {'μ_B':>7}  {'Δμ':>8}  {'t':>7}  {'df':>6}  {'p_raw':>8}  {'p_bonf':>8}  {'Sig':>4}  {'|d|':>6}")
    print("  " + "─" * 85)

    for _, row in df.iterrows():
        pair = f"{row['level_a']} vs {row['level_b']}"
        sig = _pstars(row["p_bonferroni"])
        d_abs = abs(row["cohen_d"])
        d_interp = "grand" if d_abs > 0.8 else ("moyen" if d_abs > 0.5 else ("petit" if d_abs > 0.2 else "nég."))

        print(f"  {pair:<16} {row['mean_a']:>7.3f}  {row['mean_b']:>7.3f}  {row['mean_diff']:>+8.3f}  {row['t']:>7.2f}  {row['df_welch']:>6.1f}  {row['p_raw']:>8.4f}  {row['p_bonferroni']:>8.4f}  {sig}  {d_abs:>5.2f} ({d_interp})")

    sig_pairs = df[df["significant"]]
    print()
    if len(sig_pairs) > 0:
        print(_ok(f"{len(sig_pairs)}/{len(df)} paires significatives après Bonferroni :"))
        for _, row in sig_pairs.iterrows():
            direction = "↑" if row["mean_a"] > row["mean_b"] else "↓"
            print(f"    ★  {row['level_a']} {direction} {row['level_b']}  Δ={row['mean_diff']:+.3f}  d={abs(row['cohen_d']):.2f}")
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
    print(f"  Complétude       : {comp:.1f}%  {'⚠ incomplète' if comp < 60 else '✔ correcte'}")
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
    std_mean = float(df["std"].mean())
    std_median = float(df["std"].median())
    high_var = df[df["std"] > 1.5]
    low_var = df[df["std"] < 0.5]

    print()
    print(f"  Stimuli analysés       : {n}")
    print(f"  σ moyen/stimulus       : {std_mean:.3f}  (médiane : {std_median:.3f})")
    print(f"  Les 5 les plus ambigus (σ élevé) :")
    for _, row in df.head(5).iterrows():
        bar = _bar(row["std"], 2.0, w=12)
        print(f"  {str(row.get('stimulus_id', row.index)):.<20} {row['mean']:>9.3f}  {row['std']:>7.3f}  {row['n_raters']:>4}  [{bar}]")

def _print_final_summary(perm, icc, geom, y_groove, fig_errors):
    print()
    mantel_sig = perm["significant"]
    icc_val = icc["icc"]
    agree_mean = float(np.nanmean(geom["local_agreement"]))
    groove_mu = float(np.nanmean(y_groove))

    rows = [
        ("Groove moyen global",  f"{groove_mu:.3f} / 7", "━" if groove_mu > 4 else "↓"),
        ("Mantel r",             f"{perm['observed_r']:+.4f}  p={perm['p_value']:.4f}", "✔ sig." if mantel_sig else "✗ n.s."),
        ("ICC(2,1)",             f"{icc_val:.3f}  [{icc['ci95_low']:.3f}–{icc['ci95_high']:.3f}]", icc["interpretation"]),
        ("Accord local moyen",   f"{agree_mean:.3f}", "bon" if agree_mean > 0.6 else "modéré"),
        ("Figures générées",     f"{6 - len(fig_errors)}/6", "✔" if not fig_errors else f"⚠ {len(fig_errors)} échec(s)"),
    ]
    print(f"  {'Métrique':<26} {'Valeur':<30}  Verdict")
    print("  " + "─" * 66)
    for label, val, verdict in rows:
        print(f"  {label:<26} {val:<30}  {verdict}")

def _safe_fig(filename, fn, fig_dir, fig_errors: list[str], **kwargs):
    try:
        fn(**kwargs, out_path=fig_dir / filename)
        print(f"  {filename:<40}  ✔")
    except Exception as e:
        if DEBUG_FIGS:
            raise
        fig_errors.append(filename)
        print(f"  {filename:<40}  ✗ {e}")

def _project_2d(X):
    from sklearn.decomposition import PCA
    return PCA(n_components=2).fit_transform(X)