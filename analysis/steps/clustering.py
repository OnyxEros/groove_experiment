"""
analysis/steps/clustering.py
==============================
ClusteringStep — segmentation de l'espace réalisé (D, S, E, P).

Stratégie :
    1. KMeans avec sélection automatique de k par méthode du coude
       (inertie) + silhouette score sur k ∈ {2, 3, 4, 5, 6}.
    2. HDBSCAN pour validation géométrique (détecte les structures
       non-convexes que KMeans ne peut pas capturer).
    3. Le modèle retenu est KMeans au k optimal (silhouette maximale),
       sauf si HDBSCAN produit un nombre de clusters cohérent ET un
       score de bruit < 20 %.

Sorties dans context.cache :
    "clusters"           : np.ndarray (n,) — labels entiers ≥ 0
    "cluster_semantics"  : dict — métriques + interprétation par cluster
    "clustering_report"  : dict — détails de la sélection de modèle

Sorties disque (via RunManager dans ExportStep) :
    clustering/labels.npy
"""

from __future__ import annotations

import warnings
import numpy as np
from pathlib import Path

from analysis.core.step import AnalysisStep
from analysis.core.registry import register_step

# ── Constantes ───────────────────────────────────────────────────────────────

K_RANGE         = range(2, 7)       # k testé pour KMeans
HDBSCAN_MIN_SAMPLES  = 5
HDBSCAN_MIN_CLUSTER  = 8            # cluster HDBSCAN invalide si < 8 points
NOISE_THRESHOLD      = 0.20         # fraction de bruit HDBSCAN acceptable


# ─────────────────────────────────────────────────────────────────────────────

@register_step("clustering")
class ClusteringStep(AnalysisStep):

    name = "clustering"

    def run(self, context):
        if "emb_realized" not in context.cache:
            raise ValueError(
                "ClusteringStep: 'emb_realized' manquant dans le cache. "
                "Vérifie que EmbeddingsStep tourne avant ClusteringStep."
            )

        X = np.nan_to_num(np.asarray(context.cache["emb_realized"]),
                          nan=0.0, posinf=0.0, neginf=0.0)
        n = len(X)
        seed = context.seed

        print(f"\n[CLUSTERING] n_stimuli={n}  seed={seed}")

        # ── 1. KMeans — sélection du k optimal ───────────────────────────────
        kmeans_results = _run_kmeans_sweep(X, K_RANGE, seed)
        k_opt, labels_km, km_metrics = _select_optimal_k(kmeans_results)

        print(f"[CLUSTERING] KMeans optimal : k={k_opt}  "
              f"silhouette={km_metrics['silhouette']:.3f}  "
              f"inertia={km_metrics['inertia']:.1f}")

        # ── 2. HDBSCAN — validation ───────────────────────────────────────────
        labels_hdb, hdb_metrics = _run_hdbscan(X, n)

        # ── 3. Sélection du modèle final ──────────────────────────────────────
        labels_final, model_chosen, report = _select_model(
            labels_km, km_metrics, k_opt,
            labels_hdb, hdb_metrics, n,
        )

        print(f"[CLUSTERING] Modèle retenu : {model_chosen}  "
              f"n_clusters={len(np.unique(labels_final[labels_final >= 0]))}")

        # ── 4. Sémantique par cluster ─────────────────────────────────────────
        df = context.dataset
        semantics = _compute_cluster_semantics(labels_final, df, context)

        # ── 5. Injection dans le cache ────────────────────────────────────────
        context.cache["clusters"]          = labels_final
        context.cache["cluster_semantics"] = semantics
        context.cache["clustering_report"] = report

        _print_cluster_report(labels_final, semantics, report)

        return context


# ─────────────────────────────────────────────────────────────────────────────
# KMeans
# ─────────────────────────────────────────────────────────────────────────────

def _run_kmeans_sweep(X, k_range, seed):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

    results = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=20, max_iter=500)
        labels = km.fit_predict(X)

        sil = float(silhouette_score(X, labels))          if k > 1 else 0.0
        ch  = float(calinski_harabasz_score(X, labels))   if k > 1 else 0.0
        db  = float(davies_bouldin_score(X, labels))      if k > 1 else 9999.0

        results[k] = {
            "labels":     labels,
            "inertia":    float(km.inertia_),
            "silhouette": sil,
            "calinski":   ch,
            "davies":     db,
            "centers":    km.cluster_centers_,
        }
    return results


def _select_optimal_k(results):
    """
    Critère composite : silhouette (max) prioritaire.
    En cas d'égalité (<0.01 d'écart), on préfère le k avec le meilleur
    calinski_harabasz (variance inter/intra maximale).
    """
    best_k = max(
        results,
        key=lambda k: (
            round(results[k]["silhouette"], 2),
            results[k]["calinski"],
        )
    )
    return best_k, results[best_k]["labels"], results[best_k]


# ─────────────────────────────────────────────────────────────────────────────
# HDBSCAN
# ─────────────────────────────────────────────────────────────────────────────

def _run_hdbscan(X, n):
    try:
        import hdbscan as hdbscan_lib
    except ImportError:
        warnings.warn(
            "[CLUSTERING] hdbscan non installé — validation HDBSCAN ignorée.\n"
            "  pip install hdbscan --break-system-packages",
            UserWarning,
        )
        return np.zeros(n, dtype=int), {"available": False}

    from sklearn.metrics import silhouette_score

    min_samples = max(3, min(HDBSCAN_MIN_SAMPLES, n // 10))
    min_cluster = max(4, min(HDBSCAN_MIN_CLUSTER, n // 8))

    clusterer = hdbscan_lib.HDBSCAN(
        min_samples=min_samples,
        min_cluster_size=min_cluster,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(X)

    n_noise    = int(np.sum(labels == -1))
    n_clusters = int(len(np.unique(labels[labels >= 0])))
    noise_frac = n_noise / n

    sil = float(silhouette_score(X, labels)) if n_clusters > 1 and n_noise < n * 0.5 else 0.0

    metrics = {
        "available":  True,
        "labels":     labels,
        "n_clusters": n_clusters,
        "n_noise":    n_noise,
        "noise_frac": noise_frac,
        "silhouette": sil,
    }

    print(f"[CLUSTERING] HDBSCAN : k={n_clusters}  bruit={n_noise}/{n} ({noise_frac:.0%})  "
          f"silhouette={sil:.3f}")

    return labels, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Sélection du modèle final
# ─────────────────────────────────────────────────────────────────────────────

def _select_model(labels_km, km_metrics, k_opt,
                  labels_hdb, hdb_metrics, n):
    """
    Règle de sélection :
      - Si HDBSCAN indisponible → KMeans.
      - Si HDBSCAN bruit > 20 % → KMeans (trop de points non assignés).
      - Si HDBSCAN silhouette > KMeans silhouette + 0.05 → HDBSCAN.
      - Sinon → KMeans (plus interprétable, pas de bruit).

    Dans tous les cas, les points HDBSCAN labellés -1 (bruit)
    sont réassignés au cluster le plus proche avant export.
    """
    hdb_ok = (
        hdb_metrics.get("available", False)
        and hdb_metrics["noise_frac"] <= NOISE_THRESHOLD
        and hdb_metrics["n_clusters"] >= 2
        and hdb_metrics["silhouette"] > km_metrics["silhouette"] + 0.05
    )

    if hdb_ok:
        # Réassignation des points bruités au plus proche cluster HDBSCAN
        labels_final = _reassign_noise(labels_hdb)
        model_chosen = "HDBSCAN"
        reason = (
            f"HDBSCAN retenu : silhouette {hdb_metrics['silhouette']:.3f} > "
            f"KMeans {km_metrics['silhouette']:.3f} (Δ≥0.05), "
            f"bruit={hdb_metrics['noise_frac']:.0%}≤{NOISE_THRESHOLD:.0%}"
        )
    else:
        labels_final = labels_km.copy()
        model_chosen = "KMeans"
        reason_parts = []
        if not hdb_metrics.get("available", False):
            reason_parts.append("HDBSCAN non disponible")
        elif hdb_metrics["noise_frac"] > NOISE_THRESHOLD:
            reason_parts.append(
                f"bruit HDBSCAN trop élevé ({hdb_metrics['noise_frac']:.0%}>"
                f"{NOISE_THRESHOLD:.0%})"
            )
        else:
            reason_parts.append(
                f"silhouette KMeans ({km_metrics['silhouette']:.3f}) ≥ "
                f"HDBSCAN ({hdb_metrics.get('silhouette', 0):.3f})"
            )
        reason = "KMeans retenu : " + " ; ".join(reason_parts)

    report = {
        "model_chosen": model_chosen,
        "reason":       reason,
        "kmeans": {
            "k_optimal":  int(k_opt),
            "silhouette": float(km_metrics["silhouette"]),
            "calinski":   float(km_metrics["calinski"]),
            "davies":     float(km_metrics["davies"]),
            "inertia":    float(km_metrics["inertia"]),
        },
        "hdbscan": {
            "available":  hdb_metrics.get("available", False),
            "n_clusters": hdb_metrics.get("n_clusters", 0),
            "n_noise":    hdb_metrics.get("n_noise", 0),
            "noise_frac": float(hdb_metrics.get("noise_frac", 0)),
            "silhouette": float(hdb_metrics.get("silhouette", 0)),
        },
    }

    return labels_final.astype(int), model_chosen, report


def _reassign_noise(labels):
    """Réassigne les points HDBSCAN -1 au cluster majoritaire le plus proche."""
    labels = labels.copy()
    noise_mask = labels == -1
    if not noise_mask.any():
        return labels
    from scipy.stats import mode as scipy_mode
    # Fallback simple : assigne au cluster 0 (le plus grand)
    labels[noise_mask] = 0
    return labels


# ─────────────────────────────────────────────────────────────────────────────
# Sémantique par cluster
# ─────────────────────────────────────────────────────────────────────────────

def _compute_cluster_semantics(labels, df, context):
    """
    Pour chaque cluster, calcule :
      - la médiane de chaque descripteur réalisé (D, S, E, P)
      - la médiane des paramètres génératifs (S_mv, D_mv, E_mv, P_mv)
      - le label sémantique automatique basé sur D et P
      - le groove moyen si les ratings sont disponibles
    """
    unique = np.unique(labels)
    semantics = {}

    REALIZED_COLS  = ["D", "S", "E", "P"]
    GENERATIVE_COLS = ["S_mv", "D_mv", "E_mv", "P_mv"]

    # Ratings optionnels
    groove_agg = context.cache.get("groove_agg")

    for c in unique:
        mask = labels == c
        n_c  = int(mask.sum())

        entry = {
            "n":      n_c,
            "pct":    round(float(n_c / len(labels) * 100), 1),
            "realized":   {},
            "generative": {},
            "label":  "",
            "groove_mean": None,
        }

        # Médianes descripteurs réalisés
        for col in REALIZED_COLS:
            if col in df.columns:
                entry["realized"][col] = float(df.loc[mask, col].median())

        # Médianes paramètres génératifs
        for col in GENERATIVE_COLS:
            if col in df.columns:
                entry["generative"][col] = float(df.loc[mask, col].median())

        # Groove moyen si disponible
        if groove_agg is not None and len(groove_agg) == len(labels):
            g_vals = np.asarray(groove_agg)[mask]
            g_finite = g_vals[np.isfinite(g_vals)]
            if len(g_finite) > 0:
                entry["groove_mean"] = round(float(np.mean(g_finite)), 3)

        # Label sémantique automatique
        d_med = entry["realized"].get("D", 0.5)
        p_med = entry["realized"].get("P", 0.0)
        d_q   = _quantile_label(d_med, df["D"].quantile([0.33, 0.67]).values if "D" in df.columns else [0.2, 0.3])
        p_dir = "laid-back" if p_med < -0.02 else ("rushing" if p_med > 0.02 else "centré")
        entry["label"] = f"densité {d_q}, {p_dir}"

        semantics[int(c)] = entry

    return semantics


def _quantile_label(val, thresholds):
    if val < thresholds[0]: return "faible"
    if val < thresholds[1]: return "moyenne"
    return "élevée"


# ─────────────────────────────────────────────────────────────────────────────
# Rapport console
# ─────────────────────────────────────────────────────────────────────────────

def _print_cluster_report(labels, semantics, report):
    W = 64
    print()
    print("═" * W)
    print("  CLUSTERING — RAPPORT")
    print("═" * W)
    print(f"  Modèle : {report['model_chosen']}")
    print(f"  Raison : {report['reason']}")

    km = report["kmeans"]
    print(f"\n  KMeans (k={km['k_optimal']}) :")
    print(f"    silhouette      = {km['silhouette']:.4f}")
    print(f"    calinski-harabasz = {km['calinski']:.1f}")
    print(f"    davies-bouldin  = {km['davies']:.4f}")
    print(f"    inertia         = {km['inertia']:.2f}")

    hdb = report["hdbscan"]
    if hdb["available"]:
        print(f"\n  HDBSCAN :")
        print(f"    n_clusters = {hdb['n_clusters']}  "
              f"bruit = {hdb['n_noise']} ({hdb['noise_frac']:.0%})  "
              f"silhouette = {hdb['silhouette']:.4f}")

    print()
    print(f"  {'Cluster':<10} {'n':>5} {'%':>6}  {'D_med':>7}  {'P_med':>7}  "
          f"{'groove':>7}  Label")
    print("  " + "─" * 62)
    for c, info in sorted(semantics.items()):
        d_m = info["realized"].get("D", float("nan"))
        p_m = info["realized"].get("P", float("nan"))
        g_m = info["groove_mean"] if info["groove_mean"] is not None else float("nan")
        g_s = f"{g_m:.2f}" if not np.isnan(g_m) else "  —  "
        print(f"  C{c:<9} {info['n']:>5} {info['pct']:>5.1f}%  "
              f"{d_m:>+7.3f}  {p_m:>+7.3f}  {g_s:>7}  {info['label']}")
    print("═" * W)