"""
regression/figures.py — figures publication-ready pour le mémoire
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.framealpha": 0.9,
    "legend.edgecolor": "#cccccc", "figure.dpi": 150,
}
_ACCENT = "#4f6ef7"; _GREEN = "#34d399"; _RED = "#ef4444"; _ORANGE = "#f97316"


def plot_comparison_bar(all_results: dict, out_path: Path) -> None:
    """Barres R² et MAE pour chaque feature_set × modèle."""
    plt.rcParams.update(_RC)
    fs_list = list(all_results.keys())
    models  = ["Ridge", "RandomForest"]
    n       = len(fs_list)
    labels  = {"design": "Design\n(S_mv, D_mv, E)",
                "acoustic": "Acoustic\n(D,I,V,S_real,E_real)", "all": "All"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.subplots_adjust(wspace=0.35, left=0.08, right=0.97, top=0.88, bottom=0.18)
    x, w = np.arange(n), 0.35
    colors = [_ACCENT, _GREEN]

    for ax_i, (metric, ylabel, title) in enumerate([
        ("r2_cv_mean",  "R² (CV 5-fold)",       "A  Variance expliquée (R²)"),
        ("mae_cv_mean", "MAE (CV 5-fold) [1–7]", "B  Erreur absolue moyenne"),
    ]):
        ax = axes[ax_i]
        for m_i, (model, color) in enumerate(zip(models, colors)):
            vals = [all_results.get(fs, {}).get(model, {}).get(metric, 0) for fs in 
fs_list]
            errs = [all_results.get(fs, {}).get(model, 
{}).get(metric.replace("mean","std"), 0) for fs in fs_list]
            bars = ax.bar(x + m_i*w - w/2, vals, w, yerr=errs, capsize=4,
                          color=color, alpha=0.85, label=model,
                          error_kw={"linewidth": 1.2, "ecolor": "#555555"})
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + max(vals)*0.02 if max(vals) else 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([labels.get(fs, fs) for fs in fs_list], fontsize=8)
        ax.set_ylabel(ylabel); ax.set_title(title, pad=8)
        ax.legend(); ax.grid(axis="y", alpha=0.2, linestyle=":", linewidth=0.7)
        if metric == "r2_cv_mean":
            ax.set_ylim(bottom=0); ax.axhline(0, color="#aaaaaa", linewidth=0.8)

    fig.suptitle("Comparaison des modèles de régression groove", fontsize=10, 
weight="bold", y=0.98)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white"); plt.close()
    print(f"  [fig] {out_path.name}")


def plot_coefficients(ridge_coefs: dict, rf_importances: dict, out_path: Path, 
feature_set: str = "") -> None:
    plt.rcParams.update(_RC)
    features = list(ridge_coefs.keys())
    coefs    = np.array([ridge_coefs[f] for f in features])
    imps     = np.array([rf_importances.get(f, 0) for f in features])
    order    = np.argsort(np.abs(coefs))[::-1]
    features, coefs, imps = [features[i] for i in order], coefs[order], imps[order]

    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, len(features)*0.55 + 2)))
    fig.subplots_adjust(wspace=0.45, left=0.14, right=0.97, top=0.88, bottom=0.12)
    y = np.arange(len(features))

    # Ridge
    ax = axes[0]
    colors_r = [_ACCENT if c >= 0 else _RED for c in coefs]
    bars = ax.barh(y, coefs, color=colors_r, alpha=0.85, height=0.6)
    ax.axvline(0, color="#555555", linewidth=1.0)
    ax.set_yticks(y); ax.set_yticklabels(features, fontsize=8)
    ax.set_xlabel("Coefficient (normalisé)"); ax.set_title("A  Ridge — Contributions 
linéaires", pad=8)
    ax.grid(axis="x", alpha=0.2, linestyle=":"); 
    for bar, val in zip(bars, coefs):
        ax.text(val+(0.005 if val>=0 else -0.005), bar.get_y()+bar.get_height()/2,
                f"{val:+.3f}", va="center", ha="left" if val>=0 else "right", fontsize=7)
    ax.legend(handles=[mpatches.Patch(color=_ACCENT, label="↑ groove"),
                        mpatches.Patch(color=_RED, label="↓ groove")], loc="lower right", 
fontsize=7)

    # RF
    ax = axes[1]
    order_rf = np.argsort(imps)[::-1]
    f_rf, i_rf = [features[i] for i in order_rf], imps[order_rf]
    y_rf = np.arange(len(f_rf))
    cmap = plt.cm.get_cmap("YlOrRd", len(f_rf))
    colors_rf = [cmap(i/max(len(f_rf)-1,1)) for i in range(len(f_rf))][::-1]
    bars = ax.barh(y_rf, i_rf, color=colors_rf, alpha=0.85, height=0.6)
    ax.set_yticks(y_rf); ax.set_yticklabels(f_rf, fontsize=8)
    ax.set_xlabel("Importance (MDI)"); ax.set_title("B  RandomForest — Importances", 
pad=8)
    ax.grid(axis="x", alpha=0.2, linestyle=":")
    for bar, val in zip(bars, i_rf):
        ax.text(val+0.002, bar.get_y()+bar.get_height()/2, f"{val:.3f}", va="center", 
ha="left", fontsize=7)

    title_fs = f" [{feature_set}]" if feature_set else ""
    fig.suptitle(f"Contributions des features à la prédiction du groove{title_fs}", 
fontsize=10, weight="bold", y=0.98)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white"); plt.close()
    print(f"  [fig] {out_path.name}")


def plot_shap_summary(shap_values: np.ndarray, X: np.ndarray, features: list, out_path: 
Path) -> None:
    plt.rcParams.update(_RC)
    n_feat   = len(features)
    mean_abs = np.abs(shap_values).mean(axis=0)
    order    = np.argsort(mean_abs)

    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, n_feat*0.6 + 2)))
    fig.subplots_adjust(wspace=0.4, left=0.14, right=0.97, top=0.88, bottom=0.12)
    y = np.arange(n_feat)

    axes[0].barh(y, mean_abs[order], color=_ACCENT, alpha=0.8, height=0.6)
    axes[0].set_yticks(y); axes[0].set_yticklabels([features[i] for i in order], 
fontsize=8)
    axes[0].set_xlabel("|SHAP| moyen"); axes[0].set_title("A  Importance SHAP globale", 
pad=8)
    axes[0].grid(axis="x", alpha=0.2, linestyle=":")

    ax = axes[1]; rng = np.random.default_rng(42)
    for pi, fi in enumerate(order):
        sv = shap_values[:, fi]; xv = X[:, fi]
        jitter = rng.uniform(-0.25, 0.25, len(sv))
        sc = ax.scatter(sv, pi+jitter, c=xv, cmap="coolwarm", s=12, alpha=0.65, 
linewidths=0)
    ax.axvline(0, color="#555555", linewidth=0.9, linestyle="--")
    ax.set_yticks(np.arange(n_feat)); ax.set_yticklabels([features[i] for i in order], 
fontsize=8)
    ax.set_xlabel("Valeur SHAP"); ax.set_title("B  SHAP beeswarm", pad=8)
    ax.grid(axis="x", alpha=0.15, linestyle=":")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Feature (normalisée)", fontsize=7); cbar.ax.tick_params(labelsize=6)

    fig.suptitle("Analyse SHAP — Impact des features sur le groove (RF)", fontsize=10, 
weight="bold", y=0.98)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white"); plt.close()
    print(f"  [fig] {out_path.name}")


def plot_prediction_scatter(y_true, y_pred, model_name, r2, mae, out_path: Path) -> None:
    plt.rcParams.update(_RC)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    fig.subplots_adjust(left=0.14, right=0.95, top=0.88, bottom=0.13)
    ax.scatter(y_true, y_pred, color=_ACCENT, alpha=0.65, s=30, linewidths=0.4, 
edgecolors="white", zorder=3)
    lims = [min(y_true.min(), y_pred.min())-0.2, max(y_true.max(), y_pred.max())+0.2]
    ax.plot(lims, lims, "--", color="#aaaaaa", linewidth=1.2, label="Idéal")
    xs = np.linspace(lims[0], lims[1], 100)
    z = np.polyfit(y_true, y_pred, 1); ax.plot(xs, np.poly1d(z)(xs), color=_RED, 
linewidth=1.5, alpha=0.8, label="Tendance")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Groove observé (mean rating)"); ax.set_ylabel("Groove prédit")
    ax.set_title(f"Prédit vs observé — {model_name}", pad=8)
    ax.grid(alpha=0.18, linestyle=":", linewidth=0.6)
    ax.text(0.04, 0.95, f"R² = {r2:.3f}\nMAE = {mae:.3f}", transform=ax.transAxes,
            va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#cccccc", 
alpha=0.9))
    ax.legend(loc="lower right")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white"); plt.close()
    print(f"  [fig] {out_path.name}")


def plot_cluster_groove(df: pd.DataFrame, labels: np.ndarray, umap_coords: np.ndarray, 
out_path: Path) -> None:
    if "groove_mean" not in df.columns:
        print("  [fig] cluster_groove — skipped (no groove_mean)"); return
    plt.rcParams.update(_RC)
    unique = np.unique(labels); n_clust = len(unique)
    cmap   = plt.cm.get_cmap("tab10", n_clust)
    groove = df["groove_mean"].values

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.subplots_adjust(wspace=0.38, left=0.08, right=0.97, top=0.88, bottom=0.12)

    sc = axes[0].scatter(umap_coords[:,0], umap_coords[:,1], c=groove,
                          cmap="RdYlGn", s=22, alpha=0.75, linewidths=0.3, 
edgecolors="white")
    fig.colorbar(sc, ax=axes[0], fraction=0.035, pad=0.04).set_label("groove_mean", 
fontsize=7.5)
    axes[0].set_xlabel("UMAP dim 1"); axes[0].set_ylabel("UMAP dim 2")
    axes[0].set_title("A  Espace réalisé coloré par groove", pad=8)
    axes[0].grid(alpha=0.18, linestyle=":", linewidth=0.6)

    ax = axes[1]
    cluster_data = [groove[labels == k] for k in unique]
    cluster_xlabels = [f"C{k}\n(n={int((labels==k).sum())})" for k in unique]
    bp = ax.boxplot(cluster_data, patch_artist=True,
                    medianprops=dict(color="#222222", linewidth=2),
                    whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1),
                    flierprops=dict(marker="o", markersize=3, alpha=0.5), widths=0.55)
    for patch, k in zip(bp["boxes"], unique):
        patch.set_facecolor(cmap(k % 10)); patch.set_alpha(0.75)
    ax.set_xticks(np.arange(1, n_clust+1)); ax.set_xticklabels(cluster_xlabels, 
fontsize=8)
    ax.set_ylabel("groove_mean (1–7)"); ax.set_title("B  Distribution groove par cluster", 
pad=8)
    ax.grid(axis="y", alpha=0.2, linestyle=":"); ax.set_ylim(0.5, 7.5)
    ax.axhline(groove.mean(), color=_RED, linewidth=1.2, linestyle="--", alpha=0.7,
               label=f"Moyenne ({groove.mean():.2f})")
    ax.legend(fontsize=7.5)

    fig.suptitle("Perception du groove selon les clusters rythmiques (UMAP réalisé)", 
fontsize=10, weight="bold", y=0.98)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white"); plt.close()
    print(f"  [fig] {out_path.name}")
