import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from .style import apply_thesis_style

# Palettes dédiées par type de condition
_BACKGROUND_COLORS = {
    "non_musician": "#9CA3AF",
    "amateur":      "#60A5FA",
    "semi_pro":     "#34D399",
    "pro":          "#F59E0B",
}

# Palette qualitative générique pour les autres conditions
_QUALITATIVE = [
    "#2563EB", "#16A34A", "#D97706", "#DC2626",
    "#7C3AED", "#0891B2", "#DB2777", "#65A30D",
]


def _build_palette(series: pd.Series, condition: str) -> dict:
    """
    Construit un dict {valeur: couleur} pour les valeurs présentes
    dans la série. Utilise la palette dédiée si condition='musical_background',
    sinon une palette qualitative automatique.
    """
    values = sorted(series.dropna().unique().tolist(), key=str)

    if condition == "musical_background":
        # Utiliser uniquement les clés présentes
        return {v: _BACKGROUND_COLORS.get(str(v), "#888888") for v in values}
    else:
        return {v: _QUALITATIVE[i % len(_QUALITATIVE)] for i, v in enumerate(values)}


def plot_condition_effects(
    df: pd.DataFrame,
    condition: str,
    target: str = "groove",
    out_path: Path | None = None,
) -> plt.Figure:
    """
    Distribution du score (groove/complexité) selon les conditions.
    Violin plot avec points individuels superposés.

    Compatible seaborn ≥ 0.13 : hue explicite, legend=False.
    """
    if condition not in df.columns:
        if out_path and hasattr(out_path, "name"):
            print(f"  [fig] {out_path.name} ignorée — colonne '{condition}' absente")
        return None

    apply_thesis_style()

    # Nettoyer les valeurs manquantes dans la colonne condition
    df_plot = df[[condition, target]].dropna(subset=[condition])
    if df_plot.empty:
        return None

    df_plot = df_plot.copy()
    df_plot[condition] = df_plot[condition].astype(str)

    palette = _build_palette(df_plot[condition], condition)

    # Ordre des catégories
    if condition == "musical_background":
        order = [k for k in ["non_musician", "amateur", "semi_pro", "pro"]
                 if k in df_plot[condition].values]
    else:
        order = sorted(df_plot[condition].unique().tolist(), key=str)

    fig, ax = plt.subplots(figsize=(max(5, len(order) * 1.4), 4.5))
    fig.patch.set_facecolor("#FAFAFA")

    # Violin — hue explicite pour seaborn ≥ 0.13
    sns.violinplot(
        data=df_plot,
        x=condition,
        y=target,
        hue=condition,        # ← fix deprecation
        order=order,
        hue_order=order,
        palette=palette,
        inner="quartile",
        linewidth=1.2,
        alpha=0.65,
        legend=False,         # ← supprime la légende redondante
        ax=ax,
    )

    # Strip plot
    sns.stripplot(
        data=df_plot,
        x=condition,
        y=target,
        hue=condition,        # ← fix deprecation
        order=order,
        hue_order=order,
        palette={k: "#333333" for k in palette},
        size=3,
        alpha=0.35,
        jitter=True,
        legend=False,
        ax=ax,
    )

    # Annotations : n par groupe
    for i, val in enumerate(order):
        n = int((df_plot[condition] == val).sum())
        ax.text(i, ax.get_ylim()[0] - 0.15, f"n={n}",
                ha="center", va="top", fontsize=8, color="#666666")

    # Labels et titre
    label_map = {
        "non_musician": "Non-musicien",
        "amateur":      "Amateur",
        "semi_pro":     "Semi-pro",
        "pro":          "Pro",
    }
    ax.set_xticklabels(
        [label_map.get(str(v), str(v)) for v in order],
        rotation=15, ha="right", fontsize=9,
    )

    ax.set_title(f"Effet de '{condition}' sur le score de {target}", pad=12)
    ax.set_ylabel(f"Score {target.capitalize()} (1–7)", fontsize=10)
    ax.set_xlabel("")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    sns.despine(left=False, bottom=True, ax=ax)

    # Stats rapides par groupe dans le titre (μ ± σ)
    stats_parts = []
    for val in order:
        grp = df_plot.loc[df_plot[condition] == val, target].dropna()
        if len(grp) >= 2:
            lbl = label_map.get(str(val), str(val))
            stats_parts.append(f"{lbl}: μ={grp.mean():.2f}±{grp.std():.2f}")
    if stats_parts:
        fig.text(0.5, -0.04, "  ·  ".join(stats_parts),
                 ha="center", fontsize=8, color="#6B7280")

    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="#FAFAFA")
        print(f"  [fig] {out_path.name}")

    plt.close(fig)
    return fig
