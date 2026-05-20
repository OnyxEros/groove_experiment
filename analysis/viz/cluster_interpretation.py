"""
analysis/viz/cluster_interpretation.py
========================================
Figure publication-ready : radars par cluster + comparaisons.

Corrections :
    - Labels sémantiques dérivés automatiquement depuis les profils réels
    - Normalisation absolue (0–1 sur les valeurs globales du dataset)
      au lieu de relative inter-clusters → C0 n'est plus écrasé
    - Titre radar : "Cluster N — <label sémantique>" sans redondance
    - Valeurs numériques affichées sur chaque sommet du radar
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from pathlib import Path
from math import pi

BG    = "#FAFAFA"
PANEL = "#FFFFFF"
DARK  = "#1A1A2E"
MUTED = "#6B7280"

CLUSTER_COLORS = {
    0: "#2563EB",
    1: "#16A34A",
    2: "#7C3AED",
    3: "#DB2777",
    4: "#CA8A04",
    5: "#0891B2",
}

AXES_FR = {
    "D": "Densité\n(nb d'événements)",
    "I": "Déséquilibre\n(entre voix)",
    "V": "Variabilité\n(timing global)",
    "S": "Syncopation\n(décalage métrique)",
    "E": "Expressivité\n(micro-timing)",
}
KEYS   = list(AXES_FR.keys())
N_AXES = len(KEYS)

_RC = {
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.facecolor":    PANEL,
    "figure.facecolor":  BG,
    "text.color":        DARK,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.facecolor": BG,
}


def _derive_label(profile: dict) -> tuple[str, str]:
    """
    Dérive un label court et une description depuis le profil normalisé
    (valeurs dans [0, 1] par rapport au min/max global du dataset).
    """
    D = profile["D"]
    S = profile["S"]
    E = profile["E"]
    I = profile["I"]

    tags  = []
    descs = []

    # Syncopation
    if S > 0.55:
        tags.append("syncopé")
        descs.append("frappes décalées des temps forts")
    elif S < 0.30:
        tags.append("on-beat")
        descs.append("placement métrique régulier")

    # Densité
    if D > 0.60:
        tags.append("dense")
        descs.append("beaucoup d'événements")
    elif D < 0.35:
        tags.append("sparse")
        descs.append("peu d'événements")

    # Expressivité temporelle
    if E > 0.55:
        tags.append("timing expressif")
        descs.append("micro-timing prononcé")
    elif E < 0.30:
        tags.append("timing serré")
        descs.append("timing quasi-mécanique")

    # Déséquilibre
    if I > 0.65:
        descs.append("fort déséquilibre inter-voix")

    if not tags:
        tags  = ["modéré"]
        descs = ["profil équilibré, aucune dimension saillante"]

    return ", ".join(tags), " · ".join(descs)


class ClusterInterpretation:

    def plot(self, df, labels, path):
        """
        Génère la figure d'interprétation des clusters.

        Args:
            df     : DataFrame des stimuli (colonnes D, I, V, S, E requises)
            labels : np.ndarray (n,) — labels de cluster
            path   : Path ou str — chemin de sauvegarde PNG
        """
        plt.rcParams.update(_RC)

        cluster_data    = self._build_cluster_data(df, labels)
        unique_clusters = sorted(cluster_data.keys())
        n_clusters      = len(unique_clusters)

        angles  = [n / N_AXES * 2 * pi for n in range(N_AXES)]
        angles += angles[:1]

        n_cols = min(3, n_clusters)
        n_rows = (n_clusters + n_cols - 1) // n_cols

        fig = plt.figure(figsize=(16, 5.5 * n_rows + 4.5))
        fig.patch.set_facecolor(BG)

        gs = gridspec.GridSpec(
            n_rows, n_cols, figure=fig,
            left=0.04, right=0.97,
            top=0.91, bottom=0.28,
            hspace=0.65, wspace=0.30,
        )
        gs_bar = gridspec.GridSpec(
            1, 2, figure=fig,
            left=0.06, right=0.97,
            top=0.22, bottom=0.04,
            wspace=0.35,
        )

        # ── Titre ─────────────────────────────────────────────────────────────
        fig.text(0.5, 0.975,
                 f"{n_clusters} familles de patterns rythmiques",
                 ha="center", va="top",
                 fontsize=17, fontweight="bold", color=DARK)
        fig.text(0.5, 0.955,
                 "Scores normalisés sur l'ensemble du dataset "
                 "(0 = minimum global, 1 = maximum global). "
                 "La ligne pointillée marque la médiane (0.5).",
                 ha="center", va="top", fontsize=9, color=MUTED, style="italic")

        # ── Radars ────────────────────────────────────────────────────────────
        for idx, cid in enumerate(unique_clusters):
            ax = fig.add_subplot(gs[idx // n_cols, idx % n_cols], projection="polar")
            self._draw_radar(ax, cid, cluster_data[cid], angles)

        # ── Séparateur ────────────────────────────────────────────────────────
        fig.add_artist(plt.Line2D(
            [0.04, 0.97], [0.25, 0.25],
            transform=fig.transFigure,
            color="#E5E7EB", lw=1.2,
        ))
        fig.text(0.5, 0.245,
                 "Vue comparative — les deux grands axes structurants",
                 ha="center", va="top",
                 fontsize=11, fontweight="bold", color=DARK)

        # ── Barres ────────────────────────────────────────────────────────────
        colors   = [CLUSTER_COLORS.get(c % 10, "#888888") for c in unique_clusters]
        labels_x = [
            f"C{c}\n{cluster_data[c]['label_short']}"
            for c in unique_clusters
        ]

        self._draw_bar_syncopation(
            fig.add_subplot(gs_bar[0]),
            unique_clusters, cluster_data, colors, labels_x,
        )
        self._draw_bar_density(
            fig.add_subplot(gs_bar[1]),
            unique_clusters, cluster_data, colors, labels_x,
        )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=300, bbox_inches="tight", facecolor=BG)
        plt.close()
        print(f"  [fig] {path.name}")

    # ── Build profiles ────────────────────────────────────────────────────────

    def _build_cluster_data(self, df, labels):
        """
        Profils normalisés de façon ABSOLUE :
        chaque valeur = (mean_cluster - min_global) / (max_global - min_global)

        Avantage : C0 conserve sa vraie valeur relative au dataset entier
        et n'est pas artificiellement écrasé par une normalisation inter-clusters.
        """
        # Min / max globaux sur tout le dataset
        global_min, global_max = {}, {}
        for key in KEYS:
            if df is not None and key in df.columns:
                global_min[key] = float(df[key].min())
                global_max[key] = float(df[key].max())
            else:
                global_min[key], global_max[key] = 0.0, 1.0

        cluster_data = {}
        n_total      = len(labels)

        for cid in np.unique(labels):
            mask   = labels == cid
            subset = df[mask] if df is not None else None
            n      = int(mask.sum())
            profile = {"n": n, "pct": 100.0 * n / n_total}

            for key in KEYS:
                if subset is not None and key in subset.columns:
                    raw   = float(subset[key].mean())
                    rng   = global_max[key] - global_min[key]
                    norm  = (raw - global_min[key]) / (rng + 1e-9)
                else:
                    norm  = 0.5
                profile[key] = float(np.clip(norm, 0.0, 1.0))

            label_short, desc       = _derive_label(profile)
            profile["label_short"]  = label_short
            profile["desc"]         = desc
            cluster_data[int(cid)]  = profile

        return cluster_data

    # ── Radar ─────────────────────────────────────────────────────────────────

    def _draw_radar(self, ax, cid, cdata, angles):
        color  = CLUSTER_COLORS.get(cid % 10, "#888888")
        values = [cdata[k] for k in KEYS] + [cdata[KEYS[0]]]

        # Fond de référence à 0.5
        ref = [0.5] * N_AXES + [0.5]
        ax.fill(angles, ref, alpha=0.06, color=MUTED, zorder=1)
        ax.plot(angles, ref, color=MUTED, lw=0.8, linestyle=":", alpha=0.45)

        # Surface + contour
        ax.fill(angles, values, alpha=0.28, color=color, zorder=2)
        ax.plot(angles, values, color=color, lw=2.5, zorder=3)
        ax.scatter(angles[:-1], values[:-1],
                   s=55, color=color, zorder=5,
                   edgecolors="white", linewidths=1.5)

        # Valeurs numériques sur les sommets
        for angle, key in zip(angles[:-1], KEYS):
            v      = cdata[key]
            offset = 0.14
            ax.text(angle, min(v + offset, 1.05), f"{v:.2f}",
                    ha="center", va="center",
                    fontsize=6.5, color=color, fontweight="bold")

        # Axes radar
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(list(AXES_FR.values()), fontsize=7.5, color=DARK)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75])
        ax.set_yticklabels(["¼", "½", "¾"], fontsize=6, color=MUTED)
        ax.yaxis.set_tick_params(labelsize=6)
        ax.grid(color="#E5E7EB", linewidth=0.5, linestyle=":")
        ax.spines["polar"].set_visible(False)

        # Titre : "Cluster N — label sémantique"
        n   = cdata["n"]
        pct = cdata["pct"]
        ax.set_title(
            f"Cluster {cid} — {cdata['label_short']}\n{n} stimuli ({pct:.1f}%)",
            fontsize=9.0, fontweight="bold", color=color,
            pad=14, loc="center",
        )

        # Description sous le radar
        if cdata["desc"]:
            ax.text(0.5, -0.46, cdata["desc"],
                    transform=ax.transAxes,
                    ha="center", va="top",
                    fontsize=7, color=MUTED, style="italic")

    # ── Barres comparatives ───────────────────────────────────────────────────

    def _draw_bar_syncopation(self, ax, cids, cluster_data, colors, labels_x):
        x      = np.arange(len(cids))
        s_vals = [cluster_data[c]["S"] for c in cids]
        bars   = ax.bar(x, s_vals, color=colors, alpha=0.85, width=0.6, zorder=3)

        for bar, v in zip(bars, s_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=DARK,
            )

        ax.axhline(0.5, color=MUTED, lw=1.0, linestyle="--", alpha=0.6,
                   label="Médiane globale (0.5)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels_x, fontsize=7.5, ha="center")
        ax.set_ylabel("Score de syncopation $S$\n(normalisé [0 – 1])", fontsize=8.5)
        ax.set_ylim(0, 1.15)
        ax.set_title("Syncopation par famille", fontsize=10, fontweight="bold",
                     loc="left", pad=6)
        ax.legend(fontsize=7.5, framealpha=0.9)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(-0.5, len(cids) - 0.5)

    def _draw_bar_density(self, ax, cids, cluster_data, colors, labels_x):
        x      = np.arange(len(cids))
        d_vals = [cluster_data[c]["D"] for c in cids]
        e_vals = [cluster_data[c]["E"] for c in cids]
        w      = 0.35

        bars1 = ax.bar(x - w / 2, d_vals, width=w, color=colors, alpha=0.85,
                       label="Densité ($D$)", zorder=3)
        bars2 = ax.bar(x + w / 2, e_vals, width=w, color=colors, alpha=0.45,
                       label="Expressivité ($E$)", hatch="///", zorder=3)

        for bar, v in zip(bars1, d_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=DARK)
        for bar, v in zip(bars2, e_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=DARK)

        ax.set_xticks(x)
        ax.set_xticklabels(labels_x, fontsize=7.5, ha="center")
        ax.set_ylabel("Score normalisé [0 – 1]", fontsize=8.5)
        ax.set_ylim(0, 1.25)
        ax.set_title("Densité et expressivité temporelle par famille",
                     fontsize=10, fontweight="bold", loc="left", pad=6)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(-0.5, len(cids) - 0.5)