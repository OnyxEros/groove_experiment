"""
analysis/viz/_design.py
========================
Design system partagé pour toutes les figures académiques du projet.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── PALETTE SÉMANTIQUE (PUBLICATION-READY) ──────────────────────────────────
BG          = "#F8F9FC"
PANEL       = "#FFFFFF"
DARK        = "#0F172A"  # Slate 900 (Très élégant pour le texte, évite le noir pur)
MUTED       = "#64748B"  # Slate 500
SUBTLE      = "#CBD5E1"  # Slate 300
GRID        = "#F8FAFC"  # Slate 50
RED_ACCENT  = "#EF4444"
GREEN_OK    = "#10B981"
ORANGE_WARN = "#F59E0B"
BLUE_MAIN   = "#2563EB"
PURPLE      = "#7C3AED"

# ── RCPARAMS ACADÉMIQUES ─────────────────────────────────────────────────────
_RCPARAMS = {
    "font.family":         "sans-serif",
    "font.sans-serif":     ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size":           8.5,
    "text.color":          DARK,
    "axes.labelcolor":     DARK,
    "axes.edgecolor":      SUBTLE,
    "axes.linewidth":      0.7,
    "xtick.color":         MUTED,
    "ytick.color":         MUTED,
    "xtick.major.size":    3,
    "ytick.major.size":    3,
    "xtick.labelsize":     8,
    "ytick.labelsize":     8,
    "figure.facecolor":    PANEL,
    "axes.facecolor":      PANEL,
    "savefig.facecolor":   PANEL,
    # Garantit que le rendu des symboles grecs (comme sigma) utilise un style épuré
    "mathtext.fontset":    "dejavusans", 
}

def apply_rcparams() -> None:
    """Applique la charte graphique globale à Matplotlib."""
    plt.rcParams.update(_RCPARAMS)

def strip_spines(ax: plt.Axes, keep: tuple[str, ...] = ("bottom", "left")) -> None:
    """Supprime les bordures inutiles pour alléger la charge cognitive visuelle."""
    for s, spine in ax.spines.items():
        if s in keep:
            spine.set_color(SUBTLE)
        else:
            spine.set_visible(False)