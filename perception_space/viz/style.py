# perception_space/viz/style.py
import matplotlib.pyplot as plt

def apply_thesis_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica"],
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

# Palette de couleurs "Groove"
COLORS = {
    "groove": "#4157ff",    # Bleu académique
    "complexity": "#ff9800", # Orange
    "low": "#e0e7ff",
    "high": "#312e81"
}
