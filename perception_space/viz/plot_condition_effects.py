import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from .style import apply_thesis_style, COLORS

def plot_condition_effects(
    df: pd.DataFrame, 
    condition: str, 
    target: str = 'groove', 
    out_path: Path | None = None
) -> plt.Figure:
    """
    Affiche la distribution du score (groove/complexité) selon les conditions.
    Violin plot avec points individuels superposés.
    """
    apply_thesis_style()
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Violin plot épuré
    sns.violinplot(
        data=df, x=condition, y=target, ax=ax, 
        palette=COLORS,      # Utilisation de la palette projet
        inner="quartile",    # Montre les quartiles
        linewidth=1.2,       # Bordures fines
        alpha=0.6            # Transparence pour voir les points dessous
    )
    
    # Strip plot (points) avec jitter pour éviter l'écrasement
    sns.stripplot(
        data=df, x=condition, y=target, ax=ax, 
        color="#333333", 
        size=3, 
        alpha=0.4, 
        jitter=True
    )
    
    # Labels et titre
    ax.set_title(f"Effet de '{condition}' sur le score de {target}", pad=15)
    ax.set_ylabel(f"Score {target.capitalize()}")
    ax.set_xlabel(condition.capitalize())
    
    # Nettoyage visuel
    sns.despine(left=True)
    ax.grid(axis='y', linestyle=':', alpha=0.5)

    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"  [fig] {out_path.name}")
    
    plt.close(fig)
    return fig