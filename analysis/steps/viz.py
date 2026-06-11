"""
analysis/steps/viz.py
======================
Step de visualisation — intégré au pipeline via @register_step("viz").

Utilise l'API unifiée .plot(df, path, verbose) des deux classes de figure :
    - GenerativeValidation  (analysis/viz/generative_validation.py)
    - DatasetStructureFigure (analysis/viz/dataset_structure.py)
"""

from pathlib import Path

from analysis.core.step import AnalysisStep
from analysis.core.registry import register_step
from analysis.viz.dataset_structure import DatasetStructureFigure
from analysis.viz.generative_validation import GenerativeValidation


@register_step("viz")
class VizStep(AnalysisStep):

    name = "viz"

    def run(self, context):
        rm = context.run_manager

        # ── Sources de données ────────────────────────────────────────────────
        df_gen  = context.dataset
        df_real = context.cache.get("df_real")

        if df_real is None:
            print(
                "⚠️  [VizStep] 'df_real' introuvable dans le cache. "
                "Utilisation de context.dataset par défaut."
            )
            df_real = df_gen

        # ── Répertoire de sortie ──────────────────────────────────────────────
        fig_dir = rm.run_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # ── Figure 1 : Validation générative ─────────────────────────────────
        try:
            print("\n" + "═"*64)
            print("  🎛  FIGURE 1 — Validation du Moteur Génératif")
            print("═"*64)
            GenerativeValidation().plot(
                df      = df_gen,
                path    = fig_dir / "generative_validation.pdf",
                verbose = True,
            )
            print(f"\n  ✔  generative_validation.pdf  ({len(df_gen)} stimuli)")
        except Exception as e:
            print(f"⚠️  [VizStep] GenerativeValidation échouée : {e}")

        # ── Figure 2 : Structure du dataset ──────────────────────────────────
        try:
            print("\n" + "═"*64)
            print("  📊  FIGURE 2 — Structure du Dataset Réel")
            print("═"*64)
            DatasetStructureFigure().plot(
                df      = df_real,
                path    = fig_dir / "dataset_structure.pdf",
                verbose = True,
            )
            print(f"\n  ✔  dataset_structure.pdf  ({len(df_real)} stimuli)")
        except Exception as e:
            print(f"⚠️  [VizStep] DatasetStructureFigure échouée : {e}")

        print("\n" + "═"*64 + "\n")

        return context