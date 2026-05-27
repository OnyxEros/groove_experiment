from analysis.core.step import AnalysisStep
from analysis.core.registry import register_step
from analysis.viz.dataset_structure import DatasetStructureFigure
from analysis.viz.generative_validation import GenerativeValidation


@register_step("viz")
class VizStep(AnalysisStep):
    name = "viz"

    def run(self, context):
        rm = context.run_manager
        
        # 1. On sépare explicitement les deux flux de données
        df_gen = context.dataset  # Données générées stochastiquement
        df_real = context.cache.get("df_real")  # Corpus réel injecté depuis run.py
        
        # Fallback de sécurité si df_real n'a pas été injecté (pour rétrocompatibilité)
        if df_real is None:
            print("⚠️ [VizStep] 'df_real' introuvable dans le cache. Utilisation de context.dataset par défaut.")
            df_real = df_gen

        # Sécurisation : on s'assure que le sous-dossier 'figures' existe
        fig_dir = rm.run_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        # 2. Validation générative -> Analyse du Moteur (utilise les données générées/répétées)
        GenerativeValidation().plot(
            df=df_gen,
            path=fig_dir / "generative_validation.pdf",
            verbose=True,
        )

        # 3. Structure du dataset -> Analyse du Corpus (utilise les données réelles/uniques)
        DatasetStructureFigure().plot(
            df=df_real,
            path=fig_dir / "dataset_structure.pdf",
            verbose=True,
        )

        return context