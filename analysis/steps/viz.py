from analysis.core.step import AnalysisStep
from analysis.core.registry import register_step
from analysis.viz.spaces_figure import SpacesFigure
from analysis.viz.generative_validation import GenerativeValidation
from analysis.viz.cluster_interpretation import ClusterInterpretation


@register_step("viz")
class VizStep(AnalysisStep):

    name = "viz"

    def run(self, context):
        rm = context.run_manager
        cache = context.cache
        df = context.dataset

        # =====================================================
        # SAFE EXTRACTION (corrected keys)
        # =====================================================
        umap_realized = cache.get("umap_realized", None)
        umap_realized_3d = cache.get("umap_realized_3d", None)
        umap_emergent = cache.get("umap_emergent", None)
        labels = cache.get("clusters", None)
        metrics = cache.get("metrics_matrix", None)

        # =====================================================
        # SPACES FIGURE — main contribution figure
        # =====================================================
        if umap_realized is not None and umap_emergent is not None and labels is not None:
            SpacesFigure().plot(
                df=df,
                umap_emergent=umap_emergent,
                umap_realized=umap_realized,
                labels=labels,
                path=rm.run_dir / "figures/spaces_figure.png"
            )

        # =====================================================
        # GENERATIVE VALIDATION
        # =====================================================
        if umap_realized is not None and umap_emergent is not None and labels is not None:
            GenerativeValidation().plot(
                df=df,
                path=rm.run_dir / "figures/generative_validation.png"
            )

        # =====================================================
        # CLUSTER INTERPRETATION
        # =====================================================
        if umap_realized is not None and umap_emergent is not None and labels is not None:
            ClusterInterpretation().plot(
                df=df,
                labels=labels,
                path=rm.run_dir / "figures/cluster_interpretation.png"
            )
     

        return context