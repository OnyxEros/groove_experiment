from analysis.core.step import AnalysisStep
from analysis.core.registry import register_step

from analysis.viz.umap import UMAPViz
from analysis.viz.clusters import ClusterViz
from analysis.viz.heatmap import HeatmapViz
from analysis.viz.generative import GenerativeViz
from analysis.viz.thesis_figure import ThesisFigure
from analysis.viz.spaces_figure import SpacesFigure


@register_step("viz")
class VizStep(AnalysisStep):

    name = "viz"

    def run(self, context):

        rm = context.run_manager
        cache = context.cache

        # =====================================================
        # SAFE EXTRACTION (no KeyErrors)
        # =====================================================

        umap_audio = cache.get("umap_audio_2d", None)
        umap_audio_3d = cache.get("umap_audio_3d", None)
        umap_gen = cache.get("umap_gen", None)
        umap_emergent = cache.get("umap_emergent", None)

        labels = cache.get("clusters", None)
        metrics = cache.get("metrics_matrix", None)
        df = context.dataset

        # =====================================================
        # BASIC VIZ (SAFE GUARDS)
        # =====================================================

        if umap_audio is not None and labels is not None:

            UMAPViz().plot(
                umap_audio,
                labels,
                rm.run_dir / "figures/umap_audio.png"
            )

            UMAPViz().plot(
                umap_audio,
                labels,
                rm.run_dir / "figures/umap_swing.png",
                color_by=df["S_mv"],
                title="UMAP colored by S_mv"
            )

            ClusterViz().plot(
                umap_audio,
                labels,
                rm.run_dir / "figures/clusters.png"
            )

        # =====================================================
        # HEATMAP (optional dependency)
        # =====================================================

        if metrics is not None:
            HeatmapViz().plot(
                metrics,
                rm.run_dir / "figures/heatmap.png"
            )

        # =====================================================
        # 3D LATENT SPACE
        # =====================================================

        if umap_audio_3d is not None and labels is not None:

            GenerativeViz().plot_latent_3d(
                umap_audio_3d,
                labels,
                rm.run_dir / "figures/latent_3d.png"
            )

        # =====================================================
        # PARAMETRIC GENERATIVE SPACE
        # =====================================================

        if df is not None:

            GenerativeViz().plot_3d(
                df,
                labels if labels is not None else [0] * len(df),
                rm.run_dir / "figures/generative_3d.png"
            )

        # =====================================================
        # THESIS FIGURE (robust)
        # =====================================================

        if umap_audio is not None and labels is not None:

            ThesisFigure().plot(
                df=df,
                umap_audio=umap_audio,
                umap_gen=umap_gen,
                labels=labels,
                path=rm.run_dir / "figures/thesis_figure.png"
            )

        # =====================================================
        # SPACES FIGURE (core contribution)
        # =====================================================

        if umap_audio is not None and labels is not None:

            SpacesFigure().plot(
                df=df,
                umap_gen=umap_gen,
                umap_emergent=umap_emergent,
                umap_audio=umap_audio,
                labels=labels,
                path=rm.run_dir / "figures/spaces_figure.png"
            )

        return context