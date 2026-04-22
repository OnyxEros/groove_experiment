from analysis.core.step import AnalysisStep
from analysis.core.registry import register_step
from analysis.viz.projection import UMAPViz
from analysis.viz.clusters import ClusterViz
from analysis.viz.heatmap import HeatmapViz


@register_step("viz")
class VizStep(AnalysisStep):

    name = "viz"

    def run(self, context):

        rm = context.run_manager

        umap = context.cache["umap_2d"]
        labels = context.cache["clusters"]
        metrics = context.cache["metrics_matrix"]

        UMAPViz().plot_2d(umap, labels, rm.run_dir / "figures/umap_2d.png")
        ClusterViz().plot(umap, labels, rm.run_dir / "figures/clusters.png")
        HeatmapViz().plot(metrics, rm.run_dir / "figures/heatmap.png")

        return context