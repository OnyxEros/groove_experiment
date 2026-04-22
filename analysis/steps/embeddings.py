from analysis.core.step import AnalysisStep
from analysis.core.registry import register_step
from analysis.embeddings.manager import EmbeddingManager

@register_step("embeddings")
class EmbeddingsStep(AnalysisStep):

    name = "embeddings"

    def run(self, context):

        if context.dataset is None or len(context.dataset) == 0:
            raise ValueError("EmbeddingsStep: empty dataset")

        if "stim_cache" not in context.cache:
            raise ValueError(
                "EmbeddingsStep requires 'stim_cache' in context.cache"
            )

        manager = EmbeddingManager()
        df = context.dataset

        try:
            emb_structural = manager.compute("structural", df)
            emb_realized = manager.compute("realized", df)

            emb_pattern = manager.compute(
                "pattern",
                df,
                cache=context.cache
            )

        except Exception as e:
            raise RuntimeError(f"EmbeddingsStep failed: {e}")

        context.cache["emb_structural"] = emb_structural
        context.cache["emb_realized"] = emb_realized
        context.cache["emb_pattern"] = emb_pattern

        return context