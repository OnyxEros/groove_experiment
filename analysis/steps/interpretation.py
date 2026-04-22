from analysis.core.step import AnalysisStep
from analysis.core.registry import register_step
from analysis.interpretation.cluster_profiles import ClusterProfileBuilder
from analysis.interpretation.builder import ClusterInterpreter


@register_step("interpretation")
class InterpretationStep(AnalysisStep):

    name = "interpretation"

    def run(self, context):

        df = context.dataset
        cache = context.cache

        # =====================================================
        # CHECK DEPENDENCIES
        # =====================================================

        if "clusters" not in cache:
            raise ValueError(
                "InterpretationStep requires 'clusters'. "
                "Run clustering step first."
            )

        labels = cache["clusters"]

        if len(df) != len(labels):
            raise ValueError(
                f"InterpretationStep mismatch: df={len(df)} vs labels={len(labels)}"
            )

        # =====================================================
        # BUILD PROFILES
        # =====================================================

        try:
            builder = ClusterProfileBuilder()
            profiles = builder.build(df, labels)

            interpreter = ClusterInterpreter()
            semantic = interpreter.interpret(profiles)

        except Exception as e:
            raise RuntimeError(f"InterpretationStep failed: {e}")

        # =====================================================
        # STORE
        # =====================================================

        cache["cluster_profiles"] = profiles
        cache["cluster_semantics"] = semantic

        return context