from analysis.embeddings.structural import StructuralEmbedding
from analysis.embeddings.realized import RealizedEmbedding
from analysis.embeddings.pattern import PatternEmbedding


class EmbeddingManager:

    def __init__(self):
        self.registry = {
            "structural": StructuralEmbedding(),
            "realized": RealizedEmbedding(),
            "pattern": PatternEmbedding(),
        }

    def compute(self, name, df, cache=None):
        if name not in self.registry:
            raise ValueError(f"Unknown embedding: {name}")

        return self.registry[name].compute(df, cache)
