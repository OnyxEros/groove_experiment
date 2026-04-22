from abc import ABC, abstractmethod


class BaseEmbedding(ABC):

    name: str

    @abstractmethod
    def compute(self, df, cache=None):
        pass
