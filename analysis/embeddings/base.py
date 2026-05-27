from abc import ABC, abstractmethod


class BaseEmbedding(ABC):
    """
    Un embedding n'est pas juste un vecteur :
    c'est une hypothèse sur la structure du monde.

    Chaque embedding doit répondre à :
        - Que représente chaque dimension ?
        - Quelle distance est perceptuellement significative ?
    """

    name: str

    @abstractmethod
    def compute(self, df, cache=None):
        """
        Retourne un espace vectoriel interprétable.
        """
        pass