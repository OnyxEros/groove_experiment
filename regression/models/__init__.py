# regression/models/__init__.py
from regression.models.base import GrooveModel
from regression.models.ridge import RidgeModel
from regression.models.elasticnet import ElasticNetModel
from regression.models.lmm import LMMModel

__all__ = [
    "GrooveModel",
    "RidgeModel",
    "ElasticNetModel",
    "LMMModel",
]


def build_models(seed: int = 42) -> list[GrooveModel]:
    """
    Retourne la liste des modèles pour l'analyse groove (version mémoire).

    Modèles retenus :
        Ridge      — prédiction CV, coefficients interprétables (β standardisés)
        ElasticNet — sélection L1 : identifie les features non-informatives
        LMM        — inférence statistique avec effets aléatoires participants

    Modèles écartés :
        SVR          — boîte noire injustifiable avec n=21 participants
        RandomForest — R²CV systématiquement négatif sur ce corpus
    """
    return [
        RidgeModel(seed=seed),
        ElasticNetModel(seed=seed),
        LMMModel(seed=seed),
    ]