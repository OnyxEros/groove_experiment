# regression/models/__init__.py
from regression.models.base import GrooveModel
from regression.models.ridge import RidgeModel
from regression.models.random_forest import RandomForestModel
from regression.models.svr import SVRModel
from regression.models.elasticnet import ElasticNetModel
from regression.models.lmm import LMMModel

__all__ = [
    "GrooveModel",
    "RidgeModel",
    "RandomForestModel",
    "SVRModel",
    "ElasticNetModel",
    "LMMModel",
]


def build_models(seed: int = 42) -> list[GrooveModel]:
    """
    Retourne la liste standard des modèles à entraîner.

    Ordre : Ridge · ElasticNet · SVR · RandomForest · LMM

    Changements v2 :
        + SVRModel       — remplace RF comme modèle non-linéaire principal
                           (meilleure généralisation sur n=100 stimuli)
        + ElasticNetModel — sélection automatique des features redondantes
                           (L1+L2, complément interprétatif de Ridge)
        ~ RandomForest   — conservé pour comparaison et importances MDI,
                           mais non recommandé comme modèle principal (R²CV < 0)
    """
    return [
        RidgeModel(seed=seed),
        ElasticNetModel(seed=seed),
        SVRModel(seed=seed),
        RandomForestModel(seed=seed),
        LMMModel(seed=seed),
    ]